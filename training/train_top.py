import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
import glob
import os
import pandas as pd
import dask.dataframe as dd
import sys
import itertools

import torch
from torch import nn
from torch import optim

import nflows
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
import dask.dataframe as dd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseCubicAutoregressiveTransform,
)
from nflows.transforms.autoregressive import (
    MaskedPiecewiseQuadraticAutoregressiveTransform,
)
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nets import ResidualNet
import matplotlib
from sklearn.preprocessing import StandardScaler

from nflows import transforms, flows
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

np.random.seed(42)
torch.manual_seed(42)

from train_base import ParquetDataset
from train_base import spline_inn
from train_base import BaseFlow

from ffflows.models import (
    DeltaFlowForFlow,
    ConcatFlowForFlow,
    DiscreteBaseFlowForFlow,
    DiscreteBaseConditionFlowForFlow,
    NoContextFlowForFlow,
    CustomContextFlowForFlow,
)
from ffflows import distance_penalties
from ffflows.distance_penalties import AnnealedPenalty
from ffflows.data.dist_to_dist import ConditionalDataToData, ConditionalDataToTarget
from ffflows.distance_penalties import BasePenalty


class FFFCustom(flows.Flow):
    """
    MC = left
    Data = right
    forward: MC -> Data
    inverse: Data -> MC
    """
    def __init__(self, transform, flow_mc, flow_data, embedding_net=None):
        super().__init__(transform, flow_mc, embedding_net)
        self.flow_mc = flow_mc
        self.flow_data = flow_data

    def add_penalty(self, penalty_object):
        """Add a distance penaly object to the class."""
        assert isinstance(penalty_object, BasePenalty)
        self.distance_object = penalty_object

    def base_flow_log_prob(self, inputs, input_context, target_context=None, inverse=False):
        if inverse:
            return self.flow_data.log_prob(inputs, input_context)
        else:
            return self.flow_mc.log_prob(inputs, input_context)

    def transform(self, inputs, input_context, target_context=None, inverse=False):
        context = self._embedding_net(input_context)
        transform = self._transform.inverse if inverse else self._transform
        y, logabsdet = transform(inputs, context)

        return y, logabsdet

    def log_prob(self, inputs, input_context, target_context=None, inverse=False):
        converted_input, logabsdet = self.transform(inputs, input_context, target_context, inverse=inverse)
        log_prob = self.base_flow_log_prob(converted_input, input_context, target_context, inverse=inverse)
        dist_pen = -self.distance_object(converted_input, inputs)

        return log_prob + logabsdet + dist_pen

    def batch_transform(self, inputs, input_context, target_context=None, inverse=False):
        # implemented just to keep the same interface
        return self.transform(inputs, input_context, target_context, inverse=inverse)


def get_flow4flow(name, *args, **kwargs):
    f4fdict = {
        "delta": DeltaFlowForFlow,
        "no_context": NoContextFlowForFlow,
        "concat": ConcatFlowForFlow,
        "discretebase": DiscreteBaseFlowForFlow,
        "discretebasecondition": DiscreteBaseConditionFlowForFlow,
        "customcontext": CustomContextFlowForFlow,
    }
    assert (
        name.lower() in f4fdict
    ), f"Currently {f4fdict} is not supported. Choose one of '{f4fdict.keys()}'"

    return f4fdict[name](*args, **kwargs)


def set_penalty(f4flow, penalty, weight, anneal=False):
    if penalty not in ["None", None]:
        if penalty == "l1":
            penalty_constr = distance_penalties.LOnePenalty
        elif penalty == "l2":
            penalty_constr = distance_penalties.LTwoPenalty
        penalty = penalty_constr(weight)
        if anneal:
            penalty = AnnealedPenalty(penalty)
        f4flow.add_penalty(penalty)


def dump_validation_plots_top(model, data_val, mc_val, columns, condition_columns, path, epoch, device, rng=(-5, 5)):
    print("Dumping validation plots")
    ncond = len(condition_columns)
    pairs = [p for p in itertools.combinations(columns, 2)]
    
    inputs = torch.tensor(mc_val.dataset.df.values[:, ncond:]).to(device)
    context_l = torch.tensor(mc_val.dataset.df.values[:, :ncond]).to(device)
    context_r = torch.tensor(data_val.dataset.df.values[:, :ncond]).to(device)
    with torch.no_grad():
        #mc_to_data, _ = model.batch_transform(inputs, context_l, context_r, inverse=True, batch_size=10000)
        mc_to_data, _ = model.batch_transform(inputs, context_l, target_context=context_r, inverse=False)
        #mc_to_data, _ = model.batch_transform(inputs, context_l, target_context=None, inverse=False)

    for pair in pairs:
        c1, c2 = pair
        print(f"Plotting {pair}")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist2d(data_val.dataset.df[c1], data_val.dataset.df[c2], bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
        axs[0].set_xlabel(c1)
        axs[0].set_ylabel(c2)
        axs[0].set_title("Validation data")

        #print("DIOMERDAAAAAA")
        print(mc_to_data.shape)
        index_c1 = columns.index(c1)
        index_c2 = columns.index(c2)
        axs[1].hist2d(mc_to_data[:, index_c1].cpu().numpy(), mc_to_data[:, index_c2].cpu().numpy(), bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
        axs[1].set_xlabel(c1)
        axs[1].set_ylabel(c2)
        axs[1].set_title("MC to data")
        fig.savefig(f"{path}/epoch_{epoch + 1}_{c1}-{c2}.png")
    
    # now plot in bins of condition columns
    nbins = 4
    for column in columns:
        fig, ax = plt.subplots(len(condition_columns), 2, figsize=(10, 5*nbins))
        for row, cond_column in enumerate(condition_columns):
            bins = np.linspace(data_val.dataset.df[cond_column].min(), data_val.dataset.df[cond_column].max(), nbins+1)
            cond_arr = data_val.dataset.df[cond_column].values
            for left_edge, right_edge in zip(bins[:-1], bins[1:]):
                left_edge_label = f"{left_edge:.2f}"
                right_edge_label = f"{right_edge:.2f}"
                print(f"Plotting {column} in bin {left_edge_label} to {right_edge_label} of {cond_column}")
                # plot valdata
                arr = data_val.dataset.df[(data_val.dataset.df[cond_column] > left_edge) & (data_val.dataset.df[cond_column] < right_edge)]
                ax[row, 0].hist2d(arr[cond_column], arr[column], bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
                ax[row, 0].set_xlabel(cond_column)
                ax[row, 0].set_ylabel(column)
                ax[row, 0].set_title(f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}")

                # plot sample
                x = mc_to_data[:, columns.index(column)]
                # concatenate cond_arr_rep and x and keep only values in bin
                arr = np.concatenate((cond_arr.reshape(-1, 1), x.cpu().numpy().reshape(-1, 1)), axis=1)
                arr = arr[(arr[:, 0] > left_edge) & (arr[:, 0] < right_edge)]
                ax[row, 1].hist2d(arr[:, 0], arr[:, 1], bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
                ax[row, 1].set_xlabel(cond_column)
                ax[row, 1].set_ylabel(column)
                ax[row, 1].set_title(f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}")
        fig.savefig(f"{path}/epoch_{epoch + 1}_{column}_condbins.png")


def train_batch_iterate(
    model,
    data_train,
    mc_train,
    data_val,
    mc_val,
    n_epochs,
    learning_rate,
    path,
    columns,
    condition_columns,
    rand_perm_target=False,
    inverse=False,
    loss_fig=True,
    device="cpu",
    gclip=None,
    rng_plt=(-5, 5)
):
    print(f"Training Flow4Flow  on {device} with {n_epochs} epochs and learning rate {learning_rate}, alternating every batch")
    ncond = len(condition_columns)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(data_train) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    if hasattr(model, 'distance_object.set_n_steps'):
        model.distance_object.set_n_steps(num_steps)
    train_losses = []
    val_losses = []
    best_vloss = np.inf
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        tl = []
        vl = []
        for i, (data, mc) in enumerate(zip(data_train, mc_train)):
            model.train()
            #print(data.shape, mc.shape)
            if i % 2 == 0 + 1 * int(inverse):
                print("INVERSE False, from mc to data")
                batch = mc.to(device)
                other_batch = data.to(device)
                inv = False
            else:
                print("INVERSE True, from data to mc")
                batch = data.to(device)
                other_batch = mc.to(device)
                inv = True 
           
            optimizer.zero_grad()
            inputs = batch[:, ncond:]
            context_l = batch[:, :ncond]
            context_r = other_batch[:, :ncond]
            #print(inputs.shape, context_l.shape, context_r.shape)

            loss = -model.log_prob(inputs, input_context=context_l, target_context=context_r, inverse=inv).mean()
            loss.backward()
            if gclip not in ['None', None]:
                clip_grad_norm_(model.parameters(), gclip)
            optimizer.step()
            scheduler.step()
            tl.append(loss.item())
        
        tloss = np.mean(tl)
        train_losses.append(tloss)

        for i, (data, mc) in enumerate(zip(data_val, mc_val)):
            for inv, (batch, other_batch) in enumerate(list(zip([data, mc], [mc, data]))):
                batch = batch.to(device)
                other_batch = other_batch.to(device)
                inputs = batch[:, ncond:]
                context_l = batch[:, :ncond]
                context_r = other_batch[:, :ncond]
                with torch.no_grad():
                    loss = -model.log_prob(inputs, input_context=context_l, target_context=context_r, inverse=inv).mean()
                vl.append(loss.item())
        
        vloss = np.mean(vl)
        val_losses.append(vloss)

        print(f"Epoch {epoch + 1}/{n_epochs} - Train loss: {tloss:.4f} - Val loss: {vloss:.4f}")
        if vloss < best_vloss:
            best_vloss = vloss
            print("Saving model")
            torch.save(model.state_dict(), f"{path}/epoch_{epoch + 1}_valloss_{vloss:.3f}.pt".replace('-', 'm'))
        else:
            print(f"Validation did not improve from {best_vloss:.4f} to {vloss:.4f}")

        # plot losses
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.savefig(f"{path}/losses.png")

        # dump validations plots
        if (epoch == n_epochs - 1) or (epoch == n_epochs/2):
            dump_validation_plots_top(model, data_val, mc_val, columns, condition_columns, path, epoch, device=device, rng=rng_plt)


def train_forward(
    model,
    data_train,
    mc_train,
    data_val,
    mc_val,
    n_epochs,
    learning_rate,
    path,
    columns,
    condition_columns,
    rand_perm_target=False,
    inverse=False,
    loss_fig=True,
    device="cpu",
    gclip=None,
    rng_plt=(-5, 5)
):
    print(f"Training Flow4Flow in fwd mode on {device} with {n_epochs} epochs and learning rate {learning_rate}, alternating every batch")
    ncond = len(condition_columns)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(mc_train) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    if hasattr(model, 'distance_object.set_n_steps'):
        model.distance_object.set_n_steps(num_steps)
    train_losses = []
    val_losses = []
    best_vloss = np.inf
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        tl = []
        vl = []
        for i, batch in enumerate(mc_train):
            model.train()
            batch = batch.to(device)
           
            optimizer.zero_grad()
            inputs = batch[:, ncond:]
            context_l = batch[:, :ncond]
            context_r = batch[:, :ncond]
            #print(inputs.shape, context_l.shape, context_r.shape)

            loss = -model.log_prob(inputs, input_context=context_l, target_context=context_r, inverse=inverse).mean()
            loss.backward()
            if gclip not in ['None', None]:
                clip_grad_norm_(model.parameters(), gclip)
            optimizer.step()
            scheduler.step()
            tl.append(loss.item())
        
        tloss = np.mean(tl)
        train_losses.append(tloss)

        for i, batch in enumerate(mc_val):
            batch = batch.to(device)
            inputs = batch[:, ncond:]
            context_l = batch[:, :ncond]
            context_r = batch[:, :ncond]
            with torch.no_grad():
                loss = -model.log_prob(inputs, input_context=context_l, target_context=context_r, inverse=inverse).mean()
            vl.append(loss.item())
        
        vloss = np.mean(vl)
        val_losses.append(vloss)

        print(f"Epoch {epoch + 1}/{n_epochs} - Train loss: {tloss:.4f} - Val loss: {vloss:.4f}")
        if vloss < best_vloss:
            best_vloss = vloss
            print("Saving model")
            torch.save(model.state_dict(), f"{path}/epoch_{epoch + 1}_valloss_{vloss:.3f}.pt".replace('-', 'm'))
        else:
            print(f"Validation did not improve from {best_vloss:.4f} to {vloss:.4f}")

        # plot losses
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.savefig(f"{path}/losses.png")

        # dump validations plots
        if (epoch == n_epochs - 1) or (epoch == n_epochs/2):
            dump_validation_plots_top(model, data_val, mc_val, columns, condition_columns, path, epoch, device=device, rng=rng_plt)


@hydra.main(version_base=None, config_path="config", config_name="cfg0")
def main(cfg):
    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))

    calo = cfg.general.calo
    if calo not in ["eb", "ee"]:
        raise ValueError("Calo must be either eb or ee")
    outputpath_base_str = f"{cfg.output.save_dir}/{cfg.output.name}"
    outputpath_base = pathlib.Path(outputpath_base_str)
    outputpath_base.mkdir(parents=True, exist_ok=True)
    nevs = cfg.general.nevents
    scaler = cfg.general.scaler

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if calo == "eb":
        data_train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_eb_train.parquet"
        data_val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_eb_val.parquet"
        mc_train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_eb_train.parquet"
        mc_val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_eb_val.parquet"
    elif calo == "ee":
        data_train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_ee_train.parquet"
        data_val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_ee_val.parquet"
        mc_train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_ee_train.parquet"
        mc_val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_ee_val.parquet"

    condition_columns = cfg.general.condition_columns
    columns = cfg.general.columns
    ncond = len(condition_columns)
    all_columns = condition_columns + columns

    # build flow
    label_data = f"data_{calo}"
    data_base_flow = BaseFlow(
        spline_inn(
            len(columns),
            nodes=cfg.base[label_data].nnodes,
            num_blocks=cfg.base[label_data].nblocks,
            num_stack=cfg.base[label_data].nstack,
            tail_bound=cfg.base[label_data].tail_bound,
            activation=getattr(F, cfg.base[label_data].activation),
            num_bins=cfg.base[label_data].nbins,
            context_features=ncond,
        ),
        ConditionalDiagonalNormal(
            shape=[len(columns)], context_encoder=nn.Linear(ncond, 2 * len(columns))
        ),
    )
    label_mc = f"mc_{calo}"
    mc_base_flow = BaseFlow(
        spline_inn(
            len(columns),
            nodes=cfg.base[label_mc].nnodes,
            num_blocks=cfg.base[label_mc].nblocks,
            num_stack=cfg.base[label_mc].nstack,
            tail_bound=cfg.base[label_mc].tail_bound,
            activation=getattr(F, cfg.base[label_mc].activation),
            num_bins=cfg.base[label_mc].nbins,
            context_features=ncond,
        ),
        ConditionalDiagonalNormal(
            shape=[len(columns)], context_encoder=nn.Linear(ncond, 2 * len(columns))
        ),
    )

    top_transformer = cfg[f"top_transformer_{calo}"]
    label = f"top_{calo}"
    outputpath_str = f"{outputpath_base_str}/{label}"
    outputpath = pathlib.Path(outputpath_str)
    outputpath.mkdir(parents=True, exist_ok=True)

    data_base_flow.load_state_dict(
        torch.load(top_transformer.load_path_data, map_location=device)
    )

    mc_base_flow.load_state_dict(
        torch.load(top_transformer.load_path_mc, map_location=device)
    )

    # load data
    d_dataset = ParquetDataset(files=data_train_file, columns=all_columns, scaler=scaler, nevs=nevs)
    val_dataset = ParquetDataset(files=data_val_file, columns=all_columns, scaler=scaler, nevs=nevs)
    mc_dataset = ParquetDataset(files=mc_train_file, columns=all_columns, scaler=scaler, nevs=nevs)
    mc_val_dataset = ParquetDataset(files=mc_val_file, columns=all_columns, scaler=scaler, nevs=nevs)
    # make sure we have the same number of events in data and mc
    min_evs_train = min(len(d_dataset), len(mc_dataset))
    min_evs_val = min(len(val_dataset), len(mc_val_dataset))
    d_dataset.df = d_dataset.df.iloc[:min_evs_train]
    mc_dataset.df = mc_dataset.df.iloc[:min_evs_train]
    val_dataset.df = val_dataset.df.iloc[:min_evs_val]
    mc_val_dataset.df = mc_val_dataset.df.iloc[:min_evs_val]
    
    dataloader = DataLoader(d_dataset, batch_size=cfg.base[f"data_{calo}"].batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.base[f"data_{calo}"].batch_size, shuffle=True)
    mcloader = DataLoader(mc_dataset, batch_size=cfg.base[f"mc_{calo}"].batch_size, shuffle=True)
    val_mcloader = DataLoader(mc_val_dataset, batch_size=cfg.base[f"mc_{calo}"].batch_size, shuffle=True)
    #print(len(d_dataset), len(mc_dataset), len(val_dataset), len(mc_val_dataset))

    if top_transformer.flow4flow != "FFFCustom":
        f4flow = get_flow4flow(
            top_transformer.flow4flow,
            spline_inn(
                len(columns),
                nodes=top_transformer.nnodes,
                num_blocks=top_transformer.nblocks,
                num_stack=top_transformer.nstack,
                tail_bound=top_transformer.tail_bound,
                activation=getattr(F, top_transformer.activation),
                num_bins=top_transformer.nbins,
                context_features=ncond,
                flow_for_flow=True,
            ),
            distribution_right=data_base_flow,
            distribution_left=mc_base_flow,
        )
    else:
        f4flow = FFFCustom(
            spline_inn(
                len(columns),
                nodes=top_transformer.nnodes,
                num_blocks=top_transformer.nblocks,
                num_stack=top_transformer.nstack,
                tail_bound=top_transformer.tail_bound,
                activation=getattr(F, top_transformer.activation),
                num_bins=top_transformer.nbins,
                context_features=ncond,
                flow_for_flow=True,
            ),
            mc_base_flow,
            data_base_flow,
        )
    set_penalty(
        f4flow,
        top_transformer.penalty,
        top_transformer.penalty_weight,
        top_transformer.anneal,
    )
    rng = (-top_transformer.tail_bound, top_transformer.tail_bound)

    # train_data = ConditionalDataToData(d_dataset, mc_dataset)
    # val_data = ConditionalDataToData(val_dataset, mc_val_dataset)
    # train_data.paired()

    direction = top_transformer.direction.lower()
    if pathlib.Path(top_transformer.load_path).is_file():
        print(f"Loading Flow4Flow from model: {top_transformer.load_path}")
        f4flow.load_state_dict(
            torch.load(top_transformer.load_path, map_location=device)
        )
    elif direction == "alternate":
        train_batch_iterate(
            f4flow,
            dataloader,
            mcloader,
            val_dataloader,
            val_mcloader,
            top_transformer.nepochs,
            top_transformer.lr,
            outputpath,
            columns=columns,
            condition_columns=condition_columns,
            device=device,
            gclip=top_transformer.gclip,
            rng_plt = rng,
        )
    elif direction == "forward":
        train_forward(
            f4flow,
            dataloader,
            mcloader,
            val_dataloader,
            val_mcloader,
            top_transformer.nepochs,
            top_transformer.lr,
            outputpath,
            columns=columns,
            condition_columns=condition_columns,
            device=device,
            gclip=top_transformer.gclip,
            rng_plt = rng,
        )

    # dump test datasets
    data_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_{calo}_test.parquet"
    mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_{calo}_test.parquet"

    test_dataset = ParquetDataset(
        files=data_test_file,
        columns=all_columns,
        #scaler=cfg.base[label_data].train_scaler,
    )
    test_mc = ParquetDataset(
        files=mc_test_file,
        columns=all_columns,
        #scaler=cfg.base[label_mc].train_scaler,
    )
    # shuffle test datasets
    test_dataset.df = test_dataset.df.sample(frac=1).reset_index(drop=True)
    test_mc.df = test_mc.df.sample(frac=1).reset_index(drop=True)
    min_evs_test = min(len(test_dataset), len(test_mc))
    test_dataset.df = test_dataset.df.iloc[:min_evs_test]
    test_mc.df = test_mc.df.iloc[:min_evs_test]
   
    from copy import deepcopy
    mc_scaledback_uncorr = deepcopy(test_mc)
    mc_scaledback_uncorr.scale_back()
    mc_scaledback_uncorr = mc_scaledback_uncorr.df
    
    inputs = torch.tensor(test_mc.df.values[:, ncond:]).to(device)
    context_l = torch.tensor(test_mc.df.values[:, :ncond]).to(device)
    context_r = torch.tensor(test_dataset.df.values[:, :ncond]).to(device)
    with torch.no_grad():
        print("Transforming MC to data")
        mc_to_data, _ = f4flow.batch_transform(inputs, context_l, context_r, inverse=False)
        #mc_to_data, _ = f4flow.batch_transform(inputs, context_l, target_context=None, inverse=False, batch_size=10000)
    
    # assign new columns
    for i, col in enumerate(columns):
        test_mc.df[col] = mc_to_data[:, i].cpu().numpy()
    
    # scale back
    print("Scaling back")
    test_mc.scale_back()
    test_dataset.scale_back()

    # plot histograms
    print("Plotting histograms")
    for col in columns:
        fig, ax = plt.subplots()
        ax.hist(test_mc.df[col], bins=100, density=True, label="MC")
        ax.hist(mc_scaledback_uncorr[col], bins=100, density=True, label="MC (uncorr)", alpha=0.5)
        ax.hist(test_dataset.df[col], bins=100, density=True, label="Data", alpha=0.5)
        ax.legend()
        ax.set_xlabel(col)
        ax.set_ylabel("Events/binwidth")
        fig.savefig(f"{outputpath_str}/hist_{col}.png")
        plt.close(fig)

    # dump to file as dataframe for future plotting
    #df = pd.DataFrame(mc_to_data.cpu().numpy(), columns=all_columns)
    print("Dumping to file")
    print(test_mc.df.mean())
    print(test_mc.df.std())
    test_mc.df.to_parquet(f"{outputpath_str}/mc_to_data_{calo}.parquet")


if __name__ == "__main__":
    main()
