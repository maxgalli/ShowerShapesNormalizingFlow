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
from joblib import dump, load

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

np.random.seed(42)
torch.manual_seed(42)


class ParquetDataset(Dataset):
    def __init__(self, files, columns, nevs=None, scaler=None):
        self.columns = columns
        # self.df = dd.read_parquet(files, columns=columns, engine='fastparquet')
        self.df = dd.read_parquet(
            files, columns=columns, engine="fastparquet"
        ).compute()

        if nevs is not None:
            self.df = self.df.iloc[:nevs]

        self.df = self.df[self.df["probe_pt"] < 200]
        if scaler is None:
            self.scaler = StandardScaler()
            y = self.df.values
            self.scaler.fit(y)
        else:
            self.scaler = load(scaler)
            y = self.df.values
        y_scaled = self.scaler.transform(y)
        for i, col in enumerate(columns):
            self.df[col] = y_scaled[:, i]

    def dump_scaler(self, path):
        dump(self.scaler, path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return torch.from_numpy(self.df.iloc[index].values).float()


class BaseFlow(flows.Flow):
    """
    Wrapper class around Base Flow for a flow for flow model.
    Harmonises function calls with FlowForFlow model.
    Constructed and used exactly like an nflows.Flow object.
    """

    def log_prob(
        self,
        inputs,
        context=None,
        input_context=None,
        target_context=None,
        inverse=False,
    ):
        """
        log probability of transformed inputs given context, using standard base distribution.
        Inputs:
            inputs: Input Tensor for transformer.
            input_context: Context tensor for samples.
            context: Context tensor for samples if input_context is not defined.
            target_context: Ignored. Exists for interoperability with FLow4Flow models.
            inverse: Ignored. Exists for interoperability with Flow4Flow models.
        """
        context = input_context if input_context is not None else context
        return super(BaseFlow, self).log_prob(inputs, context=context)


def spline_inn(
    inp_dim,
    nodes=128,
    num_blocks=2,
    num_stack=3,
    tail_bound=3.5,
    tails="linear",
    activation=F.relu,
    lu=0,
    num_bins=10,
    context_features=None,
    flow_for_flow=False,
):
    transform_list = []
    for i in range(num_stack):
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                inp_dim,
                nodes,
                num_blocks=num_blocks,
                tail_bound=tail_bound,
                num_bins=num_bins,
                tails=tails,
                activation=activation,
                context_features=context_features,
            )
        ]
        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    if not (flow_for_flow and (num_stack % 2 == 0)):
        # If the above conditions are satisfied then you want to permute back to the original ordering such that the
        # output features line up with their original ordering.
        transform_list = transform_list[:-1]

    return transforms.CompositeTransform(transform_list)


@hydra.main(version_base=None, config_path="config", config_name="cfg0")
def main(cfg):
    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))

    sample = cfg.general.sample
    if sample not in ["data", "mc"]:
        raise ValueError("Sample must be either data or mc")
    calo = cfg.general.calo
    if calo not in ["eb", "ee"]:
        raise ValueError("Calo must be either eb or ee")
    outputpath_base_str = f"{cfg.output.save_dir}/{cfg.output.name}"
    outputpath_base = pathlib.Path(outputpath_base_str)
    outputpath_base.mkdir(parents=True, exist_ok=True)
    with open(f"{outputpath_base_str}/{cfg.output.name}.yaml", "w") as file:
        OmegaConf.save(config=cfg, f=file)
    nevs = cfg.general.nevents

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get training data
    if sample == "data":
        if calo == "eb":
            train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_eb_train.parquet"
            val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_eb_val.parquet"
        elif calo == "ee":
            train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_ee_train.parquet"
            val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_ee_val.parquet"
    elif sample == "mc":
        if calo == "eb":
            train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_eb_train.parquet"
            val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_eb_val.parquet"
        elif calo == "ee":
            train_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_ee_train.parquet"
            val_file = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_ee_val.parquet"

    condition_columns = cfg.general.condition_columns
    columns = cfg.general.columns
    ncond = len(condition_columns)
    all_columns = condition_columns + columns

    base_conf = cfg.base[f"{sample}_{calo}"]

    datadataset = ParquetDataset(files=train_file, columns=all_columns, nevs=nevs)
    datadataset.dump_scaler(f"{outputpath_base_str}/{sample}_{calo}_train_scaler.save")

    valdataset = ParquetDataset(files=val_file, columns=all_columns, nevs=nevs)
    valdataset.dump_scaler(f"{outputpath_base_str}/{sample}_{calo}_val_scaler.save")
   
    dataloader = DataLoader(datadataset, batch_size=base_conf.batch_size, shuffle=True)
    valdataloader = DataLoader(
        valdataset, batch_size=base_conf.batch_size, shuffle=True
    )

    label = f"{sample}_{calo}"
    outputpath_str = f"{outputpath_base_str}/{label}"
    outputpath = pathlib.Path(outputpath_str)
    outputpath.mkdir(parents=True, exist_ok=True)

    # build flow
    flow = BaseFlow(
        spline_inn(
            len(columns),
            nodes=base_conf.nnodes,
            num_blocks=base_conf.nblocks,
            num_stack=base_conf.nstack,
            tail_bound=base_conf.tail_bound,
            activation=getattr(F, base_conf.activation),
            num_bins=base_conf.nbins,
            context_features=ncond,
        ),
        ConditionalDiagonalNormal(shape=[len(columns)], context_encoder=nn.Linear(ncond, 2 * len(columns))),
    )

    # train
    if pathlib.Path(base_conf.load_path).is_file():
        print(f"Loading base_{label} from model: {base_conf.load_path}")
        flow.load_state_dict(
            torch.load(base_conf.load_path, map_location=device)
        )
    else:
        n_epochs = base_conf.nepochs
        print(f"Training {cfg.output.name} on {device} with {n_epochs} epochs and learning rate {base_conf.lr}.")
        optimizer = optim.Adam(flow.parameters(), lr=base_conf.lr)
        train_losses = []
        val_losses = []
        best_vloss = np.inf
        
        for epoch in range(n_epochs):
            tl = []
            vl = []
            flow.to(device)
            flow.train()
            for i, x in enumerate(dataloader):
                x_input = torch.tensor(x[:, ncond:], dtype=torch.float32).to(device)
                x_cond = torch.tensor(x[:, :ncond], dtype=torch.float32).to(device)
                #print(x_input.shape)
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=x_input, context=x_cond).mean()
                #loss = -flow.log_prob(inputs=x_input).mean()
                tl.append(loss.item())
                loss.backward()
                optimizer.step()
                #print(loss.item())
            # validation
            for i, x in enumerate(valdataloader):
                x_input = torch.tensor(x[:, ncond:], dtype=torch.float32).to(device)
                x_cond = torch.tensor(x[:, :ncond], dtype=torch.float32).to(device)
                loss = -flow.log_prob(inputs=x_input, context=x_cond).mean()
                vl.append(loss.item())
           
            epoch_vloss = np.mean(vl)
            train_losses.append(np.mean(tl))
            val_losses.append(epoch_vloss)
            print(f"epoch {epoch + 1}: loss = {np.mean(tl)}, val loss = {epoch_vloss}")

            if epoch_vloss < best_vloss:
                print("Saving model")
                torch.save(flow.state_dict(), f'{outputpath_str}/epoch_{epoch + 1}_valloss_{epoch_vloss:.3f}.pt'.replace('-', 'm'))
                best_vloss = epoch_vloss
            else:
                print(f"Validation loss did not improve from {best_vloss:.3f} to {epoch_vloss:.3f}.")

            # dump validation plots only at the end and at the middle of the training
            if (epoch == n_epochs - 1) or (epoch == n_epochs/2):
                print("Dumping validation plots")
                pairs = [p for p in itertools.combinations(columns, 2)]
                for pair in pairs:
                    c1, c2 = pair
                    print(f"Plotting {pair}")
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].hist2d(valdataset.df[c1], valdataset.df[c2], bins=100, norm=matplotlib.colors.LogNorm())
                    axs[0].set_xlabel(c1)
                    axs[0].set_ylabel(c2)
                    axs[0].set_title("Validation data")
                
                    xcond = torch.tensor(valdataset.df.values[:, :ncond].astype(np.float32)).to(device)
                    nsample = 5
                    with torch.no_grad():
                        #sample = flow.sample(nsample, context=xcond)
                        # move model to cpu to sample
                        #flow = flow.cpu()
                        sample = flow.sample(nsample, context=xcond)
                        # keep only the appropriate columns
                        sample = sample[:, columns.index(c1):columns.index(c2) + 1]
                        x = sample.reshape(sample.shape[0]*sample.shape[1], sample.shape[2])
                        #plt.hist2d(x[:, 0].numpy(), x[:, 1].numpy(), bins=100, range=[[-0.5, 1.5], [-0.2 ,1.2]], norm=matplotlib.colors.LogNorm())
                        axs[1].hist2d(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), bins=100, norm=matplotlib.colors.LogNorm())
                        axs[1].set_xlabel(c1)
                        axs[1].set_ylabel(c2)
                        axs[1].set_title("Sampled data")
                        fig.savefig(f"{outputpath_str}/epoch_{epoch + 1}_{c1}-{c2}.png")

        # plot losses
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.savefig(f"{outputpath_str}/losses.png")


if __name__ == "__main__":
    main()
