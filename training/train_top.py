import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import pathlib
import glob
import os
import pandas as pd
import sys
import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from nflows.distributions.normal import ConditionalDiagonalNormal
import matplotlib
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import ParquetDataset
from utils import spline_inn
from utils import BaseFlow
from utils import FFFCustom
from utils import get_flow4flow
from utils import set_penalty
from utils import train_batch_iterate
from utils import train_forward

np.random.seed(42)
torch.manual_seed(42)


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
    writer = SummaryWriter(outputpath_str)

    data_base_flow.load_state_dict(
        torch.load(top_transformer.load_path_data, map_location=device)
    )

    mc_base_flow.load_state_dict(
        torch.load(top_transformer.load_path_mc, map_location=device)
    )

    # load data
    d_dataset = ParquetDataset(
        files=data_train_file, columns=all_columns, scaler=scaler, nevs=nevs
    )
    val_dataset = ParquetDataset(
        files=data_val_file, columns=all_columns, scaler=scaler, nevs=nevs
    )
    mc_dataset = ParquetDataset(
        files=mc_train_file, columns=all_columns, scaler=scaler, nevs=nevs
    )
    mc_val_dataset = ParquetDataset(
        files=mc_val_file, columns=all_columns, scaler=scaler, nevs=nevs
    )
    # make sure we have the same number of events in data and mc
    min_evs_train = min(len(d_dataset), len(mc_dataset))
    min_evs_val = min(len(val_dataset), len(mc_val_dataset))
    d_dataset.df = d_dataset.df.iloc[:min_evs_train]
    mc_dataset.df = mc_dataset.df.iloc[:min_evs_train]
    val_dataset.df = val_dataset.df.iloc[:min_evs_val]
    mc_val_dataset.df = mc_val_dataset.df.iloc[:min_evs_val]

    dataloader = DataLoader(
        d_dataset, batch_size=cfg.base[f"data_{calo}"].batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.base[f"data_{calo}"].batch_size, shuffle=True
    )
    mcloader = DataLoader(
        mc_dataset, batch_size=cfg.base[f"mc_{calo}"].batch_size, shuffle=True
    )
    val_mcloader = DataLoader(
        mc_val_dataset, batch_size=cfg.base[f"mc_{calo}"].batch_size, shuffle=True
    )
    # print(len(d_dataset), len(mc_dataset), len(val_dataset), len(mc_val_dataset))

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
            rng_plt=rng,
            writer=writer,
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
            rng_plt=rng,
            writer=writer,
        )

    # dump test datasets
    data_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_{calo}_test.parquet"
    mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_{calo}_test.parquet"

    test_dataset = ParquetDataset(
        files=data_test_file,
        columns=all_columns,
        # scaler=cfg.base[label_data].train_scaler,
    )
    test_mc = ParquetDataset(
        files=mc_test_file,
        columns=all_columns,
        # scaler=cfg.base[label_mc].train_scaler,
    )
    # shuffle test datasets
    test_dataset.df = test_dataset.df.sample(frac=1).reset_index(drop=True)
    test_mc.df = test_mc.df.sample(frac=1).reset_index(drop=True)
    min_evs_test = min(len(test_dataset), len(test_mc))
    test_dataset.df = test_dataset.df.iloc[:min_evs_test]
    test_mc.df = test_mc.df.iloc[:min_evs_test]

    from copy import deepcopy

    mc_uncorr = deepcopy(test_mc)
    mc_uncorr_df = mc_uncorr.df
    mc_uncorr_scaledback = deepcopy(test_mc)
    mc_uncorr_scaledback.scale_back()
    mc_scaledback_uncorr = mc_uncorr_scaledback.df

    inputs = torch.tensor(test_mc.df.values[:, ncond:]).to(device)
    context_l = torch.tensor(test_mc.df.values[:, :ncond]).to(device)
    context_r = torch.tensor(test_dataset.df.values[:, :ncond]).to(device)
    with torch.no_grad():
        print("Transforming MC to data")
        mc_to_data, _ = f4flow.batch_transform(
            inputs, context_l, context_r, inverse=False
        )
        # mc_to_data, _ = f4flow.batch_transform(inputs, context_l, target_context=None, inverse=False, batch_size=10000)

    # assign new columns
    for i, col in enumerate(columns):
        test_mc.df[col] = mc_to_data[:, i].cpu().numpy()

    print("Plotting histograms not scaled back")
    for col in columns:
        fig, ax = plt.subplots()
        ax.hist(test_mc.df[col], bins=100, density=True, label="MC")
        ax.hist(
            mc_uncorr_df[col], bins=100, density=True, label="MC (uncorr)", alpha=0.5
        )
        ax.hist(test_dataset.df[col], bins=100, density=True, label="Data", alpha=0.5)
        ax.legend()
        ax.set_xlabel(col)
        ax.set_ylabel("Events/binwidth")
        fig.savefig(f"{outputpath_str}/hist_unscaled_{col}.png")
        plt.close(fig)

    # scale back
    print("Scaling back")
    test_mc.scale_back()
    test_dataset.scale_back()

    # plot histograms
    print("Plotting histograms")
    for col in columns:
        fig, ax = plt.subplots()
        ax.hist(test_mc.df[col], bins=100, density=True, label="MC")
        ax.hist(
            mc_scaledback_uncorr[col],
            bins=100,
            density=True,
            label="MC (uncorr)",
            alpha=0.5,
        )
        ax.hist(test_dataset.df[col], bins=100, density=True, label="Data", alpha=0.5)
        ax.legend()
        ax.set_xlabel(col)
        ax.set_ylabel("Events/binwidth")
        fig.savefig(f"{outputpath_str}/hist_{col}.png")
        plt.close(fig)

    # dump to file as dataframe for future plotting
    # df = pd.DataFrame(mc_to_data.cpu().numpy(), columns=all_columns)
    print("Dumping to file")
    print(test_mc.df.mean())
    print(test_mc.df.std())
    test_mc.df.to_parquet(f"{outputpath_str}/mc_to_data_{calo}.parquet")


if __name__ == "__main__":
    main()
