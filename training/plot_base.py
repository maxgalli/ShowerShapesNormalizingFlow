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

from train_base import ParquetDataset
from train_base import spline_inn
from train_base import BaseFlow
from train_base import dump_validation_plots


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

    valdataset = ParquetDataset(files=val_file, columns=all_columns, nevs=nevs)
   
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
        print("Necessary base model not found.")
    dump_validation_plots(flow, valdataset, columns, condition_columns, 1, device, outputpath_str, "Final")

if __name__ == "__main__":
    main()
