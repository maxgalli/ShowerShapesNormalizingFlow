from shutil import which
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from pathlib import Path
import pandas as pd
import concurrent.futures
import json
from sklearn.utils import shuffle
from time import time
import pickle
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client

hep.style.use("CMS")


def main():
    output_dir = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed"
    with open("var_specs.json", "r") as f:
        vars_config = json.load(f)
    columns = [dct["name"] for dct in vars_config]
    columns += [v.replace("probe", "tag") for v in columns if v.startswith("probe")]

    # start a local cluster for parallel processing
    cluster = LocalCluster()
    client = Client(cluster)

    data_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/data/DoubleEG/nominal/*.parquet"
    mc_uncorr_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/mc_uncorr/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/nominal/*.parquet"

    data_df = dd.read_parquet(data_file_pattern, columns=columns, engine='fastparquet')
    mc_uncorr_df = dd.read_parquet(mc_uncorr_file_pattern, columns=columns, engine='fastparquet')

    print("Reading data...")
    data_df = data_df.compute()
    print("Reading MC...")
    mc_uncorr_df = mc_uncorr_df.compute()

    data_df_eb = data_df[np.abs(data_df.probe_eta) < 1.4442]
    data_df_ee = data_df[np.abs(data_df.probe_eta) > 1.56]
    mc_uncorr_df_eb = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) < 1.4442]
    mc_uncorr_df_ee = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) > 1.56]

    output_names = ["data_eb", "data_ee", "mc_eb", "mc_ee"]
    for name, df in zip(output_names, [data_df_eb, data_df_ee, mc_uncorr_df_eb, mc_uncorr_df_ee]):
        print(f"Processing {name}...")
        # common cuts
        df = df[df.probe_r9 < 1.2]

        # shuffle and separate in train, val, test
        nevs = 1000000 if len(df) > 1000000 else int(0.6*len(df))
        df = df.sample(frac=1).reset_index(drop=True)
        df_train_val = df[:nevs]
        df_test = df[nevs:]
        df_train = df_train_val[:int(0.8*nevs)]
        df_val = df_train_val[int(0.8*nevs):]

        # save
        for ext, df_ in zip(["train", "val", "test"], [df_train, df_val, df_test]):
            df_.to_parquet(f"{output_dir}/{name}_{ext}.parquet", engine='fastparquet')


if __name__ == "__main__":
    main()
