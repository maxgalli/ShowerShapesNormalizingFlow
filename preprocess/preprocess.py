import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
import cloudpickle
from copy import deepcopy
import os


# More than one pipeline defined since we don't know yet if the first one will be the best
pipelines = {
    "pipe0": {
        "probe_pt": Pipeline(
            [
                ("box_cox", PowerTransformer(method="box-cox")),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "probe_eta": Pipeline([("scaler", MinMaxScaler((0, 1)))]),
        "probe_phi": Pipeline([("scaler", MinMaxScaler((0, 1)))]),
        "probe_phi": Pipeline([("scaler", MinMaxScaler((0, 1)))]),
        "probe_r9": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log(x + 1e-2),
                        inverse_func=lambda x: np.exp(x) - 1e-2,
                    ),
                ),
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x * 10),
                        inverse_func=lambda x: np.tan(x) / 10,
                    ),
                ),
                ("box_cox", PowerTransformer()),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "probe_s4": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log(x + 1e-2),
                        inverse_func=lambda x: np.exp(x) - 1e-2,
                    ),
                ),
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x * 6),
                        inverse_func=lambda x: (np.tan(x)) / 6,
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "probe_sieie": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log(x * 10 + 1e-1),
                        inverse_func=lambda x: (np.exp(x) - 1e-1) / 10,
                    ),
                ),
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x - 1.25),
                        inverse_func=lambda x: (np.tan(x) + 1.25),
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "probe_sieip": Pipeline([("scaler", MinMaxScaler((0, 1)))]),
        "probe_etaWidth": Pipeline(
            [
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x * 100 - 0.15),
                        inverse_func=lambda x: (np.tan(x) + 0.15) / 100,
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "probe_phiWidth": Pipeline(
            [
                ("box_cox", PowerTransformer(method="box-cox")),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
    },
}


def main():
    output_dir = (
        "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed"
    )
    fig_output = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/figures"
    with open("var_specs.json", "r") as f:
        vars_config = json.load(f)
    columns = [dct["name"] for dct in vars_config]
    columns += [v.replace("probe", "tag") for v in columns if v.startswith("probe")]

    # start a local cluster for parallel processing
    cluster = LocalCluster()
    client = Client(cluster)

    data_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/data/DoubleEG/nominal/*.parquet"
    mc_uncorr_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/mc_uncorr/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/nominal/*.parquet"

    data_df = dd.read_parquet(data_file_pattern, columns=columns, engine="fastparquet")
    mc_uncorr_df = dd.read_parquet(
        mc_uncorr_file_pattern, columns=columns, engine="fastparquet"
    )

    print("Reading data...")
    data_df = data_df.compute()
    print("Reading MC...")
    mc_uncorr_df = mc_uncorr_df.compute()

    data_df_eb = data_df[np.abs(data_df.probe_eta) < 1.4442]
    data_df_ee = data_df[np.abs(data_df.probe_eta) > 1.56]
    mc_uncorr_df_eb = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) < 1.4442]
    mc_uncorr_df_ee = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) > 1.56]


    output_names = ["data_eb", "data_ee", "mc_eb", "mc_ee"]
    for name, df in zip(
        output_names, [data_df_eb, data_df_ee, mc_uncorr_df_eb, mc_uncorr_df_ee]
    ):
        print(f"Processing {name}...")

        # common cuts
        df = df[df.probe_r9 < 1.2]

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        print(f"Train {name}: {len(df_train)}")
        print(f"Test {name}: {len(df_test)}")

        # save
        for ext, df_ in zip(["train", "test"], [df_train, df_test]):
            df_.to_parquet(f"{output_dir}/{name}_{ext}.parquet", engine="fastparquet")

        # plot
        pipelines_copy = deepcopy(pipelines)
        for version in pipelines_copy.keys():
            dct = pipelines_copy[version]
            for var, pipe in dct.items():
                print(f"Plot {name} {version} {var}...")
                cfg = [dct for dct in vars_config if dct["name"] == var][0]
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                arr = df_train[var].values
                ax[0].hist(arr, bins=100, density=True, range=cfg["range"])
                transformed_arr = pipe.fit_transform(arr.reshape(-1, 1))
                ax[1].hist(transformed_arr, bins=100, density=True)
                ax[0].set_title(f"{var} before")
                ax[1].set_title(f"{var} after")
                ax[0].set_xlabel(var)
                ax[1].set_xlabel(var)
                fig.tight_layout()
                for format in ["png", "pdf"]:
                    fig.savefig(
                        os.path.join(
                            fig_output,
                            f"{name}_{version}_{var}.{format}",
                        )
                    )
        
        # save pipelines
        # we save one for each combination sample/calo, containing all the versions
        # this way when we load them durtig the training the transformations are already fitted
        with open(os.path.join(output_dir, f"pipelines_{name}.pkl"), "wb") as f:
            cloudpickle.dump(pipelines_copy, f)


if __name__ == "__main__":
    main()
