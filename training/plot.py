from shutil import which
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from pathlib import Path
import pandas as pd
import concurrent.futures
import json
import xgboost as xgb
from sklearn.utils import shuffle
from time import time
import pickle
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from scipy.stats import wasserstein_distance

hep.style.use("CMS")


def plot_variable(data_array, mc_array, mc_array_uncorr, variable_conf, output_dir, subdetector):
    name = variable_conf["name"]
    title = variable_conf["title"] + "_" + subdetector
    x_label = variable_conf["x_label"]
    bins = variable_conf["bins"]
    range = variable_conf["range"]

    # specific ranges for EB and EE
    if name == "probe_sieie" and subdetector == "EE":
        range = [0.005, 0.04]

    print("Plotting variable: {}".format(name))

    fig, (up, down) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": (2, 1)},
        sharex=True,
        )
    mc_hist, mc_bins, _ = up.hist(
        mc_array[name],
        bins=bins,
        range=range,
        histtype="step",
        label=f"MC - {subdetector}",
        density=True,
        weights=mc_array["weight_clf"],
        linewidth=2,
        color="r"
    )
    mc_uncorr_hist, mc_uncorr_bins, _ = up.hist(
        mc_array_uncorr[name],
        bins=bins,
        range=range,
        histtype="step",
        label=f"MC - {subdetector} (uncorr.)",
        density=True,
        weights=mc_array_uncorr["weight_clf"],
        linewidth=2,
        color="b"
    )
    data_hist, data_bins = np.histogram(
        data_array[name], bins=bins, range=range, density=True
    )
    data_centers = (data_bins[1:] + data_bins[:-1]) / 2
    up.plot(
        data_centers,
        data_hist,
        label=f"Data - {subdetector}",
        color="k",
        marker="o",
        linestyle="",
    )
    down.plot(
        data_centers,
        data_hist / mc_hist,
        color="r",
        marker="o",
        linestyle="",
    )
    down.plot(
       data_centers,
       data_hist / mc_uncorr_hist,
       color="b",
       marker="o",
       linestyle="",
    )
   
    if name in ["probe_pfChargedIsoPFPV", "probe_pfPhoIso03"]:
        up.set_yscale("log")
    if name == "probe_sieip":
        ticks = [-0.0002, -0.0001, 0, 0.0001, 0.0002]
        down.set_xticks(ticks)
        down.set_xticklabels(ticks)
    down.set_xlabel(x_label)
    up.set_ylabel("Events / BinWidth")
    down.set_ylabel("Data / MC")
    down.set_xlim(range[0], range[1])
    down.set_ylim(0.5, 1.5)
    down.axhline(1, color="grey", linestyle="--", )
    y_minor_ticks = np.arange(0.5, 1.5, 0.1)
    down.set_yticks(y_minor_ticks, minor=True)
    down.grid(True, alpha=0.4, which="minor")
    up.legend()
    fig.savefig(output_dir + "/" + name + "_" + subdetector + ".pdf", bbox_inches="tight")
    fig.savefig(output_dir + "/" + name + "_" + subdetector + ".png", bbox_inches="tight")
    hep.cms.label(loc=0, data=True, llabel="Work in Progress", rlabel="", ax=up, pad=.05)
    plt.close(fig)


def clf_reweight(df_mc, df_data, clf_name, n_jobs=1, cut=None):
    """
    See https://github.com/maxgalli/qRC/blob/TO_MERGE/quantile_regression_chain/syst/qRC_systematics.py#L91-L107
    """
    features = ['probe_pt','probe_fixedGridRhoAll','probe_eta','probe_phi']
    try:
        clf = pickle.load(open(f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/{clf_name}.pkl", "rb"))
        print("Loaded classifier from file {}.pkl".format(clf_name))
    except FileNotFoundError:
        clf = xgb.XGBClassifier(learning_rate=0.05,n_estimators=500,max_depth=10,gamma=0,n_jobs=n_jobs)
        if cut is not None:
            X_data = df_data.query(cut, engine='python').sample(min(min(df_mc.query(cut, engine='python').index.size,df_data.query(cut, engine='python').index.size), 1000000)).loc[:,features].values
            X_mc = df_mc.query(cut, engine='python').sample(min(min(df_mc.query(cut, engine='python').index.size,df_data.query(cut, engine='python').index.size), 1000000)).loc[:,features].values
        else:
            X_data = df_data.sample(min(min(df_mc.index.size,df_data.index.size), 1000000)).loc[:,features].values
            X_mc = df_mc.sample(min(min(df_mc.index.size,df_data.index.size), 1000000)).loc[:,features].values
        X = np.vstack([X_data,X_mc])
        y = np.vstack([np.ones((X_data.shape[0],1)),np.zeros((X_mc.shape[0],1))])
        X, y = shuffle(X,y)

        start = time()
        clf.fit(X,y)
        print("Classifier trained in {:.2f} seconds".format(time() - start))
        with open(f"{clf_name}.pkl", "wb") as f:
            pickle.dump(clf, f)
    eps = 1.e-3
    return np.apply_along_axis(lambda x: x[1]/(x[0]+eps), 1, clf.predict_proba(df_mc.loc[:,features].values))


def main():
    calo = "eb"
    #output_dir = "/eos/home-g/gallim/www/plots/Hgg/NormFlowsCorrections/test"
    output_dir = "/eos/home-g/gallim/www/plots/Hgg/NormFlowsCorrections/cfg16"
    with open("/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
    features = ['probe_pt','probe_fixedGridRhoAll','probe_eta','probe_phi']
    #columns = ["probe_r9", "probe_s4"]
    columns = ["probe_r9", "probe_s4", "probe_sieip", "probe_sieie"]
    var_config_to_plot = [dct for dct in vars_config if dct["name"] in columns]
    print(var_config_to_plot)

    # start a local cluster for parallel processing
    cluster = LocalCluster()
    client = Client(cluster)

    data_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/data_{calo}_test.parquet"
    mc_uncorr_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/preprocessed/mc_{calo}_test.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg6/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg7/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg7_1/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg7_3/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg9/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg10/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg11/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg14b/top_{calo}/mc_to_data_{calo}.parquet"
    #mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg15/top_{calo}/mc_to_data_{calo}.parquet"
    mc_test_file = f"/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/training/output/cfg16/top_{calo}/mc_to_data_{calo}.parquet"

    data_test_df = dd.read_parquet(data_test_file, columns=columns + features, engine='fastparquet').compute()
    mc_uncorr_test_df = dd.read_parquet(mc_uncorr_test_file, columns=columns + features, engine='fastparquet').compute()
    mc_test_df = dd.read_parquet(mc_test_file, columns=columns, engine='fastparquet').compute()
    print(len(data_test_df), len(mc_uncorr_test_df), len(mc_test_df))

    # stuck to mc_test_df the columns needed for the reweighting
    mc_test_df = pd.concat([mc_test_df, mc_uncorr_test_df[["probe_pt", "probe_eta", "probe_phi", "probe_fixedGridRhoAll"]]], axis=1)

    print("Calculating weights...")
    for df in [mc_uncorr_test_df, mc_test_df]:
        df["weight_clf"] = clf_reweight(df, data_test_df, "clf_{}".format(calo), n_jobs=10, cut=None)

    for var_conf in var_config_to_plot:
        plot_variable(data_test_df, mc_test_df, mc_uncorr_test_df, var_conf, output_dir, calo)

    # plot unweighted distributions
    print("Plotting unweighted distributions...")
    for var_conf in var_config_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for df, label in zip([data_test_df, mc_uncorr_test_df, mc_test_df], ["data", "mc_uncorr", "mc"]):
            ax.hist(df[var_conf["name"]], bins=var_conf["bins"], histtype="step", label=label, density=True)
        ax.set_xlabel(var_conf["title"])
        ax.set_ylabel("Normalized entries")
        ax.legend()
        fig.savefig(f"{output_dir}/unweighted_{var_conf['name']}_{calo}.png")
        fig.savefig(f"{output_dir}/unweighted_{var_conf['name']}_{calo}.pdf")

if __name__ == "__main__":
    main()
