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
        clf = pickle.load(open(f"{clf_name}.pkl", "rb"))
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
    output_dir = "/eos/home-g/gallim/www/plots/Hgg/NormFlowsCorrections/cqr_plots"
    with open("var_specs.json", "r") as f:
        vars_config = json.load(f)
    columns = [dct["name"] for dct in vars_config]
    columns += [v.replace("probe", "tag") for v in columns if v.startswith("probe")]

    # start a local cluster for parallel processing
    cluster = LocalCluster()
    client = Client(cluster)

    data_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/data/DoubleEG/nominal/*.parquet"
    mc_uncorr_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/mc_uncorr/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/nominal/*.parquet"
    mc_corr_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/mc_corr/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/nominal/*.parquet"

    data_df = dd.read_parquet(data_file_pattern, columns=columns, engine='fastparquet')
    mc_uncorr_df = dd.read_parquet(mc_uncorr_file_pattern, columns=columns, engine='fastparquet')
    mc_corr_df = dd.read_parquet(mc_corr_file_pattern, columns=columns, engine='fastparquet')

    data_df = data_df.compute()
    mc_uncorr_df = mc_uncorr_df.compute()
    mc_corr_df = mc_corr_df.compute()

    # Thomas stuff, as it is
    eb_cut_clf = "abs(probe_eta)<1.4442 and tag_pt>40 and probe_pt>20 and mass>80 and mass<100 and abs(tag_eta)<2.5" # passEleVeto == 0
    ee_cut_clf = "abs(probe_eta)>1.56 and tag_pt>40 and probe_pt>20 and mass>80 and mass<100 and abs(tag_eta)<2.5 and abs(probe_eta)<2.5"

    print("Calculating weights...")
    for mc_df in [mc_uncorr_df, mc_corr_df]:
        mc_df.loc[np.abs(mc_df['probe_eta'])<1.4442, 'weight_clf'] = clf_reweight(mc_df.query('abs(probe_eta)<1.4442'), data_df, "clf_eb", n_jobs=10, cut=eb_cut_clf)
        mc_df.loc[np.abs(mc_df['probe_eta'])>1.56,'weight_clf'] = clf_reweight(mc_df.query('abs(probe_eta)>1.56'), data_df, "clf_ee", n_jobs=10, cut=ee_cut_clf)

    data_df_eb = data_df[np.abs(data_df.probe_eta) < 1.4442]
    data_df_ee = data_df[np.abs(data_df.probe_eta) > 1.56]
    mc_uncorr_df_eb = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) < 1.4442]
    mc_uncorr_df_ee = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) > 1.56]
    mc_corr_df_eb = mc_corr_df[np.abs(mc_corr_df.probe_eta) < 1.4442]
    mc_corr_df_ee = mc_corr_df[np.abs(mc_corr_df.probe_eta) > 1.56]

    for var_conf in vars_config:
        plot_variable(data_df_eb, mc_corr_df_eb, mc_uncorr_df_eb, var_conf, output_dir, "EB")
        plot_variable(data_df_ee, mc_corr_df_ee, mc_uncorr_df_ee, var_conf, output_dir, "EE")


if __name__ == "__main__":
    main()
