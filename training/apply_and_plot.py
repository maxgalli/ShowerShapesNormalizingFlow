import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import pickle as pkl
import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from custom_models import load_fff_mixture_model
from utils import ParquetDataset
from plot import divide_dist, dump_profile_plot, plot_variable, clf_reweight


@hydra.main(version_base=None, config_path="config_inference", config_name="cfg_test")
def main(cfg):
    # save the config
    cfg_name = HydraConfig.get().job.name
    with open(f"{os.getcwd()}/{cfg_name}.yaml", "w") as file:
        OmegaConf.save(config=cfg, f=file)
   
    file_name = "checkpoint-latest.pt"
    top_file = os.path.join(cfg.top.path, file_name)
    #top_file = os.path.join(cfg.top.path, "model_@epoch_200.pt")
    mc_file = os.path.join(cfg.mc, file_name)
    data_file = os.path.join(cfg.data, file_name)
    print(f"Files:\n {top_file}\n {mc_file}\n {data_file}")
    all_variables = cfg.context_variables + cfg.target_variables
    print(all_variables)

    # make output dir
    additional_output_str = f"{cfg.additional_output}/{cfg_name}"
    Path(additional_output_str).mkdir(parents=True, exist_ok=True)

    model = load_fff_mixture_model(top_file, mc_file, data_file, {"penalty_type": cfg.top.penalty, "penalty_weight": cfg.top.penalty_weight, "anneal": cfg.top.anneal})[0]
    
    calo = cfg.calo
    test_file_mc = f"../../../preprocess/preprocessed/mc_{calo}_test.parquet"
    test_file_data = f"../../../preprocess/preprocessed/data_{calo}_test.parquet"
    data_df = pd.read_parquet(test_file_data, columns=all_variables).reset_index(drop=True)
    mc_df = pd.read_parquet(test_file_mc, columns=all_variables).reset_index(drop=True)
    lim = 1000000
    # get first lim events
    data_df = data_df[:lim]
    mc_df = mc_df[:lim]

    with open(f"../../../preprocess/preprocessed/pipelines_{calo}.pkl", "rb") as file:
        pipelines_data = pkl.load(file)
        pipelines_data = pipelines_data[cfg.pipelines]

    # preprocess
    for var in all_variables:
        try:
            data_df[var] = pipelines_data[var].transform(data_df[var].values.reshape(-1, 1)).reshape(-1)
            mc_df[var] = pipelines_data[var].transform(mc_df[var].values.reshape(-1, 1)).reshape(-1)
        except KeyError:
            pass
    
    test_dataset_mc = ParquetDataset(
        test_file_mc,
        cfg.context_variables, 
        cfg.target_variables,
        pipelines=pipelines_data,
        #retrain_pipelines=True,
        rows=lim
        )
    test_loader_mc = DataLoader(test_dataset_mc, batch_size=2048, shuffle=False)
    
    print("Applying model to MC")
    with torch.no_grad():
        model.eval()
        mc_corr_list = []
        for context, target in test_loader_mc:
            target_mc_corr, _ = model.transform(target, context, inverse=False)
            target_mc_corr = target_mc_corr.detach().cpu().numpy()
            mc_corr_list.append(target_mc_corr)
    mc_corr = np.concatenate(mc_corr_list, axis=0)
    mc_corr_df = pd.DataFrame(mc_corr, columns=cfg.target_variables)

    # Plot scaled distributions
    print("Plotting scaled distributions")
    for var in cfg.target_variables:
        mn = min(data_df[var].min(), mc_df[var].min(), mc_corr_df[var].min())
        mx = max(data_df[var].max(), mc_df[var].max(), mc_corr_df[var].max())
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(data_df[var], bins=100, range=(mn, mx), histtype="step", label="data", density=True)
        ax.hist(mc_df[var], bins=100, range=(mn, mx), histtype="step", label="MC", density=True)
        ax.hist(mc_corr_df[var], bins=100, range=(mn, mx), histtype="step", label="MC corrected", density=True)
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        ax.legend()
        for ext in ["png", "pdf"]:
            fig.savefig(f"{additional_output_str}/{var}_scaled.{ext}")

    # preprocess back
    print("Preprocessing back")
    for var in all_variables:
        try:
            data_df[var] = pipelines_data[var].inverse_transform(data_df[var].values.reshape(-1, 1)).reshape(-1)
            mc_df[var] = pipelines_data[var].inverse_transform(mc_df[var].values.reshape(-1, 1)).reshape(-1)
        except KeyError:
            pass
    for var in cfg.target_variables:
        #mc_corr_df[var] = pipelines_mc[var].inverse_transform(mc_corr_df[var].values.reshape(-1, 1)).reshape(-1)
        mc_corr_df[var] = pipelines_data[var].inverse_transform(mc_corr_df[var].values.reshape(-1, 1)).reshape(-1)

    # Plot distributions
    with open(
        "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/preprocess/var_specs.json",
        "r",
    ) as f:
        vars_config = json.load(f)
    var_config_to_plot = [dct for dct in vars_config if dct["name"] in cfg.target_variables]
   
    # now the usual CQR plotting procedure
    print(f"Dataset lengths: {len(data_df)}, {len(mc_df)}, {len(mc_corr_df)}")
    mc_corr_df = mc_corr_df.reset_index(drop=True)
    mc_corr_df = pd.concat([mc_corr_df, mc_df[["probe_pt", "probe_eta", "probe_phi", "probe_fixedGridRhoAll"]]], axis=1)

    print("Calculating weights...")
    for df in [mc_df, mc_corr_df]:
        df["weight_clf"] = clf_reweight(
            df, data_df, "clf_{}".format(calo), n_jobs=10, cut=None
        )

    print("Plotting distributions...")

    for var_conf in var_config_to_plot:
        plot_variable(
            data_df, mc_corr_df, mc_df, var_conf, [additional_output_str, "."], calo
        )


if __name__ == "__main__":
    main()