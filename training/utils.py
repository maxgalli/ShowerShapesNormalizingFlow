from torch.utils.data import Dataset
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
import dask.dataframe as dd
import os
import json
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from joblib import dump
import pandas as pd
import torch
from nflows import transforms, flows
from nflows.distributions.normal import ConditionalDiagonalNormal
import matplotlib
import matplotlib.pyplot as plt
import itertools
from torch.nn import functional as F
from ffflows.models import (
    DeltaFlowForFlow,
    ConcatFlowForFlow,
    DiscreteBaseFlowForFlow,
    DiscreteBaseConditionFlowForFlow,
    NoContextFlowForFlow,
)
from ffflows import distance_penalties
from ffflows.distance_penalties import AnnealedPenalty
from ffflows.distance_penalties import BasePenalty
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class ParquetDataset(Dataset):
    def __init__(
        self,
        parquet_file,
        context_variables,
        target_variables,
        device=None,
        pipelines=None,
        retrain_pipelines=False,
        rows=None,
    ):
        self.parquet_file = parquet_file
        self.context_variables = context_variables
        self.target_variables = target_variables
        self.all_variables = context_variables + target_variables
        data = pd.read_parquet(
            parquet_file, columns=self.all_variables, engine="fastparquet"
        )
        self.pipelines = pipelines
        if self.pipelines is not None:
            for var, pipeline in self.pipelines.items():
                if var in self.all_variables:
                    trans = pipeline.fit_transform if retrain_pipelines else pipeline.transform
                    data[var] = trans(
                        data[var].values.reshape(-1, 1)
                    ).reshape(-1)
        if rows is not None:
            data = data.iloc[:rows]
        self.target = data[target_variables].values
        self.context = data[context_variables].values
        if device is not None:
            self.target = torch.tensor(self.target, dtype=torch.float32).to(device)
            self.context = torch.tensor(self.context, dtype=torch.float32).to(device)

    def __len__(self):
        assert len(self.context) == len(self.target)
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]


class BaseFlow(flows.Flow):
    """
    Wrapper class around Base Flow for a flow for flow model.
    Harmonises function calls with FlowForFlow model.
    Constructed and used exactly like an nflows.Flow object.
    """

    def forward(self, inputs, context=None):
        # raise RuntimeError("Forward method cannot be called for a Distribution object.")
        return self.log_prob(inputs, context)

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
    dropout_probability=0.0,
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
                dropout_probability=dropout_probability,
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


def create_baseflow_model(
    input_dim,
    context_dim,
    nnodes,
    nblocks,
    nstack,
    tail_bound,
    activation,
    dropout_probability,
    nbins,
):
    flow = BaseFlow(
        spline_inn(
            input_dim,
            nodes=nnodes,
            num_blocks=nblocks,
            num_stack=nstack,
            tail_bound=tail_bound,
            activation=getattr(F, activation),
            dropout_probability=dropout_probability,
            num_bins=nbins,
            context_features=context_dim,
        ),
        ConditionalDiagonalNormal(
            shape=[input_dim], context_encoder=nn.Linear(context_dim, 2 * input_dim)
        ),
    )

    return flow


def divide_dist(distribution, bins):
    sorted_dist = np.sort(distribution)
    subgroup_size = len(distribution) // bins
    edges = [sorted_dist[0]]
    for i in range(subgroup_size, len(sorted_dist), subgroup_size):
        edges.append(sorted_dist[i])
    edges[-1] = sorted_dist[-1]
    return edges


def dump_profile_plot(
    ax, ss_name, cond_name, sample_name, ss_arr, cond_arr, color, cond_edges
):
    df = pd.DataFrame({ss_name: ss_arr, cond_name: cond_arr})
    quantiles = [0.25, 0.5, 0.75]
    qlists = [[], [], []]
    centers = []
    for left_edge, right_edge in zip(cond_edges[:-1], cond_edges[1:]):
        dff = df[(df[cond_name] > left_edge) & (df[cond_name] < right_edge)]
        qlist = np.quantile(dff[ss_name], quantiles)
        for i, q in enumerate(qlist):
            qlists[i].append(q)
        centers.append((left_edge + right_edge) / 2)
    mid_index = len(quantiles) // 2
    for qlist in qlists[:mid_index]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    for qlist in qlists[mid_index:]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    ax.plot(centers, qlists[mid_index], color=color, label=sample_name)

    return ax


def sample_and_plot_base(
    test_loader,
    model,
    epoch,
    writer,
    context_variables,
    target_variables,
    device,
):
    target_size = len(target_variables)
    with torch.no_grad():
        gen, reco, samples = [], [], []
        for context, target in test_loader:
            context = context.to(device)
            target = target.to(device)
            sample = model.sample(num_samples=1, context=context)
            context = context.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            sample = sample.detach().cpu().numpy()
            sample = sample.reshape(-1, target_size)
            gen.append(context)
            reco.append(target)
            samples.append(sample)
    gen = np.concatenate(gen, axis=0)
    reco = np.concatenate(reco, axis=0)
    samples = np.concatenate(samples, axis=0)
    gen = pd.DataFrame(gen, columns=context_variables)
    reco = pd.DataFrame(reco, columns=target_variables)
    samples = pd.DataFrame(samples, columns=target_variables)

    # plot the reco and sampled distributions
    for var in target_variables:
        mn = min(reco[var].min(), samples[var].min())
        mx = max(reco[var].max(), samples[var].max())
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), tight_layout=True)
        #ax.hist(reco[var], bins=100, histtype="step", label="reco", range=(mn, mx))
        ax.hist(reco[var], bins=21, histtype="step", label="reco", range=(mn, mx))
        ws = wasserstein_distance(reco[var], samples[var])
        ax.hist(
            samples[var],
            #bins=100,
            bins=21,
            histtype="step",
            label=f"sampled (wasserstein={ws:.3f})",
            range=(mn, mx),
        )
        ax.set_xlabel(var)
        ax.legend()
        if device == 0 or type(device) != int:
            writer.add_figure(f"{var}_reco_sampled", fig, epoch)

    # plot after preprocessing back
    preprocess_dct = test_loader.dataset.pipelines
    reco_back = {}
    samples_back = {}
    with open("../../../preprocess/var_specs.json", "r") as f:
        lst = json.load(f)
        original_ranges = {dct["name"]: dct["range"] for dct in lst}
    for var in target_variables:
        reco_back[var] = (
            preprocess_dct[var]
            .inverse_transform(reco[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        samples_back[var] = (
            preprocess_dct[var]
            .inverse_transform(samples[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    reco_back = pd.DataFrame(reco_back)
    samples_back = pd.DataFrame(samples_back)
    for var in target_variables:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), tight_layout=True)
        ax.hist(
            reco_back[var],
            bins=100,
            histtype="step",
            label="reco",
            range=original_ranges[var],
        )
        ax.hist(
            samples_back[var],
            bins=100,
            histtype="step",
            label="sampled",
            range=original_ranges[var],
        )
        ax.set_xlabel(var)
        ax.legend()
        if device == 0 or type(device) != int:
            writer.add_figure(f"{var}_reco_sampled_back", fig, epoch)


def transform_and_plot_top(
    mc_loader,
    data_loader,
    model,
    epoch,
    writer,
    context_variables,
    target_variables,
    device,
):
    target_size = len(target_variables)
    with torch.no_grad():
        data_lst, mc_lst, mc_corr_lst = [], [], []
        data_context_lst, mc_context_lst, mc_corr_context_lst = [], [], []
        for data, mc in zip(data_loader, mc_loader):
            context_data, target_data = data
            context_mc, target_mc = mc
            target_mc_corr, _ = model.transform(target_mc, context_mc, inverse=False)
            target_data = target_data.detach().cpu().numpy()
            target_mc = target_mc.detach().cpu().numpy()
            target_mc_corr = target_mc_corr.detach().cpu().numpy()
            context_data = context_data.detach().cpu().numpy()
            context_mc = context_mc.detach().cpu().numpy()
            data_lst.append(target_data)
            mc_lst.append(target_mc)
            mc_corr_lst.append(target_mc_corr)
            data_context_lst.append(context_data)
            mc_context_lst.append(context_mc)
            mc_corr_context_lst.append(context_mc)
    data = np.concatenate(data_lst, axis=0)
    mc = np.concatenate(mc_lst, axis=0)
    mc_corr = np.concatenate(mc_corr_lst, axis=0)
    data = pd.DataFrame(data, columns=target_variables)
    mc = pd.DataFrame(mc, columns=target_variables)
    mc_corr = pd.DataFrame(mc_corr, columns=target_variables)
    data_context = np.concatenate(data_context_lst, axis=0)
    mc_context = np.concatenate(mc_context_lst, axis=0)
    mc_corr_context = np.concatenate(mc_corr_context_lst, axis=0)
    data_context = pd.DataFrame(data_context, columns=context_variables)
    mc_context = pd.DataFrame(mc_context, columns=context_variables)
    mc_corr_context = pd.DataFrame(mc_corr_context, columns=context_variables)
        
    # plot the reco and sampled distributions
    for var in target_variables:
        mn = min(data[var].min(), mc[var].min(), mc_corr[var].min())
        mx = max(data[var].max(), mc[var].max(), mc_corr[var].max())
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), tight_layout=True)
        #ax.hist(data[var], bins=100, histtype="step", label="data", range=(mn, mx))
        ax.hist(data[var], bins=21, histtype="step", label="data", range=(mn, mx))
        for smp, name in zip([mc, mc_corr], ["mc", "mc corr"]):
            ws = wasserstein_distance(data[var], smp[var])
            ax.hist(
                smp[var],
                #bins=100,
                bins=21,
                histtype="step",
                label=f"{name} (wasserstein={ws:.3f})",
                range=(mn, mx),
            )
        ax.set_xlabel(var)
        ax.legend()
        if device == 0 or type(device) != int:
            writer.add_figure(f"{var}_reco_sampled", fig, epoch)

    # now plot profiles
    nbins = 8
    for column in target_variables:
        for cond_column in context_variables:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            data_ss_arr = data[column].values
            data_cond_arr = data_context[cond_column].values
            mc_uncorr_ss_arr = mc[column].values
            mc_uncorr_cond_arr = mc_context[cond_column].values
            mc_corr_ss_arr = mc_corr[column].values
            mc_corr_cond_arr = mc_corr_context[cond_column].values
            cond_edges = divide_dist(data_cond_arr, nbins)

            for name, ss_arr, cond_arr, color in [
                ("data", data_ss_arr, data_cond_arr, "blue"),
                ("mc", mc_uncorr_ss_arr, mc_uncorr_cond_arr, "red"),
                ("mc corr", mc_corr_ss_arr, mc_corr_cond_arr, "green"),
            ]:
                ax = dump_profile_plot(
                    ax=ax,
                    ss_name=column,
                    cond_name=cond_column,
                    sample_name=name,
                    ss_arr=ss_arr,
                    cond_arr=cond_arr,
                    color=color,
                    cond_edges=cond_edges,
                )
            ax.legend()
            ax.set_xlabel(cond_column)
            ax.set_ylabel(column)
            if writer is not None:
                writer.add_figure(f"profiles_{column}_{cond_column}", fig, epoch)
    
    # sample back
    data_pipeline = data_loader.dataset.pipelines
    mc_pipeline = mc_loader.dataset.pipelines

    ranges = {
        "probe_r9": (0, 1.2),
        "probe_s4": (0, 1.2),
        "probe_sieie": (0.002, 0.014),
        "probe_sieip": (-0.002, 0.002),
        "probe_etaWidth": (0, 0.03),
        "probe_phiWidth": (0, 0.1),
    }
    for var in target_variables:
        data[var] = data_pipeline[var].inverse_transform(data[var].values.reshape(-1, 1)).reshape(-1)
        mc[var] = mc_pipeline[var].inverse_transform(mc[var].values.reshape(-1, 1)).reshape(-1)
        mc_corr[var] = mc_pipeline[var].inverse_transform(mc_corr[var].values.reshape(-1, 1)).reshape(-1)
        rng = ranges[var]
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), tight_layout=True)
        ax.hist(data[var], bins=100, histtype="step", label="data", range=rng)
        for smp, name in zip([mc, mc_corr], ["mc", "mc corr"]):
            ws = wasserstein_distance(data[var], smp[var])
            ax.hist(
                smp[var],
                bins=100,
                histtype="step",
                label=f"{name} (wasserstein={ws:.3f})",
                range=rng,
            )
        ax.set_xlabel(var)
        ax.legend()
        if device == 0 or type(device) != int:
            writer.add_figure(f"{var}_reco_sampled_back", fig, epoch)

    # now plot profiles
    nbins = 8
    for column in target_variables:
        for cond_column in context_variables:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            data_ss_arr = data[column].values
            data_cond_arr = data_context[cond_column].values
            mc_uncorr_ss_arr = mc[column].values
            mc_uncorr_cond_arr = mc_context[cond_column].values
            mc_corr_ss_arr = mc_corr[column].values
            mc_corr_cond_arr = mc_corr_context[cond_column].values
            cond_edges = divide_dist(data_cond_arr, nbins)

            for name, ss_arr, cond_arr, color in [
                ("data", data_ss_arr, data_cond_arr, "blue"),
                ("mc", mc_uncorr_ss_arr, mc_uncorr_cond_arr, "red"),
                ("mc corr", mc_corr_ss_arr, mc_corr_cond_arr, "green"),
            ]:
                ax = dump_profile_plot(
                    ax=ax,
                    ss_name=column,
                    cond_name=cond_column,
                    sample_name=name,
                    ss_arr=ss_arr,
                    cond_arr=cond_arr,
                    color=color,
                    cond_edges=cond_edges,
                )
            ax.legend()
            ax.set_xlabel(cond_column)
            ax.set_ylabel(column)
            if writer is not None:
                writer.add_figure(f"profiles_{column}_{cond_column}_sampled_back", fig, epoch)
    # close figures
    plt.close("all")


def dump_validation_plots(
    flow,
    valdataset,
    columns,
    condition_columns,
    nsample,
    device,
    path,
    epoch,
    rng=(-5, 5),
    writer=None,
):
    epoch = str(epoch + 1) if type(epoch) == int else epoch
    print("Dumping validation plots")
    ncond = len(condition_columns)
    xcond = torch.tensor(valdataset.df.values[:, :ncond].astype(np.float32)).to(device)
    with torch.no_grad():
        sample = flow.sample(nsample, context=xcond)
    # 1D plots
    for c in columns:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        print(f"Dumping {c} 1D plot")
        ax.hist(
            valdataset.df[c],
            bins=100,
            range=rng,
            density=True,
            histtype="step",
            label="Data",
        )
        sub_sample = sample[:, :, columns.index(c)]
        x = sub_sample.reshape(sub_sample.shape[0] * sub_sample.shape[1])
        ax.hist(
            x.cpu().numpy(),
            bins=100,
            range=rng,
            density=True,
            histtype="step",
            label="Sampled",
        )
        ax.legend()
        fig.savefig(f"{path}/epoch_{epoch}_{c}_1D.png")
        if writer is not None:
            writer.add_figure(f"epoch_{c}_1D", fig, epoch)

    # now plot profiles
    nbins = 8
    for column in columns:
        for cond_column in condition_columns:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            cond_arr = valdataset.df[cond_column].values
            data_arr = valdataset.df[column].values
            sub_sample = sample[:, :, columns.index(column)]
            x = sub_sample.reshape(sub_sample.shape[0] * sub_sample.shape[1])
            cond_edges = divide_dist(cond_arr, nbins)
            for name, arr, color in [
                ("Data", data_arr, "blue"),
                ("Sampled", x.cpu().numpy(), "red"),
            ]:
                ax = dump_profile_plot(
                    ax=ax,
                    ss_name=column,
                    cond_name=cond_column,
                    sample_name=name,
                    ss_arr=arr,
                    cond_arr=cond_arr,
                    color=color,
                    cond_edges=cond_edges,
                )
            ax.legend()
            ax.set_xlabel(cond_column)
            ax.set_ylabel(column)
            fig.savefig(f"{path}/profiles_epoch_{epoch}_{column}_{cond_column}.png")
            if writer is not None:
                writer.add_figure(f"profiles_{column}_{cond_column}", fig, epoch)


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

    def base_flow_log_prob(
        self, inputs, input_context, target_context=None, inverse=False
    ):
        if inverse:
            return self.flow_data.log_prob(inputs, input_context)
        else:
            return self.flow_mc.log_prob(inputs, input_context)

    def transform(self, inputs, input_context, target_context=None, inverse=False):
        context = self._embedding_net(input_context)
        transform = self._transform.inverse if inverse else self._transform
        # convert to float32
        inputs = inputs.float()
        context = context.float()
        y, logabsdet = transform(inputs, context)

        return y, logabsdet

    def log_prob(self, inputs, input_context, target_context=None, inverse=False):
        converted_input, logabsdet = self.transform(
            inputs, input_context, target_context, inverse=inverse
        )
        log_prob = self.base_flow_log_prob(
            converted_input, input_context, target_context, inverse=inverse
        )
        dist_pen = -self.distance_object(converted_input, inputs)

        total_log_prob = log_prob + logabsdet + dist_pen

        return total_log_prob, log_prob, logabsdet, dist_pen

    def batch_transform(
        self, inputs, input_context, target_context=None, inverse=False
    ):
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


def dump_validation_plots_top(
    model,
    data_val,
    mc_val,
    columns,
    condition_columns,
    path,
    epoch,
    device,
    rng=(-5, 5),
    writer=None,
):
    print("Dumping validation plots")
    ncond = len(condition_columns)

    inputs = torch.tensor(mc_val.dataset.df.values[:, ncond:]).to(device)
    context_l = torch.tensor(mc_val.dataset.df.values[:, :ncond]).to(device)
    context_r = torch.tensor(data_val.dataset.df.values[:, :ncond]).to(device)
    with torch.no_grad():
        # mc_to_data, _ = model.batch_transform(inputs, context_l, context_r, inverse=True, batch_size=10000)
        mc_to_data, _ = model.batch_transform(
            inputs, context_l, target_context=context_r, inverse=False
        )
        # mc_to_data, _ = model.batch_transform(inputs, context_l, target_context=None, inverse=False)

    # 1D histograms
    scaled_back_df_data = data_val.dataset.get_scaled_back_df()
    scaled_back_df_mc = mc_val.dataset.get_scaled_back_df()

    for i, column in enumerate(columns):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(
            data_val.dataset.df[column],
            bins=100,
            histtype="step",
            label="data",
            density=True,
            range=rng,
        )
        ax.hist(
            mc_val.dataset.df[column],
            bins=100,
            histtype="step",
            label="mc",
            density=True,
            range=rng,
        )
        ax.hist(
            mc_to_data[:, i].cpu().numpy(),
            bins=100,
            histtype="step",
            label="mc to data",
            density=True,
            range=rng,
        )
        ax.set_label(column)
        ax.set_ylabel("density")
        ax.legend()
        fig.savefig(f"{path}/epoch_{epoch}_{column}_scaled_1D.png")
        if writer is not None:
            writer.add_figure(f"scaled_1D_{column}", fig, epoch)

        # scale back
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        copied_mc_val = deepcopy(mc_val)
        for col in columns:
            copied_mc_val.dataset.df[col] = (
                mc_to_data[:, columns.index(col)].cpu().numpy()
            )
        copied_mc_val.dataset.scale_back()
        # get minimum and maximum as the minimum between the three
        min_val = min(
            scaled_back_df_data[column].min(),
            scaled_back_df_mc[column].min(),
            copied_mc_val.dataset.df[column].min(),
        )
        max_val = max(
            scaled_back_df_data[column].max(),
            scaled_back_df_mc[column].max(),
            copied_mc_val.dataset.df[column].max(),
        )
        rng_scaledback = (min_val, max_val)
        ax.hist(
            scaled_back_df_data[column],
            bins=100,
            histtype="step",
            label="data",
            density=True,
            range=rng_scaledback,
        )
        ax.hist(
            scaled_back_df_mc[column],
            bins=100,
            histtype="step",
            label="mc",
            density=True,
            range=rng_scaledback,
        )
        ax.hist(
            copied_mc_val.dataset.df[column],
            bins=100,
            histtype="step",
            label="mc to data",
            density=True,
            range=rng_scaledback,
        )
        ax.set_xlabel(column)
        ax.set_ylabel("density")
        ax.legend()
        fig.savefig(f"{path}/epoch_{epoch}_{column}_scaledback_1D.png")
        if writer is not None:
            writer.add_figure(f"scaledback_1D_{column}", fig, epoch)

    # now plot profiles
    nbins = 8
    for column in columns:
        for cond_column in condition_columns:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            data_ss_arr = data_val.dataset.df[column].values
            data_cond_arr = data_val.dataset.df[cond_column].values
            mc_uncorr_ss_arr = mc_val.dataset.df[column].values
            mc_uncorr_cond_arr = mc_val.dataset.df[cond_column].values
            mc_corr_ss_arr = mc_to_data[:, columns.index(column)].cpu().numpy()
            mc_corr_cond_arr = mc_val.dataset.df[cond_column].values
            cond_edges = divide_dist(data_cond_arr, nbins)

            for name, ss_arr, cond_arr, color in [
                ("data", data_ss_arr, data_cond_arr, "blue"),
                ("mc", mc_uncorr_ss_arr, mc_uncorr_cond_arr, "red"),
                ("mc to data", mc_corr_ss_arr, mc_corr_cond_arr, "green"),
            ]:
                ax = dump_profile_plot(
                    ax=ax,
                    ss_name=column,
                    cond_name=cond_column,
                    sample_name=name,
                    ss_arr=ss_arr,
                    cond_arr=cond_arr,
                    color=color,
                    cond_edges=cond_edges,
                )
            ax.legend()
            ax.set_xlabel(cond_column)
            ax.set_ylabel(column)
            fig.savefig(f"{path}/profiles_epoch_{epoch}_{column}_{cond_column}.png")
            if writer is not None:
                writer.add_figure(f"profiles_{column}_{cond_column}", fig, epoch)


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
    rng_plt=(-5, 5),
    writer=None,
):
    print(
        f"Training Flow4Flow  on {device} with {n_epochs} epochs and learning rate {learning_rate}, alternating every batch"
    )
    ncond = len(condition_columns)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.001
    )
    num_steps = len(data_train) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_steps, last_epoch=-1, eta_min=0
    )
    if hasattr(model, "distance_object.set_n_steps"):
        model.distance_object.set_n_steps(num_steps)
    train_losses = []
    val_losses = []
    best_vloss = np.inf
    for epoch in range(n_epochs):
        epoch += 1
        print(f"Epoch {epoch}/{n_epochs}")
        tl = []
        tlp = []
        tla = []
        tld = []
        vl = []
        vlp = []
        vla = []
        vld = []
        # print how many batches we have
        # print(f"Training batches data: {len(data_train)}")
        # print(f"Validation batches data: {len(data_val)}")
        # print(f"Training batches mc: {len(mc_train)}")
        # print(f"Validation batches mc: {len(mc_val)}")
        # print(len(data_train.dataset.df))
        for i, (data, mc) in enumerate(zip(data_train, mc_train)):
            model.train()
            # print(data.shape, mc.shape)
            if i % 2 == 0 + 1 * int(inverse):
                # print("INVERSE False, from mc to data")
                batch = mc.to(device)
                other_batch = data.to(device)
                inv = False
            else:
                # print("INVERSE True, from data to mc")
                batch = data.to(device)
                other_batch = mc.to(device)
                inv = True

            optimizer.zero_grad()
            inputs = batch[:, ncond:]
            context_l = batch[:, :ncond]
            context_r = other_batch[:, :ncond]
            # print(inputs.shape, context_l.shape, context_r.shape)

            loss, logprob, logabsdet, distance = model.log_prob(
                inputs, input_context=context_l, target_context=context_r, inverse=inv
            )
            loss = -loss.mean()
            logprob = -logprob.mean()
            logabsdet = -logabsdet.mean()
            distance = -distance.mean()

            loss.backward()

            if gclip not in ["None", None]:
                clip_grad_norm_(model.parameters(), gclip)
            optimizer.step()
            scheduler.step()
            tl.append(loss.item())
            tlp.append(logprob.item())
            tla.append(logabsdet.item())
            tld.append(distance.item())

        tloss = np.mean(tl)
        train_losses.append(tloss)
        tlp_mean = np.mean(tlp)
        tla_mean = np.mean(tla)
        tld_mean = np.mean(tld)

        for i, (data, mc) in enumerate(zip(data_val, mc_val)):
            for inv, batch, other_batch in zip([False, True], [data, mc], [mc, data]):
                batch = batch.to(device)
                other_batch = other_batch.to(device)
                inputs = batch[:, ncond:]
                context_l = batch[:, :ncond]
                context_r = other_batch[:, :ncond]
                with torch.no_grad():
                    loss, logprob, logabsdet, distance = model.log_prob(
                        inputs,
                        input_context=context_l,
                        target_context=context_r,
                        inverse=inv,
                    )
                    loss = -loss.mean()
                    logprob = -logprob.mean()
                    logabsdet = -logabsdet.mean()
                    distance = -distance.mean()

                vl.append(loss.item())
                vlp.append(logprob.item())
                vla.append(logabsdet.item())
                vld.append(distance.item())

        vloss = np.mean(vl)
        val_losses.append(vloss)
        vlp_mean = np.mean(vlp)
        vla_mean = np.mean(vla)
        vld_mean = np.mean(vld)

        if writer is not None:
            writer.add_scalars(
                "losses",
                {
                    "train": tloss,
                    "val": vloss,
                },
                epoch,
            )
            writer.add_scalars(
                "logprob",
                {
                    "train": tlp_mean,
                    "val": vlp_mean,
                },
                epoch,
            )
            writer.add_scalars(
                "logabsdet",
                {
                    "train": tla_mean,
                    "val": vla_mean,
                },
                epoch,
            )
            writer.add_scalars(
                "lossdistance",
                {
                    "train": tld_mean,
                    "val": vld_mean,
                },
                epoch,
            )

        print(
            f"Epoch {epoch}/{n_epochs} - Train loss: {tloss:.4f} - Val loss: {vloss:.4f}"
        )
        if vloss < best_vloss:
            best_vloss = vloss
            print("Saving model")
        else:
            print(f"Validation did not improve from {best_vloss:.4f} to {vloss:.4f}")
        torch.save(
            model.state_dict(),
            f"{path}/epoch_{epoch}_valloss_{vloss:.3f}.pt".replace("-", "m"),
        )

        # plot losses
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.savefig(f"{path}/losses.png")

        # dump validations plots
        if epoch % 2 == 0:
            dump_validation_plots_top(
                model,
                data_val,
                mc_val,
                columns,
                condition_columns,
                path,
                epoch,
                device=device,
                rng=rng_plt,
                writer=writer,
            )


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
    rng_plt=(-5, 5),
    writer=None,
):
    print(
        f"Training Flow4Flow in fwd mode on {device} with {n_epochs} epochs and learning rate {learning_rate}, alternating every batch"
    )
    ncond = len(condition_columns)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(mc_train) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_steps, last_epoch=-1, eta_min=0
    )
    if hasattr(model, "distance_object.set_n_steps"):
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
            # print(inputs.shape, context_l.shape, context_r.shape)

            loss = -model.log_prob(
                inputs,
                input_context=context_l,
                target_context=context_r,
                inverse=inverse,
            ).mean()
            loss.backward()
            if gclip not in ["None", None]:
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
                loss = -model.log_prob(
                    inputs,
                    input_context=context_l,
                    target_context=context_r,
                    inverse=inverse,
                ).mean()
            vl.append(loss.item())

        vloss = np.mean(vl)
        val_losses.append(vloss)

        print(
            f"Epoch {epoch + 1}/{n_epochs} - Train loss: {tloss:.4f} - Val loss: {vloss:.4f}"
        )
        if vloss < best_vloss:
            best_vloss = vloss
            print("Saving model")
            torch.save(
                model.state_dict(),
                f"{path}/epoch_{epoch + 1}_valloss_{vloss:.3f}.pt".replace("-", "m"),
            )
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
        if (epoch == n_epochs - 1) or (epoch == n_epochs / 2) or (epoch % 5 == 0):
            dump_validation_plots_top(
                model,
                data_val,
                mc_val,
                columns,
                condition_columns,
                path,
                epoch,
                device=device,
                rng=rng_plt,
                writer=writer,
            )
