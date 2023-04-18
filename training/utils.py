from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from nflows import transforms, flows
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from ffflows.models import BaseFlow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.distributions import StandardNormal
from nflows.flows.base import Flow
import json
from scipy.stats import wasserstein_distance
import os
from torch.distributed import init_process_group
from pathlib import Path
from ffflows.models import (
    DeltaFlowForFlow,
    ConcatFlowForFlow,
    DiscreteBaseFlowForFlow,
    DiscreteBaseConditionFlowForFlow,
    NoContextFlowForFlow,
    #CustomContextFlowForFlow,
)
from ffflows import distance_penalties
from ffflows.distance_penalties import AnnealedPenalty, BasePenalty


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
                    data[var] = pipeline.transform(
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


def get_conditional_base_flow(
    input_dim,
    context_dim,
    nstack,
    nnodes,
    nblocks,
    tail_bound,
    nbins,
    activation,
    dropout_probability,
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


class FFFCustom(flows.Flow):
    """
    MC = left
    Data = right
    forward: MC -> Data
    inverse: Data -> MC
    """

    def __init__(self, transform, distribution_right, distribution_left, embedding_net=None):
        super().__init__(transform, distribution_right, embedding_net)
        self.flow_mc = distribution_left
        self.flow_data = distribution_right
        
        self.distance_object = BasePenalty()

    def forward(self, inputs, input_context, target_context=None, inverse=False):
        return self.log_prob(inputs, input_context, target_context, inverse)

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
        #print("inputs", inputs)
        #print("converted_input", converted_input)
        #print("logabsdet", logabsdet)
        #print(f"log_prob: {log_prob}")
        #print(f"dist_pen: {dist_pen}")

        return log_prob + logabsdet + dist_pen

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
        "custom": FFFCustom,
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


def create_random_transform(param_dim):
    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def create_mixture_flow_model(
    input_dim, context_dim, base_kwargs, transform_type, fff_type=None, mc_flow=None, data_flow=None
):
    transform = []
    for _ in range(base_kwargs["num_steps_maf"]):
        transform.append(
            MaskedAffineAutoregressiveTransform(
                features=input_dim,
                use_residual_blocks=base_kwargs["use_residual_blocks_maf"],
                num_blocks=base_kwargs["num_transform_blocks_maf"],
                hidden_features=base_kwargs["hidden_dim_maf"],
                context_features=context_dim,
                dropout_probability=base_kwargs["dropout_probability_maf"],
                use_batch_norm=base_kwargs["batch_norm_maf"],
            )
        )
        transform.append(create_random_transform(param_dim=input_dim))

    for _ in range(base_kwargs["num_steps_arqs"]):
        transform.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=input_dim,
                tails="linear",
                use_residual_blocks=base_kwargs["use_residual_blocks_arqs"],
                hidden_features=base_kwargs["hidden_dim_arqs"],
                num_blocks=base_kwargs["num_transform_blocks_arqs"],
                tail_bound=base_kwargs["tail_bound_arqs"],
                num_bins=base_kwargs["num_bins_arqs"],
                context_features=context_dim,
                dropout_probability=base_kwargs["dropout_probability_arqs"],
                use_batch_norm=base_kwargs["batch_norm_arqs"],
            )
        )
        transform.append(create_random_transform(param_dim=input_dim))

    transform_fnal = transforms.CompositeTransform(transform)

    if fff_type is None and data_flow is None and mc_flow is None:
        distribution = StandardNormal((input_dim,))
        flow = Flow(transform_fnal, distribution)
    else:
        #flow = FFFM(transform_fnal, mc_flow, data_flow)
        flow = get_flow4flow(name=fff_type, transform=transform_fnal, distribution_right=data_flow, distribution_left=mc_flow)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": base_kwargs,
        "transform_type": transform_type,
    }

    return flow


def load_mixture_model(device, model_dir=None, filename=None):
    if model_dir is None:
        raise NameError(
            "Model directory must be specified."
            " Store in attribute PosteriorModel.model_dir"
        )

    p = Path(model_dir)
    checkpoint = torch.load(p / filename, map_location="cpu")

    model_hyperparams = checkpoint["model_hyperparams"]
    # added because of a bug in the old create_mixture_flow_model function
    try:
        if checkpoint["model_hyperparams"]["base_transform_kwargs"] is not None:
            checkpoint["model_hyperparams"]["base_kwargs"] = checkpoint[
                "model_hyperparams"
            ]["base_transform_kwargs"]
            del checkpoint["model_hyperparams"]["base_transform_kwargs"]
    except KeyError:
        pass
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_mixture_flow_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["last_lr"]
    elif checkpoint["last_lr"] is not None:
        flow_lr = checkpoint["last_lr"][0]
    else:
        flow_lr = None

    # Set the epoch to the correct value. This is needed to resume
    # training.
    epoch = checkpoint["epoch"]

    return (
        model,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
    )


def save_model(
    epoch,
    model,
    scheduler,
    train_history,
    test_history,
    name,
    model_dir=None,
    optimizer=None,
    is_ddp=False,
    save_both=False,
):
    """Save a model and optimizer to file.
    Args:
        model:      model to be saved
        optimizer:  optimizer to be saved
        epoch:      current epoch number
        model_dir:  directory to save the model in
        filename:   filename for saved model
    """

    if model_dir is None:
        raise NameError("Model directory must be specified.")

    filename = name + f"_@epoch_{epoch}.pt"
    resume_filename = "checkpoint-latest.pt"

    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "train_history": train_history,
        "test_history": test_history,
        "model_hyperparams": model.module.model_hyperparams
        if is_ddp
        else model.model_hyperparams,
        "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        dict["scheduler_state_dict"] = scheduler.state_dict()
        dict["last_lr"] = scheduler.get_last_lr()

    torch.save(dict, p / resume_filename)
    if save_both:
        torch.save(dict, p / filename)


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
        ax.hist(reco[var], bins=100, histtype="step", label="reco", range=(mn, mx))
        ws = wasserstein_distance(reco[var], samples[var])
        ax.hist(
            samples[var],
            bins=100,
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
        ax.hist(data[var], bins=100, histtype="step", label="data", range=(mn, mx))
        for smp, name in zip([mc, mc_corr], ["mc", "mc corr"]):
            ws = wasserstein_distance(data[var], smp[var])
            ax.hist(
                smp[var],
                bins=100,
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