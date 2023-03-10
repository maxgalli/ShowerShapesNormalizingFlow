from torch.utils.data import Dataset
import dask.dataframe as dd
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
    CustomContextFlowForFlow,
)
from ffflows import distance_penalties
from ffflows.distance_penalties import AnnealedPenalty
from ffflows.distance_penalties import BasePenalty
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy


def saturate_df(df):
    df["probe_pt"] = df["probe_pt"].clip(upper=200)
    if "probe_sieie" in df.columns:
        df["probe_sieie"] = df["probe_sieie"].clip(upper=0.012, lower=0.005)
    if "probe_sieip" in df.columns:
        df["probe_sieip"] = df["probe_sieip"].clip(upper=0.0001, lower=-0.0001)
    if "probe_etaWidth" in df.columns:
        df["probe_etaWidth"] = df["probe_etaWidth"].clip(upper=0.025)
    if "probe_phiWidth" in df.columns:
        df["probe_phiWidth"] = df["probe_phiWidth"].clip(upper=0.125)
    return df


def scale_pt(pt_array):
    pt_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
    pt_scaled = pt_transformer.transform(pt_array.reshape(-1, 1))
    return pt_scaled, pt_transformer


def standard_scaling(df, saturate=True, **kwargs):
    # saturate
    if saturate:
        df = saturate_df(df)

    # scale pt
    pt = df["probe_pt"].values
    df["probe_pt"], pt_transformer = scale_pt(pt)

    # standard scaling
    y = df.values
    scaler = StandardScaler()
    scaler.fit(y)
    y_scaled = scaler.transform(y)
    df = pd.DataFrame(y_scaled, columns=df.columns)

    scalers = {
        "scaler": scaler,
        "pt_transformer": pt_transformer,
    }

    return df, scalers


def qtgaus_scaling(df, saturate=True, **kwargs):
    # saturate
    if saturate:
        df = saturate_df(df)

    # scale pt
    pt = df["probe_pt"].values
    df["probe_pt"], pt_transformer = scale_pt(pt)

    # qtgaus scaling
    y = df.values
    scaler = QuantileTransformer(
        output_distribution="normal", n_quantiles=1000, random_state=0
    )
    scaler.fit(y)
    y_scaled = scaler.transform(y)
    df = pd.DataFrame(y_scaled, columns=df.columns)

    scalers = {
        "scaler": scaler,
        "pt_transformer": pt_transformer,
    }

    return df, scalers


def standard_scaling_inv(df, scalers):
    # standard scaling
    y = df.values
    scaler = scalers["scaler"]
    y_scaled = scaler.inverse_transform(y)
    df = pd.DataFrame(y_scaled, columns=df.columns)

    # unscale pt
    pt_transformer = scalers["pt_transformer"]
    pt_scaled = df["probe_pt"].values
    df["probe_pt"] = pt_transformer.inverse_transform(pt_scaled.reshape(-1, 1))

    return df


def custom_scaling_1(df, saturate=True, **kwargs):
    # saturate
    if saturate:
        df = saturate_df(df)

    # scale pt
    pt = df["probe_pt"].values
    df["probe_pt"], pt_transformer = scale_pt(pt)

    scaler_one_columns = [
        "probe_pt",
        "probe_eta",
        "probe_phi",
        "probe_fixedGridRhoAll",
        "probe_r9",
        "probe_s4",
    ]
    scaler_two_columns = [
        "probe_sieie",
        "probe_sieip",
        "probe_etaWidth",
        "probe_phiWidth",
    ]
    y_one = df[scaler_one_columns].values
    y_two = df[scaler_two_columns].values
    scaler_one = StandardScaler()
    scaler_two = QuantileTransformer(output_distribution="normal")
    scaler_one.fit(y_one)
    scaler_two.fit(y_two)
    y_one_scaled = scaler_one.transform(y_one)
    y_two_scaled = scaler_two.transform(y_two)

    df[scaler_one_columns] = pd.DataFrame(y_one_scaled, columns=scaler_one_columns)
    df[scaler_two_columns] = pd.DataFrame(y_two_scaled, columns=scaler_two_columns)

    scalers = {
        "scaler_one": scaler_one,
        "scaler_two": scaler_two,
        "pt_transformer": pt_transformer,
    }

    return df, scalers


def custom_scaling_1_inv(df, scalers):
    scaler_one_columns = [
        "probe_pt",
        "probe_eta",
        "probe_phi",
        "probe_fixedGridRhoAll",
        "probe_r9",
        "probe_s4",
    ]
    scaler_two_columns = [
        "probe_sieie",
        "probe_sieip",
        "probe_etaWidth",
        "probe_phiWidth",
    ]

    y_one = df[scaler_one_columns].values
    scaler_one = scalers["scaler_one"]
    y_one_scaled = scaler_one.inverse_transform(y_one)
    y_two = df[scaler_two_columns].values
    scaler_two = scalers["scaler_two"]
    y_two_scaled = scaler_two.inverse_transform(y_two)
    df[scaler_one_columns] = pd.DataFrame(y_one_scaled, columns=scaler_one_columns)
    df[scaler_two_columns] = pd.DataFrame(y_two_scaled, columns=scaler_two_columns)

    # unscale pt
    pt_transformer = scalers["pt_transformer"]
    pt_scaled = df["probe_pt"].values
    df["probe_pt"] = pt_transformer.inverse_transform(pt_scaled.reshape(-1, 1))

    return df


scaling_functions = {
    "standard_scaling": standard_scaling,
    "qtgaus_scaling": qtgaus_scaling,
    "custom_scaling_1": custom_scaling_1,
}
scaling_functions_inv = {
    "standard_scaling_inv": standard_scaling_inv,
    "qtgaus_scaling_inv": standard_scaling_inv,
    "custom_scaling_1_inv": custom_scaling_1_inv,
}


class ParquetDataset(Dataset):
    def __init__(
        self,
        files,
        columns,
        nevs=None,
        scale_function="standard_scaling",
        inverse_scale_function="standard_scaling_inv",
        rng=(-5, 5),
        saturate=True,
    ):
        self.columns = columns
        # self.df = dd.read_parquet(files, columns=columns, engine='fastparquet')
        self.df = dd.read_parquet(
            files, columns=columns, engine="fastparquet"
        ).compute()

        if nevs is not None:
            self.df = self.df.iloc[:nevs]

        self.scaling_function = scaling_functions[scale_function]
        self.inverse_scale_function = scaling_functions_inv[inverse_scale_function]

        self.df, self.scalers = self.scaling_function(self.df, saturate=saturate)

    def get_scaled_back_df(self):
        df = self.inverse_scale_function(self.df, self.scalers)
        return df

    def scale_back(self):
        self.df = self.inverse_scale_function(self.df, self.scalers)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
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
            torch.save(
                model.state_dict(),
                f"{path}/epoch_{epoch}_valloss_{vloss:.3f}.pt".replace("-", "m"),
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
