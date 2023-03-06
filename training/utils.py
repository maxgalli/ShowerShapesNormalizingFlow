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


class ParquetDataset(Dataset):
    def __init__(self, files, columns, nevs=None, scaler="minmax", rng=(-5, 5)):
        self.columns = columns
        # self.df = dd.read_parquet(files, columns=columns, engine='fastparquet')
        self.df = dd.read_parquet(
            files, columns=columns, engine="fastparquet"
        ).compute()

        if nevs is not None:
            self.df = self.df.iloc[:nevs]

        # saturate pt
        self.df["probe_pt"] = self.df["probe_pt"].clip(upper=200)
        if "probe_sieie" in self.columns:
            self.df["probe_sieie"] = self.df["probe_sieie"].clip(upper=0.012, lower=0.005)
        if "probe_sieip" in self.columns:
            self.df["probe_sieip"] = self.df["probe_sieip"].clip(upper=0.0001, lower=-0.0001)
        if "probe_etaWidth" in self.columns:
            self.df["probe_etaWidth"] = self.df["probe_etaWidth"].clip(upper=0.025)
        if "probe_phiWidth" in self.columns:
            self.df["probe_phiWidth"] = self.df["probe_phiWidth"].clip(upper=0.125)
        
        # scale pt
        self.pt_transformer = FunctionTransformer(
            np.log1p, inverse_func=np.expm1, validate=True
        )
        pt_scaled = self.pt_transformer.transform(
            self.df["probe_pt"].to_numpy().reshape(-1, 1)
        )
        self.df["probe_pt"] = pt_scaled

        # scale the rest
        y = self.df.values
        if scaler == "minmax":
            self.scaler = MinMaxScaler(rng)
        elif scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "maxabs":
            self.scaler = MaxAbsScaler()
        elif scaler == "robust":
            self.scaler = RobustScaler()
        elif scaler == "poweryeo":
            self.scaler = PowerTransformer(method="yeo-johnson")
        elif scaler == "qtgaus":
            self.scaler = QuantileTransformer(output_distribution="normal")
        self.scaler.fit(y)
        y_scaled = self.scaler.transform(y)
        self.df = pd.DataFrame(y_scaled, columns=self.columns)

    def dump_scaler(self, path):
        dump(self.scaler, path)
        dump(self.pt_transformer, path.replace(".save", "_pt.save"))

    def get_scaled_back_array(self):
        y_scaled = self.df.values
        y = self.scaler.inverse_transform(y_scaled)
        # scale back pt
        pt_partially_scaled = y[:, self.columns.index("probe_pt")].reshape(-1, 1)
        pt_scaled_back = self.pt_transformer.inverse_transform(pt_partially_scaled)
        y[:, self.columns.index("probe_pt")] = pt_scaled_back.reshape(-1)
        return y

    def scale_back(self):
        y_scaled = self.get_scaled_back_array()
        for i, col in enumerate(self.columns):
            self.df[col] = y_scaled[:, i]

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
    pairs = [p for p in itertools.combinations(columns, 2)]
    # 1D plots
    for c in columns:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist(valdataset.df[c], bins=100, range=rng, density=True, histtype="step", label="Data")
        sub_sample = sample[:, :, columns.index(c)]
        x = sub_sample.reshape(sub_sample.shape[0] * sub_sample.shape[1])
        ax.hist(x.cpu().numpy(), bins=100, range=rng, density=True, histtype="step", label="Sampled")
        fig.savefig(f"{path}/epoch_{epoch}_{c}_1D.png")
        if writer is not None:
            writer.add_figure(f"epoch_{epoch}_{c}_1D", fig, epoch)
    for pair in pairs:
        c1, c2 = pair
        print(f"Plotting {pair}")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist2d(
            valdataset.df[c1],
            valdataset.df[c2],
            bins=100,
            range=[rng, rng],
            norm=matplotlib.colors.LogNorm(),
        )
        axs[0].set_xlabel(c1)
        axs[0].set_ylabel(c2)
        axs[0].set_title("Validation data")

        # print(sample.shape)
        # print(columns.index(c1), columns.index(c2))
        # keep only the appropriate columns
        sub_sample = sample[:, :, [columns.index(c1), columns.index(c2)]]
        # print(sample.shape)
        x = sub_sample.reshape(
            sub_sample.shape[0] * sub_sample.shape[1], sub_sample.shape[2]
        )
        # print(x.shape)
        # plt.hist2d(x[:, 0].numpy(), x[:, 1].numpy(), bins=100, range=[[-0.5, 1.5], [-0.2 ,1.2]], norm=matplotlib.colors.LogNorm())
        axs[1].hist2d(
            x[:, 0].cpu().numpy(),
            x[:, 1].cpu().numpy(),
            bins=100,
            range=[rng, rng],
            norm=matplotlib.colors.LogNorm(),
        )
        axs[1].set_xlabel(c1)
        axs[1].set_ylabel(c2)
        axs[1].set_title("Sampled data")
        fig.savefig(f"{path}/epoch_{epoch}_{c1}-{c2}.png")
        if writer is not None:
            writer.add_figure(f"epoch_{c1}-{c2}", fig, epoch)

    # now plot in bins of the condition columns
    nbins = 4
    for column in columns:
        fig, ax = plt.subplots(len(condition_columns), 2, figsize=(10, 5 * nbins))
        for row, cond_column in enumerate(condition_columns):
            # divide into 4 bins
            bins = np.linspace(
                valdataset.df[cond_column].min(),
                valdataset.df[cond_column].max(),
                nbins + 1,
            )
            cond_arr = valdataset.df[cond_column].values
            cond_arr_rep = np.repeat(cond_arr, nsample)
            for left_edge, right_edge in zip(bins[:-1], bins[1:]):
                left_edge_label = f"{left_edge:.2f}"
                right_edge_label = f"{right_edge:.2f}"
                print(
                    f"Plotting {column} in bin {left_edge_label} to {right_edge_label} of {cond_column}"
                )
                # plot valdata
                arr = valdataset.df[
                    (valdataset.df[cond_column] > left_edge)
                    & (valdataset.df[cond_column] < right_edge)
                ]
                ax[row, 0].hist2d(
                    arr[cond_column],
                    arr[column],
                    bins=100,
                    range=[rng, rng],
                    norm=matplotlib.colors.LogNorm(),
                )
                ax[row, 0].set_xlabel(cond_column)
                ax[row, 0].set_ylabel(column)
                ax[row, 0].set_title(
                    f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}"
                )

                # plot sample
                sub_sample = sample[:, :, columns.index(column)]
                x = sub_sample.reshape(sub_sample.shape[0] * sub_sample.shape[1])
                # concatenate cond_arr_rep and x and keep only values in bin
                arr = np.concatenate(
                    (cond_arr_rep.reshape(-1, 1), x.cpu().numpy().reshape(-1, 1)),
                    axis=1,
                )
                arr = arr[(arr[:, 0] > left_edge) & (arr[:, 0] < right_edge)]
                ax[row, 1].hist2d(
                    arr[:, 0],
                    arr[:, 1],
                    bins=100,
                    range=[rng, rng],
                    norm=matplotlib.colors.LogNorm(),
                )
                ax[row, 1].set_xlabel(cond_column)
                ax[row, 1].set_ylabel(column)
                ax[row, 1].set_title(
                    f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}"
                )
        fig.savefig(f"{path}/epoch_{epoch}_{column}_condbins.png")
        if writer is not None:
            writer.add_figure(f"epoch_{column}_condbins", fig, epoch)

    """
    if nsample == 1:
        for column in columns:
            for cond_column in condition_columns:
                print(f"Plotting {column} ratio vs {cond_column}")
                sub_sample = sample[:, :, columns.index(column)]
                x = sub_sample.reshape(sub_sample.shape[0]*sub_sample.shape[1])
                r = x.cpu().numpy() / valdataset.df[column].values
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.hist2d(valdataset.df[cond_column], r, bins=100, range=[None, [-29, 31]], norm=matplotlib.colors.LogNorm())
                ax.set_xlabel(cond_column)
                ax.set_ylabel(f"{column} ratio")
                ax.set_title(f"{column} ratio vs {cond_column}")
                fig.savefig(f"{path}/epoch_{epoch}_{column}_ratio_vs_{cond_column}.png")
    """


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
    pairs = [p for p in itertools.combinations(columns, 2)]

    inputs = torch.tensor(mc_val.dataset.df.values[:, ncond:]).to(device)
    context_l = torch.tensor(mc_val.dataset.df.values[:, :ncond]).to(device)
    context_r = torch.tensor(data_val.dataset.df.values[:, :ncond]).to(device)
    with torch.no_grad():
        # mc_to_data, _ = model.batch_transform(inputs, context_l, context_r, inverse=True, batch_size=10000)
        mc_to_data, _ = model.batch_transform(
            inputs, context_l, target_context=context_r, inverse=False
        )
        # mc_to_data, _ = model.batch_transform(inputs, context_l, target_context=None, inverse=False)

    for pair in pairs:
        c1, c2 = pair
        print(f"Plotting {pair}")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist2d(
            data_val.dataset.df[c1],
            data_val.dataset.df[c2],
            bins=100,
            range=[rng, rng],
            norm=matplotlib.colors.LogNorm(),
        )
        axs[0].set_xlabel(c1)
        axs[0].set_ylabel(c2)
        axs[0].set_title("Validation data")

        # print("DIOMERDAAAAAA")
        print(mc_to_data.shape)
        index_c1 = columns.index(c1)
        index_c2 = columns.index(c2)
        axs[1].hist2d(
            mc_to_data[:, index_c1].cpu().numpy(),
            mc_to_data[:, index_c2].cpu().numpy(),
            bins=100,
            range=[rng, rng],
            norm=matplotlib.colors.LogNorm(),
        )
        axs[1].set_xlabel(c1)
        axs[1].set_ylabel(c2)
        axs[1].set_title("MC to data")
        fig.savefig(f"{path}/epoch_{epoch + 1}_{c1}-{c2}.png")
        if writer is not None:
            writer.add_figure(f"epoch_{c1}-{c2}", fig, epoch + 1)

    # now plot in bins of condition columns
    nbins = 4
    for column in columns:
        fig, ax = plt.subplots(len(condition_columns), 2, figsize=(10, 5 * nbins))
        for row, cond_column in enumerate(condition_columns):
            bins = np.linspace(
                data_val.dataset.df[cond_column].min(),
                data_val.dataset.df[cond_column].max(),
                nbins + 1,
            )
            cond_arr = data_val.dataset.df[cond_column].values
            for left_edge, right_edge in zip(bins[:-1], bins[1:]):
                left_edge_label = f"{left_edge:.2f}"
                right_edge_label = f"{right_edge:.2f}"
                print(
                    f"Plotting {column} in bin {left_edge_label} to {right_edge_label} of {cond_column}"
                )
                # plot valdata
                arr = data_val.dataset.df[
                    (data_val.dataset.df[cond_column] > left_edge)
                    & (data_val.dataset.df[cond_column] < right_edge)
                ]
                ax[row, 0].hist2d(
                    arr[cond_column],
                    arr[column],
                    bins=100,
                    range=[rng, rng],
                    norm=matplotlib.colors.LogNorm(),
                )
                ax[row, 0].set_xlabel(cond_column)
                ax[row, 0].set_ylabel(column)
                ax[row, 0].set_title(
                    f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}"
                )

                # plot sample
                x = mc_to_data[:, columns.index(column)]
                # concatenate cond_arr_rep and x and keep only values in bin
                arr = np.concatenate(
                    (cond_arr.reshape(-1, 1), x.cpu().numpy().reshape(-1, 1)), axis=1
                )
                arr = arr[(arr[:, 0] > left_edge) & (arr[:, 0] < right_edge)]
                ax[row, 1].hist2d(
                    arr[:, 0],
                    arr[:, 1],
                    bins=100,
                    range=[rng, rng],
                    norm=matplotlib.colors.LogNorm(),
                )
                ax[row, 1].set_xlabel(cond_column)
                ax[row, 1].set_ylabel(column)
                ax[row, 1].set_title(
                    f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}"
                )
        fig.savefig(f"{path}/epoch_{epoch + 1}_{column}_condbins.png")
        if writer is not None:
            writer.add_figure(f"epoch_{column}_condbins", fig, epoch + 1)
        
        # and 1D histograms
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # scale back 
        scaled_back_arr_data = data_val.dataset.get_scaled_back_array() 
        scaled_back_df_data = pd.DataFrame(scaled_back_arr_data, columns=data_val.dataset.df.columns)
        scaled_back_arr_mc = mc_val.dataset.get_scaled_back_array()
        scaled_back_df_mc = pd.DataFrame(scaled_back_arr_mc, columns=mc_val.dataset.df.columns)
        copied_mc_val = deepcopy(mc_val)
        for col in columns:
            copied_mc_val.dataset.df[col] = mc_to_data[:, columns.index(col)].cpu().numpy()
        copied_mc_val.dataset.scale_back()
        ax.hist(scaled_back_df_data[column], bins=100, histtype='step', label='valdata', density=True)
        ax.hist(copied_mc_val.dataset.df[column], bins=100, histtype='step', label='valmc', density=True)
        ax.hist(scaled_back_df_mc[column], bins=100, histtype='step', label='mc UNCORR', density=True)
        ax.set_xlabel(column)
        ax.set_ylabel('density')
        ax.legend()
        fig.savefig(f"{path}/epoch_{epoch + 1}_{column}_1D.png")
        if writer is not None:
            writer.add_figure(f"epoch_{column}_1D", fig, epoch + 1)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        print(f"Epoch {epoch + 1}/{n_epochs}")
        tl = []
        tlp = []
        tla = []
        tld = []
        vl = []
        vlp = []
        vla = []
        vld = []
        # print how many batches we have
        print(f"Training batches data: {len(data_train)}")
        print(f"Validation batches data: {len(data_val)}")
        print(f"Training batches mc: {len(mc_train)}")
        print(f"Validation batches mc: {len(mc_val)}")
        print(len(data_train.dataset.df))
        for i, (data, mc) in enumerate(zip(data_train, mc_train)):
            model.train()
            # print(data.shape, mc.shape)
            if i % 2 == 0 + 1 * int(inverse):
                #print("INVERSE False, from mc to data")
                batch = mc.to(device)
                other_batch = data.to(device)
                inv = False
            else:
                #print("INVERSE True, from data to mc")
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
            for inv, (batch, other_batch) in enumerate(
                list(zip([data, mc], [mc, data]))
            ):
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
                epoch + 1,
            )
            writer.add_scalars(
                "logprob",
                {
                    "train": tlp_mean,
                    "val": vlp_mean,
                },
                epoch + 1,
            )
            writer.add_scalars(
                "logabsdet",
                {
                    "train": tla_mean,
                    "val": vla_mean,
                },
                epoch + 1,
            )
            writer.add_scalars(
                "lossdistance",
                {
                    "train": tld_mean,
                    "val": vld_mean,
                },
                epoch + 1,
            )

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
        if epoch%2==0:
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
        if (epoch == n_epochs - 1) or (epoch == n_epochs / 2) or (epoch%5 == 0):
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
