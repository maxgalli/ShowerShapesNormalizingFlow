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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer

from nflows import transforms, flows
from torch.nn import functional as F

np.random.seed(42)
torch.manual_seed(42)


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
        # scale pt
        self.pt_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
        pt_scaled = self.pt_transformer.transform(self.df['probe_pt'].to_numpy().reshape(-1, 1))

        # scale the rest
        df_no_pt = self.df.drop(columns=['probe_pt'])
        y = df_no_pt.values
        if scaler == "minmax":
            self.scaler = MinMaxScaler(rng)
        elif scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "maxabs":
            self.scaler = MaxAbsScaler()
        elif scaler == "robust":
            self.scaler = RobustScaler()
        elif scaler == "poweryeo":
            self.scaler = PowerTransformer(method='yeo-johnson')
        elif scaler == "qtgaus":
            self.scaler = QuantileTransformer(output_distribution='normal')
        self.scaler.fit(y)
        y_scaled = self.scaler.transform(y)
        # insert pt back in the correct place
        pt_index = self.columns.index('probe_pt')
        y_scaled = np.insert(y_scaled, pt_index, pt_scaled[:, 0], axis=1)
        self.df = pd.DataFrame(y_scaled, columns=self.columns)
    
    def saturate(self, df, column, value):
        df[column] = df[column].clip(upper=value)
        return df

    def dump_scaler(self, path):
        dump(self.scaler, path)
        dump(self.pt_transformer, path.replace(".save", "_pt.save"))

    def get_scaled_back_array(self):
        # scale back pt
        pt_scaled = self.df['probe_pt'].to_numpy().reshape(-1, 1)
        pt = self.pt_transformer.inverse_transform(pt_scaled)
        # scale back the rest
        df_no_pt = self.df.drop(columns=['probe_pt'])
        y_scaled = df_no_pt.values
        y = self.scaler.inverse_transform(y_scaled)
        # insert pt back in the correct place
        pt_index = self.columns.index('probe_pt')
        y = np.insert(y, pt_index, pt[:, 0], axis=1)
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


def dump_validation_plots(flow, valdataset, columns, condition_columns, nsample, device, path, epoch, rng=(-5, 5)):
    epoch = str(epoch + 1) if type(epoch) == int else epoch
    print("Dumping validation plots")
    ncond = len(condition_columns)
    xcond = torch.tensor(valdataset.df.values[:, :ncond].astype(np.float32)).to(device)
    with torch.no_grad():
        sample = flow.sample(nsample, context=xcond)
    pairs = [p for p in itertools.combinations(columns, 2)]
    for pair in pairs:
        c1, c2 = pair
        print(f"Plotting {pair}")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist2d(valdataset.df[c1], valdataset.df[c2], bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
        axs[0].set_xlabel(c1)
        axs[0].set_ylabel(c2)
        axs[0].set_title("Validation data")
    
        #print(sample.shape)
        #print(columns.index(c1), columns.index(c2))
        # keep only the appropriate columns
        sub_sample = sample[:, :, [columns.index(c1), columns.index(c2)]]
        #print(sample.shape)
        x = sub_sample.reshape(sub_sample.shape[0]*sub_sample.shape[1], sub_sample.shape[2])
        #print(x.shape)
        #plt.hist2d(x[:, 0].numpy(), x[:, 1].numpy(), bins=100, range=[[-0.5, 1.5], [-0.2 ,1.2]], norm=matplotlib.colors.LogNorm())
        axs[1].hist2d(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
        axs[1].set_xlabel(c1)
        axs[1].set_ylabel(c2)
        axs[1].set_title("Sampled data")
        fig.savefig(f"{path}/epoch_{epoch}_{c1}-{c2}.png")

    # now plot in bins of the condition columns
    nbins = 4
    for column in columns:
        fig, ax = plt.subplots(len(condition_columns), 2, figsize=(10, 5*nbins))
        for row, cond_column in enumerate(condition_columns):
            # divide into 4 bins 
            bins = np.linspace(valdataset.df[cond_column].min(), valdataset.df[cond_column].max(), nbins+1)
            cond_arr = valdataset.df[cond_column].values
            cond_arr_rep = np.repeat(cond_arr, nsample)
            for left_edge, right_edge in zip(bins[:-1], bins[1:]):
                left_edge_label = f"{left_edge:.2f}"
                right_edge_label = f"{right_edge:.2f}"
                print(f"Plotting {column} in bin {left_edge_label} to {right_edge_label} of {cond_column}")
                # plot valdata
                arr = valdataset.df[(valdataset.df[cond_column] > left_edge) & (valdataset.df[cond_column] < right_edge)]
                ax[row, 0].hist2d(arr[cond_column], arr[column], bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
                ax[row, 0].set_xlabel(cond_column)
                ax[row, 0].set_ylabel(column)
                ax[row, 0].set_title(f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}")

                # plot sample
                sub_sample = sample[:, :, columns.index(column)]
                x = sub_sample.reshape(sub_sample.shape[0]*sub_sample.shape[1])
                # concatenate cond_arr_rep and x and keep only values in bin
                arr = np.concatenate((cond_arr_rep.reshape(-1, 1), x.cpu().numpy().reshape(-1, 1)), axis=1)
                arr = arr[(arr[:, 0] > left_edge) & (arr[:, 0] < right_edge)]
                ax[row, 1].hist2d(arr[:, 0], arr[:, 1], bins=100, range=[rng, rng], norm=matplotlib.colors.LogNorm())
                ax[row, 1].set_xlabel(cond_column)
                ax[row, 1].set_ylabel(column)
                ax[row, 1].set_title(f"{column} in bin {left_edge_label} to {right_edge_label} of {cond_column}")
        fig.savefig(f"{path}/epoch_{epoch}_{column}_condbins.png")

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
    scaler = cfg.general.scaler

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

    rng = (-base_conf.tail_bound, base_conf.tail_bound)
    datadataset = ParquetDataset(files=train_file, columns=all_columns, nevs=nevs, scaler=scaler, rng=rng)
    datadataset.dump_scaler(f"{outputpath_base_str}/{sample}_{calo}_train_scaler.save")

    valdataset = ParquetDataset(files=val_file, columns=all_columns, nevs=nevs, scaler=scaler, rng=rng)
    valdataset.dump_scaler(f"{outputpath_base_str}/{sample}_{calo}_val_scaler.save")
   
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
        n_epochs = base_conf.nepochs
        print(f"Training {cfg.output.name} on {device} with {n_epochs} epochs and learning rate {base_conf.lr}.")
        optimizer = optim.Adam(flow.parameters(), lr=base_conf.lr)
        train_losses = []
        val_losses = []
        best_vloss = np.inf
        
        for epoch in range(n_epochs):
            tl = []
            vl = []
            flow.to(device)
            flow.train()
            for i, x in enumerate(dataloader):
                x_input = torch.tensor(x[:, ncond:], dtype=torch.float32).to(device)
                x_cond = torch.tensor(x[:, :ncond], dtype=torch.float32).to(device)
                #print(x_input.shape)
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=x_input, context=x_cond).mean()
                #loss = -flow.log_prob(inputs=x_input).mean()
                tl.append(loss.item())
                loss.backward()
                optimizer.step()
                #print(loss.item())
            # validation
            for i, x in enumerate(valdataloader):
                x_input = torch.tensor(x[:, ncond:], dtype=torch.float32).to(device)
                x_cond = torch.tensor(x[:, :ncond], dtype=torch.float32).to(device)
                loss = -flow.log_prob(inputs=x_input, context=x_cond).mean()
                vl.append(loss.item())
           
            epoch_vloss = np.mean(vl)
            train_losses.append(np.mean(tl))
            val_losses.append(epoch_vloss)
            print(f"epoch {epoch + 1}: loss = {np.mean(tl)}, val loss = {epoch_vloss}")

            if epoch_vloss < best_vloss:
                print("Saving model")
                torch.save(flow.state_dict(), f'{outputpath_str}/epoch_{epoch + 1}_valloss_{epoch_vloss:.3f}.pt'.replace('-', 'm'))
                best_vloss = epoch_vloss
            else:
                print(f"Validation loss did not improve from {best_vloss:.3f} to {epoch_vloss:.3f}.")

            # dump validation plots only at the end and at the middle of the training
            if (epoch == n_epochs - 1) or (epoch == n_epochs/2):
                dump_validation_plots(flow, valdataset, columns, condition_columns, 1, device, outputpath_str, epoch, rng=rng)

            # plot losses
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(train_losses, label="train")
            ax.plot(val_losses, label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            fig.savefig(f"{outputpath_str}/losses.png")


if __name__ == "__main__":
    main()
