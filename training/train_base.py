import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import pathlib
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
from torch.nn import functional as F
from nflows.distributions.normal import ConditionalDiagonalNormal
from utils import ParquetDataset, BaseFlow, spline_inn, dump_validation_plots
from torch.utils.tensorboard import SummaryWriter

np.random.seed(42)
torch.manual_seed(42)


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
    datadataset = ParquetDataset(
        files=train_file, columns=all_columns, nevs=nevs, scaler=scaler, rng=rng
    )
    datadataset.dump_scaler(f"{outputpath_base_str}/{sample}_{calo}_train_scaler.save")

    valdataset = ParquetDataset(
        files=val_file, columns=all_columns, nevs=nevs, scaler=scaler, rng=rng
    )
    valdataset.dump_scaler(f"{outputpath_base_str}/{sample}_{calo}_val_scaler.save")

    dataloader = DataLoader(datadataset, batch_size=base_conf.batch_size, shuffle=True)
    valdataloader = DataLoader(
        valdataset, batch_size=base_conf.batch_size, shuffle=True
    )

    label = f"{sample}_{calo}"
    outputpath_str = f"{outputpath_base_str}/{label}"
    outputpath = pathlib.Path(outputpath_str)
    outputpath.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(outputpath_str)

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
        ConditionalDiagonalNormal(
            shape=[len(columns)], context_encoder=nn.Linear(ncond, 2 * len(columns))
        ),
    )

    # train
    if pathlib.Path(base_conf.load_path).is_file():
        print(f"Loading base_{label} from model: {base_conf.load_path}")
        flow.load_state_dict(torch.load(base_conf.load_path, map_location=device))
    else:
        n_epochs = base_conf.nepochs
        print(
            f"Training {cfg.output.name} on {device} with {n_epochs} epochs and learning rate {base_conf.lr}."
        )
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
                # print(x_input.shape)
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=x_input, context=x_cond).mean()
                # loss = -flow.log_prob(inputs=x_input).mean()
                tl.append(loss.item())
                loss.backward()
                optimizer.step()
                # print(loss.item())
            # validation
            for i, x in enumerate(valdataloader):
                x_input = torch.tensor(x[:, ncond:], dtype=torch.float32).to(device)
                x_cond = torch.tensor(x[:, :ncond], dtype=torch.float32).to(device)
                loss = -flow.log_prob(inputs=x_input, context=x_cond).mean()
                vl.append(loss.item())

            epoch_tloss = np.mean(tl)
            epoch_vloss = np.mean(vl)
            train_losses.append(epoch_tloss)
            val_losses.append(epoch_vloss)
            writer.add_scalars(
                "losses",
                {"train": epoch_tloss, "val": epoch_vloss},
                epoch + 1,
            )
            print(f"epoch {epoch + 1}: loss = {np.mean(tl)}, val loss = {epoch_vloss}")

            if epoch_vloss < best_vloss:
                print("Saving model")
                torch.save(
                    flow.state_dict(),
                    f"{outputpath_str}/epoch_{epoch + 1}_valloss_{epoch_vloss:.3f}.pt".replace(
                        "-", "m"
                    ),
                )
                best_vloss = epoch_vloss
            else:
                print(
                    f"Validation loss did not improve from {best_vloss:.3f} to {epoch_vloss:.3f}."
                )

            # dump validation plots only at the end and at the middle of the training
            if (epoch == n_epochs - 1) or (epoch == n_epochs / 2) or (epoch%5 == 0):
                dump_validation_plots(
                    flow,
                    valdataset,
                    columns,
                    condition_columns,
                    1,
                    device,
                    outputpath_str,
                    epoch,
                    rng=rng,
                    writer=writer,
                )

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
