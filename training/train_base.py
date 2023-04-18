import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl
import torch.multiprocessing as mp

from utils import (
    ddp_setup,
    ParquetDataset,
    get_conditional_base_flow,
    sample_and_plot_base,
    create_mixture_flow_model,
    load_mixture_model,
    save_model,
)


def train_base(device, cfg, world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    if world_size is not None:
        ddp_setup(device, world_size)

    device_id = device_ids[device] if device_ids is not None else device

    # create (and load) the model
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    flow_params_dct = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": {
            "num_steps_maf": cfg.model.maf.num_steps,
            "num_steps_arqs": cfg.model.arqs.num_steps,
            "num_transform_blocks_maf": cfg.model.maf.num_transform_blocks,
            "num_transform_blocks_arqs": cfg.model.arqs.num_transform_blocks,
            "activation": cfg.model.activation,
            "dropout_probability_maf": cfg.model.maf.dropout_probability,
            "dropout_probability_arqs": cfg.model.arqs.dropout_probability,
            "use_residual_blocks_maf": cfg.model.maf.use_residual_blocks,
            "use_residual_blocks_arqs": cfg.model.arqs.use_residual_blocks,
            "batch_norm_maf": cfg.model.maf.batch_norm,
            "batch_norm_arqs": cfg.model.arqs.batch_norm,
            "num_bins_arqs": cfg.model.arqs.num_bins,
            "tail_bound_arqs": cfg.model.arqs.tail_bound,
            "hidden_dim_maf": cfg.model.maf.hidden_dim,
            "hidden_dim_arqs": cfg.model.arqs.hidden_dim,
            "init_identity": cfg.model.init_identity,
        },
        "transform_type": cfg.model.transform_type,
    }
    model = create_mixture_flow_model(**flow_params_dct).to(device)

    if cfg.checkpoint is not None:
        # assume that the checkpoint is path to a directory
        model, _, _, start_epoch, _, _ = load_mixture_model(
            model, model_dir=cfg.checkpoint, filename="checkpoint-latest.pt"
        )
        model = model.to(device)
        print("Loaded model from checkpoint: ", cfg.checkpoint)
        print("Resuming from epoch ", start_epoch)
    else:
        start_epoch = 1

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            # find_unused_parameters=True,
        )
        model = ddp_model.module
    else:
        ddp_model = model
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    # make datasets
    sample = cfg.sample
    calo = cfg.calo

    if sample == "data":
        if calo == "eb":
            train_file = "../../../preprocess/preprocessed/data_eb_train.parquet"
            test_file = "../../../preprocess/preprocessed/data_eb_test.parquet"
        elif calo == "ee":
            train_file = "../../../preprocess/preprocessed/data_ee_train.parquet"
            test_file = "../../../preprocess/preprocessed/data_ee_test.parquet"
    elif sample == "mc":
        if calo == "eb":
            train_file = "../../../preprocess/preprocessed/mc_eb_train.parquet"
            test_file = "../../../preprocess/preprocessed/mc_eb_test.parquet"
        elif calo == "ee":
            train_file = "../../../preprocess/preprocessed/mc_ee_train.parquet"
            test_file = "../../../preprocess/preprocessed/mc_ee_test.parquet"

    with open(
        f"../../../preprocess/preprocessed/pipelines_{sample}_{calo}.pkl", "rb"
    ) as file:
        pipelines = pkl.load(file)
        pipelines = pipelines[cfg.pipelines]

    train_dataset = ParquetDataset(
        train_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines,
        rows=cfg.train.size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
        # num_workers=2,
        # pin_memory=True,
    )
    test_dataset = ParquetDataset(
        test_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=train_dataset.pipelines,
        rows=cfg.test.size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        # num_workers=2,
        # pin_memory=True,
    )

    # train the model
    writer = SummaryWriter(log_dir="runs")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    train_history, test_history = [], []
    for epoch in range(start_epoch, cfg.epochs + 1):
        if world_size is not None:
            b_sz = len(next(iter(train_loader))[0])
            print(
                f"[GPU{device_id}] | Rank {device} | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}"
            )
            train_loader.sampler.set_epoch(epoch)
        print(f"Epoch {epoch}/{cfg.epochs}:")

        train_losses = []
        test_losses = []
        # train
        start = time.time()
        print("Training...")
        for i, (context, target) in enumerate(train_loader):
            # context, target = context.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()

            loss = -ddp_model(target, context=context)
            loss = loss.mean()
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_train_loss = np.mean(train_losses)
        train_history.append(epoch_train_loss)

        # test
        print("Testing...")
        for i, (context, target) in enumerate(test_loader):
            # context, target = context.to(device), target.to(device)
            with torch.no_grad():
                model.eval()
                loss = -ddp_model(target, context=context)
                loss = loss.mean()
                test_losses.append(loss.item())

        epoch_test_loss = np.mean(test_losses)
        test_history.append(epoch_test_loss)
        if device == 0 or world_size is None:
            writer.add_scalars(
                "Losses", {"train": epoch_train_loss, "val": epoch_test_loss}, epoch
            )

        # sample and validation
        if epoch % cfg.sample_every == 0 or epoch == 1:
            print("Sampling and plotting...")
            sample_and_plot_base(
                test_loader=test_loader,
                model=model,
                epoch=epoch,
                writer=writer,
                context_variables=cfg.context_variables,
                target_variables=cfg.target_variables,
                device=device,
            )

        duration = time.time() - start
        print(
            f"Epoch {epoch} | GPU{device_id} | Rank {device} - train loss: {epoch_train_loss:.4f} - val loss: {epoch_test_loss:.4f} - time: {duration:.2f}s"
        )

        if device == 0 or world_size is None:
            save_model(
                epoch,
                ddp_model,
                scheduler,
                train_history,
                test_history,
                name="model",
                model_dir=".",
                optimizer=optimizer,
                is_ddp=world_size is not None,
                save_both=epoch % cfg.sample_every == 0,
            )

    writer.close()


@hydra.main(version_base=None, config_path="config_base", config_name="cfg_test")
def main(cfg):
    # This because in the hydra config we enable the feature that changes the cwd to the experiment dir
    initial_dir = get_original_cwd()
    print("Initial dir: ", initial_dir)
    print("Current dir: ", os.getcwd())

    # save the config
    cfg_name = HydraConfig.get().job.name
    with open(f"{os.getcwd()}/{cfg_name}.yaml", "w") as file:
        OmegaConf.save(config=cfg, f=file)

    env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_var:
        actual_devices = env_var.split(",")
        actual_devices = [int(d) for d in actual_devices]
    else:
        actual_devices = list(range(torch.cuda.device_count()))
    print("Actual devices: ", actual_devices)

    print("Training with cfg: \n", OmegaConf.to_yaml(cfg))
    if cfg.distributed:
        world_size = torch.cuda.device_count()
        # make a dictionary with k: rank, v: actual device
        dev_dct = {i: actual_devices[i] for i in range(world_size)}
        print(f"Devices dict: {dev_dct}")
        mp.spawn(
            train_base,
            args=(cfg, world_size, dev_dct),
            nprocs=world_size,
            join=True,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_base(device, cfg)


if __name__ == "__main__":
    main()
