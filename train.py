"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion

from data_loading import (
    get_processed_data_loader,
    feature_size,
    window_and_relative_time,
    load_and_process_beatmap,
)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group(args.dist)
    world_size = dist.get_world_size()
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(
            args.results_dir,
            exist_ok=True,
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace(
            "/",
            "-",
        )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        checkpoint_dir = ""
        logger = create_logger(None, rank)

    # Create model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        context_size=feature_size - 3 + 128,
        class_dropout_prob=0.2,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema: torch.nn.Module = deepcopy(model).to(
        device,
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=args.noise_schedule,
        use_l1=args.l1_loss,
    )  # default: 1000 steps, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Setup data:
    subset_ids = None
    if args.fine_tune_ids is not None:
        import pandas as pd

        subset_ids = pd.read_csv(args.fine_tune_ids)["BeatmapID"].tolist()

    global_start = args.data_start
    global_end = args.data_end
    per_rank = int(np.ceil((global_end - global_start) / float(world_size)))
    dataset_start = global_start + rank * per_rank
    dataset_end = min(dataset_start + per_rank, global_end)
    batch_size = int(args.global_batch_size // world_size)

    loader = get_processed_data_loader(
        dataset_path=args.data_path,
        start=dataset_start,
        end=dataset_end,
        seq_len=args.seq_len,
        stride=args.stride,
        cycle_length=batch_size // 2,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        subset_ids=subset_ids,
        seq_func=load_and_process_beatmap,
        win_func=window_and_relative_time,
    )
    logger.info(
        f"Dataset contains {(dataset_end - dataset_start):,} beatmap sets ({args.data_path})",
    )

    # Prepare models for training:
    update_ema(
        ema,
        model.module,
        decay=0,
    )  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Load checkpoint
    if args.ckpt is not None:
        assert os.path.isfile(
            args.ckpt,
        ), f"Could not find DiT checkpoint at {args.ckpt}"
        checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"Restored from checkpoint at {args.ckpt}")

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for (x, o, c), y in loader:
            x = x.to(device)
            o = o.to(device)
            c = c.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.use_amp,
            ):
                model_kwargs = dict(o=o, c=c, y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}",
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=52670)
    parser.add_argument("--data-end", type=int, default=13402)
    parser.add_argument("--data-start", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-B",
    )
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--use-amp", type=bool, default=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--dist", type=str, default="nccl")
    parser.add_argument("--fine-tune-ids", type=str, default=None)
    parser.add_argument("--noise-schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--l1-loss", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
