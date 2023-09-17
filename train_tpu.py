"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import os
from glob import glob

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

from data_loading import context_size
from data_loading import get_processed_data_loader
from diffusion import create_diffusion
from models import DiT_models


#################################################################################
#                                  Training Loop                                #
#################################################################################


def train(args, wrapped_model):
    """
    Trains a new DiT model.
    """

    # Setup DDP:
    assert (
        args.global_batch_size % xm.xrt_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = xm.get_ordinal()
    device = xm.xla_device()

    seed = args.global_seed * xm.xrt_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={xm.xrt_world_size()}.")

    # Setup an experiment folder:
    checkpoint_dir = ""
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
        print(f"[xla:{rank}] Experiment directory created at {experiment_dir}")

    model = wrapped_model.to(device)

    diffusion = create_diffusion(
        timestep_respacing="",
    )  # default: 1000 steps, linear noise schedule
    print(
        f"[xla:{rank}] DiT Parameters: {sum(p.numel() for p in model.parameters()):,}",
    )

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # Scale learning rate to world size
    lr = 1e-4 * xm.xrt_world_size()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # Setup data:
    global_start = args.data_start
    global_end = args.data_end
    per_rank = int(np.ceil((global_end - global_start) / float(xm.xrt_world_size())))
    dataset_start = global_start + rank * per_rank
    dataset_end = min(dataset_start + per_rank, global_end)

    batch_size = int(args.global_batch_size // xm.xrt_world_size())

    loader = get_processed_data_loader(
        dataset_path=args.data_path,
        start=dataset_start,
        end=dataset_end,
        seq_len=args.seq_len,
        stride=args.stride,
        cycle_length=batch_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    print(
        f"[xla:{rank}] Dataset contains {(dataset_end - dataset_start):,} beatmap sets ({args.data_path})",
    )

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0

    print(f"[xla:{rank}] Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        tracker = xm.RateTracker()
        print(f"[xla:{rank}] Beginning epoch {epoch}...")

        mp_device_loader = pl.MpDeviceLoader(loader, device)
        for x, c, y in mp_device_loader:
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            opt.zero_grad(set_to_none=True)
            model_kwargs = dict(c=c, y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()
            xm.optimizer_step(opt)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            tracker.add(batch_size)
            if train_steps % args.log_every == 0:
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                reduce_avg_loss = xm.all_reduce(
                    xm.REDUCE_SUM,
                    avg_loss,
                    1 / xm.xrt_world_size(),
                )
                print(
                    f"[xla:{rank}]({train_steps}) Loss={reduce_avg_loss:.5f} Rate={tracker.rate():.2f} GlobalRate={tracker.global_rate():.2f}",
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    xm.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    print(f"[xla:{rank}] Done!")


# Start training processes
def _mp_fn(rank, args, wrapped_model):
    torch.set_default_tensor_type("torch.FloatTensor")
    train(args, wrapped_model)


def main(args):
    # Create model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        context_size=context_size,
    )

    wrapped_model = xmp.MpModelWrapper(model)

    xmp.spawn(
        _mp_fn,
        args=(args, wrapped_model),
        nprocs=args.num_cores,
        start_method="fork",
    )


SERIAL_EXEC = xmp.MpSerialExecutor()


if __name__ == "__main__":
    os.environ["XLA_USE_F16"] = "1"

    # Default args here will train DiT-XL with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--data-end", type=int, required=True)
    parser.add_argument("--data-start", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-XL",
    )
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--use-amp", type=bool, default=True)
    parser.add_argument("--num_cores", type=int, default=8)
    args = parser.parse_args()
    main(args)
