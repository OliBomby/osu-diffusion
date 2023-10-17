# osu-diffusion
Generate osu! standard beatmap object coordinates using a diffusion model with a transformer backbone.

# How to train your own model
This project is still in development, so things are likely to change.

## Step 1: Create a dataset
Use the `Mapperator.ConsoleApp` in [the ML-extractor branch of Mapperator](https://github.com/mappingtools/Mapperator/tree/ML-extractor) to generate a dataset from your osu! folder.
You need to clone and compile the project using .NET 6.

```
Mapperator.ConsoleApp.exe dataset -m Standard -s Ranked -i 200000 -o "path to output folder"
```

This command generates a dataset with all ranked osu! standard gamemode beatmaps in your folder whose ID is at least 200k.
There are several ways to filter the beatmaps to put in the dataset, so use the help command to figure out the arguments.

## Step 2: Install dependencies
You need PyTorch with CUDA in order to use the GPU to train models. It will not work without GPU.

Versions I used:
- Python 3.10
- PyTorch 2.1.0
- CUDA 11.8

Other dependencies:
- slider
- numpy
- matplotlib
- pandas

## Step 3: Train the model
You train the model using `train.py`. It has several arguments that control the model, data, and hyperparameters.

Important arguments:
- `--data-path` The path to your dataset.
- `--num-classes` The number of beatmaps in your dataset.
- `--data-start` The start index for range of mapsets in the dataset.
- `--data-end` The end index for range of mapsets in the dataset. Not inclusive.
- `--model` The model to train. There are 4 models with increasing sizes.
- `--global-batch-size` The combined batch size over all GPUs.
- `--num-workers` The number of parallel data loading processes per GPU.
- `--ckpt-every` The number of training steps between checkpoints.
- `--seq-len` The length of subsequences of the beatmap for training examples. Determines the context size.
- `--stride` The distance between windows during data loading. Bigger stride means smaller epochs.
- `--ckpt` Path to a checkpoint file to resume training from.
- `--dist` The distribution strategy to use. `gloo` works for Windows.
- `--lr` The learning rate.
- `--relearn-embeds` Forget the learnt embeddings in the `ckpt` and learn a new embedding table. Important if you train on a different dataset from what your checkpoint was trained on.

`--nproc-per-node` determines the number of GPUs to use.

```
torchrun --nproc-per-node=1 train.py --data-path "..\all_ranked_sets_ever" --model DiT-B --num-workers 4 --epochs 100 --global-batch-size 256 --ckpt-every 20000 --seq-len 128 --dist gloo
```

## Step 4: Sample new beatmaps
`sample.py` lets you generate new beatmaps using the rhythm and spacing from an existing beatmap.
You can also provide a specific style to map in by providing the beatmap ID of a map in the training data.

First generate `beatmap_idx.pickle` for your dataset by running `generate_beatmap_idx.py` in the testing folder.
You need to edit the path to your dataset in the script before running.

Then run `sample.py`:

```
python sample.py --beatmap "path to beatmap" --ckpt "results\DiT-B-00\0400000.pt"
```

Important arguments:
- `--beatmap` The beatmap to take rhythm and spacing from.
- `--ckpt` The training checkpoint to use.
- `--model` The model corresponding to the checkpoint.
- `--num-classes` The number of beatmaps in the dataset used to train the model.
- `--beatmap_idx` Path to the `beatmap_idx.pickle` file specific to your dataset.
- `--num-sampling-steps` The number of diffusion steps. Should be between 1 and 1000.
- `--cfg-scale` Scalar for classifier-free guidance. Amplifies the effect of the style transfer. 1.0 for normal style.
- `--style-id` The beatmap ID of the beatmap in the training data to use the style from.
