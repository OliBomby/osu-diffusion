# osu-diffusion
Generate osu! standard beatmap object coordinates using a diffusion model with a transformer backbone.

[Showcase video](https://www.youtube.com/watch?v=DdchpYN6pBo)

## Purpose
This project is sort of a successor to [Mapperator](https://github.com/mappingtools/Mapperator) as both can turn a partial beatmap which has just the rhythm and spacing into a fully playable beatmap.
With this, the task of automatically generating a beatmap from mp3 can be split up into two parts, one which generates rhythm and spacing from audio, and this part which turns it into a fully playable beatmap.

The second purpose style-transfer between beatmaps. Because you can extract the required features from existing beatmaps, its possible to have this AI remap any beatmap into another style.

The third purpose was to prove that deep learning AI is capable of modelling more complex geometric relations in beatmaps. Hit object relations come in such a large variety that it is almost impossible to model this algorithmically. In that sense, it provides similar challenges to image recognition and for that deep learning approaches have been shown to be very effective.  

# Using osu-diffusion

## Dependencies
You need PyTorch with CUDA in order to use the GPU to train models. Training will not work without GPU. Sampling can be done without GPU but would be significantly faster with GPU.

Versions I used:
- Python 3.10
- PyTorch 2.1.0
- CUDA 11.8

Other dependencies:
- slider
- numpy
- matplotlib
- pandas


## How to sample new beatmaps
The easiest way to create your own beatmaps is to use our [colab notebook.](https://github.com/OliBomby/osu-diffusion/blob/master/colab/osu_diffusion_sample.ipynb)

***

`sample.py` lets you generate new beatmaps using the rhythm and spacing from an existing beatmap.
You can also provide a specific style to map in by providing the beatmap ID of a map in the training data.

1. Get a trained checkpoints [from here](https://drive.google.com/file/d/1oX8SPNnyswhaI8euWRkJ10tmncxaGsBG/view?usp=sharing) or from training your own model.
2. Generate `beatmap_idx.pickle` for your dataset by running `generate_beatmap_idx.py` in the testing folder. You need to edit the path to your dataset in the script before running. If you downloaded the checkpoint from here, you can use the `beatmap_idx.pickle` that is already present in the repository.
3. Run `sample.py`:

```
python sample.py --beatmap "path to beatmap" --ckpt "DiT-B-00-0700000.pt"
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



## How to train your own model
This project is still in development, so things are likely to change.

### Step 1: Create a dataset
Use the `Mapperator.ConsoleApp` in [Mapperator](https://github.com/mappingtools/Mapperator) to generate a dataset from your osu! folder.
Just grab the latest release and run it using .NET 6.

```
Mapperator.ConsoleApp.exe dataset -m Standard -s Ranked -i 200000 -o "path to output folder"
```

This command generates a dataset with all ranked osu! standard gamemode beatmaps in your folder whose ID is at least 200k.
There are several ways to filter the beatmaps to put in the dataset, so use the help command to figure out the arguments.

***

**Alternatively [you can download the dataset here.](https://drive.google.com/file/d/1hzkDPrjqjkE6xII6OMd4hqW-XE6_SdBT/view?usp=sharing)**

### Step 2: Train the model
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

# How it works

## Model IO

The diffusion model takes a sequence of data points where each data point represents a single thing in a beatmap that has a coordinate. These could for example be a circle, slider head, spinner start, red anchor, catmull anchor, slider end with 2 repeats, etc. Together these data points describe all the hit objects.

A data point contains the following information:

- The time of the datapoint.
- The distance to the previous datapoint.
- The type of the datapoint.

There are the following types:

1. is circle
2. is circle NC
3. is spinner
4. is spinner end
5. is sliderhead
6. is sliderhead NC
7. is bezier anchor
8. is perfect anchor
9. is catmull anchor
10. is red anchor
11. is last anchor
12. is slider end 0 repeat
13. is slider end 1 repeat
14. is slider end 2 repeats
15. is slider end even repeats
16. is slider end uneven repeats

The output of the diffusion model is a sequence which gives the X and Y coordinates for each data point.

## Model architecture

I copied the model architecture from [Scalable Diffusion Models with Transformers](https://github.com/facebookresearch/DiT) and modified it for my purpose.

The main changes are:
- Remove the patchify layers so IO concerns only sequences instead of images.
- Remove the auto-encoder so this is not a latent diffusion model.
- Remove positional embedding and embed the data point time instead of position in sequence.
- Add inputs for the additional information of the data point.
- Add attention masking so sequence length can be extended during sampling without too much loss in quality.

## Training data

Training beatmaps are converted to sequences of data points and then split into overlapping windows with a fixed number of data points.
The time values in a window get a random offset, so the absolute position of a window is unknown while the relative timing stays intact.
Also data augmentation is used to flip the positions of data points horizontally or vertically.

The beatmap ID of the beatmap where the window came from is provided as a class label.
This causes the model to learn embeddings of each beatmap in the training data which are somehow discriptive about how objects are placed.

## Sampling

While training is done on windows of 128 data points, to sample entire beatmaps you generally need more than 128 data points.
Luckily self-attention allows us to freely change the sequence length after training.
To help with this, we added an attention mask which limits attention to only look at a small neighbourhood near each data point instead of the whole sequence.

