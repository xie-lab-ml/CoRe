---
license: mit
language:
- en
metrics:
- T2I-Compbench
- GenEval
- PickScore
- AES
- ImageReward
- HPSV2
new_version: v0.1
pipeline_tag: text-to-image
library_name: diffusers
tags:
- inference-enhanced algorithm
- efficiency
- effectiveness
- generalization
- weak-to-strong guidance
---

# The Official Implementation of our Arxiv 2025 paper:

> **[CoRe^2: _Collect, Reflect and Refine_ to Generate Better and Faster](https://arxiv.org/abs/2503.09662)** <br>

Authors:

>**<em>Shitong Shao, Zikai Zhou, Dian Xie, Yuetong Fang, Tian Ye, Lichen Bai</em> and <em>Zeke Xie*</em>** <br>
> xLeaf Lab, HKUST (GZ) <br>
> *: Corresponding author

## New

- [x] Release the inference code of SD3.5 and SDXL.

- [ ] Release the inference code of FLUX.

- [ ] Release the inference code of LlamaGen.

- [ ] Release the implementation of the Collect phase.

- [ ] Release the implementation of the Reflect phase.


## Overview

This guide provides instructions on how to use the CoRe^2.

Here we provide the inference code which supports different models like ***Stable Diffusion XL, Stable Diffusion 3.5 Large.***

## Requirements

- `python version == 3.8`
- `pytorch with cuda version`
- `diffusers`
- `PIL`
- `bitsandbytes`
- `numpy`
- `timm`
- `argparse`
- `einops`

## Installation🚀️

Make sure you have successfully built `python` environment and installed `pytorch` with cuda version. Before running the script, ensure you have all the required packages installed. You can install them using:

```bash
pip install diffusers, PIL, numpy, timm, argparse, einops
```

## Usage👀️ 

To use the CoRe^2 pipeline, you need to run the `sample_img.py` script with appropriate command-line arguments. Below are the available options:

### Command-Line Arguments

- `--pipeline`: Select the model pipeline (`sdxl`, `sd35`). Default is `sdxl`.
- `--prompt`: The textual prompt based on which the image will be generated. Default is "Mickey Mouse painting by Frank Frazetta."
- `--inference-step`: Number of inference steps for the diffusion process. Default is 50.
- `--cfg`: Classifier-free guidance scale. Default is 5.5.
- `--pretrained-path`: Path to the pretrained model weights. Default is a specified path in the script.
- `--size`: The size (height and width) of the generated image. Default is 1024.
- `--method`: Select the inference method (`standard`, `core`, `zigzag`, `z-core`)

### Running the Script

Run the script from the command line by navigating to the directory containing `sample_img.py` and executing:

```
python sample_img.py --pipeline sdxl --prompt "A banana on the left of an apple." --size 1024
```

This command will generate an image based on the prompt using the Stable Diffusion XL model with an image size of 1024x1024 pixels.

### Output🎉️ 

The script will save one image.

