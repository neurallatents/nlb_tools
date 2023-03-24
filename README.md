# NLB Codepack (nlb_tools)
Python tools for participating in Neural Latents Benchmark '21.

## Overview
Neural Latents Benchmark '21 (NLB'21) is a benchmark suite for unsupervised modeling of neural population activity.
The suite includes four datasets spanning a variety of brain areas and experiments.
The primary task in the benchmark is co-smoothing, or inference of firing rates of unseen neurons in the population.

This repo contains code to facilitate participation in NLB'21:
* `nlb_tools/` has code to load and preprocess our dataset files, format data for modeling, and locally evaluate results
* `examples/tutorials/` contains tutorial notebooks demonstrating basic usage of `nlb_tools`
* `examples/baselines/` holds the code we used to run our baseline methods. They may serve as helpful references on more extensive usage of `nlb_tools`

## Installation
The package can be installed with the following command:
```
pip install nlb-tools
```
However, to run the tutorial notebooks locally or make any modifications to the code, you should clone the repo. The package can then be installed with the following commands:
```
git clone https://github.com/neurallatents/nlb_tools.git
cd nlb_tools
pip install -e .
```
This package requires Python 3.7+ and was developed in Python 3.7, which is the Python version we recommend you use.

## Getting started
We recommend reading/running through `examples/tutorials/basic_example.ipynb` to learn how to use `nlb_tools` to load and 
format data for our benchmark. You can also find Jupyter notebooks demonstrating running GPFA and SLDS for the benchmark in
`examples/tutorials/`.

## Other resources
For more information on the benchmark:
* our [main webpage](https://neurallatents.github.io) contains general information on our benchmark pipeline and introduces the datasets
* our [EvalAI challenge](https://eval.ai/web/challenges/challenge-page/1256/overview) is where submissions are evaluated and displayed on the leaderboard
* our datasets are available on DANDI: [MC_Maze](https://dandiarchive.org/#/dandiset/000128), [MC_RTT](https://dandiarchive.org/#/dandiset/000129), [Area2_Bump](https://dandiarchive.org/#/dandiset/000127), [DMFC_RSG](https://dandiarchive.org/#/dandiset/000130), [MC_Maze_Large](https://dandiarchive.org/#/dandiset/000138), [MC_Maze_Medium](https://dandiarchive.org/#/dandiset/000139), [MC_Maze_Small](https://dandiarchive.org/#/dandiset/000140)
* our [paper](http://arxiv.org/abs/2109.04463) describes our motivations behind this benchmarking effort as well as various technical details and explanations of design choices made in preparing NLB'21
* our [Slack workspace](https://neurallatents.slack.com) lets you interact directly with the developers and other participants. Please email `fpei6 [at] gatech [dot] edu` for an invite link
