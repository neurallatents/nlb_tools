# nlb_tools
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
The package can be installed with the following commands:
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
* our [EvalAI challenge](https://eval.ai) is where submissions are evaluated and displayed on the leaderboard
* our [paper]() goes into more detail about our motivations and philosophy and technical details
