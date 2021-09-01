# SLDS

SLDS is a method that infers latent states that evolve according to multiple distinct linear dynamical systems that switch with each other over time, allowing for the approximation of complex non-linear dynamics.
You can read more about it in [Linderman et al. 2016](https://arxiv.org/abs/1610.08466).

This directory contains files used to optimize SLDS for NLB'21:
* `run_slds_randsearch.py` runs a random search over certain parameter values using a portion of the training data.
* `run_slds.py` runs SLDS and generates a submission for NLB'21. The parameters in `default_dict` in the file were found by a combination of the random search and manual tuning.

## Dependencies
* [nlb_tools](https://github.com/neurallatents/nlb_tools)
* [ssm](https://github.com/felixp8/ssm)