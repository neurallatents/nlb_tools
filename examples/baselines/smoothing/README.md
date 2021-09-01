# Spike smoothing

Spike smoothing is a simple approach to denoising firing rates by convolving spikes with a Gaussian kernel.

This directory contains files used to optimize SLDS for NLB'21:
* `smoothing_cv_sweep.py` runs a 5-fold cross-validated grid search over certain parameter values.
* `run_smoothing.py` runs smoothing and generates a submission for NLB'21. The best parameters found by `smoothing_cv_sweep.py` are stored in `default_dict` in the file.

## Dependencies
* [nlb_tools](https://github.com/neurallatents/nlb_tools)
* sklearn>=0.23
