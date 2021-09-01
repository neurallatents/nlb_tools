# GPFA

Gaussian Process Factor Analysis (GPFA) is a classic method of extracting low-dimensional neural trajectories from spiking activity.
You can read more about it in [Yu et al. 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2712272/).

This directory contains files used to optimize GPFA for NLB'21:
* `gpfa_cv_sweep.py` runs a 3-fold cross-validated grid search over certain parameter values.
* `run_gpfa.py` runs GPFA and generates a submission for NLB'21. The best parameters found by `gpfa_cv_sweep.py` are stored in `default_dict` in the file.

## Dependencies
* [nlb_tools](https://github.com/neurallatents/nlb_tools)
* [elephant](https://github.com/NeuralEnsemble/elephant)
* sklearn>=0.23