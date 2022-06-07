# AutoLFADS

Latent Factor Analysis via Dynamical Systems (LFADS) is a deep learning method to infer latent dynamics from single-trial neural spiking data.
AutoLFADS utilizes Population Based Training (PBT) to optimize LFADS hyperparameters efficiently.
You can read more about LFADS in [Pandarinath et al. 2018](https://www.nature.com/articles/s41592-018-0109-9) and AutoLFADS in [Keshtkaran et al. 2021](https://www.biorxiv.org/content/10.1101/2021.01.13.426570v1)

This directory contains files used to run AutoLFADS for NLB'21:
* `lfads_data_prep.py` saves input data in the expected format for LFADS.
* `run_lfads.py` trains AutoLFADS on the training data and performs inference on test data.
* `post_lfads_prep.py` takes LFADS output and reformats it in the expected submission format for NLB'21.
* `config/` contains the run config YAML files used to run AutoLFADS

## Dependencies
* [nlb_tools](https://github.com/neurallatents/nlb_tools)
* [autolfads-tf2](https://github.com/snel-repo/autolfads-tf2)
