# NDT

The Neural Data Transformer (NDT) uses an attention mechanism to model neural population activity without recurrence, enabling much faster inference than RNN-based models.
You can read more about it in [Ye et al. 2021](https://www.biorxiv.org/content/10.1101/2021.01.16.426955v2).

The code for running NDT for NLB'21 can be found in the [neural-data-transformers repo](https://github.com/snel-repo/neural-data-transformers). Config files for each dataset can be found in `configs/` and the random searches can be run with `python ray_random.py -e <variant_name>`, as mentioned in the repo's README.

## Dependencies
* [nlb_tools](https://github.com/neurallatents/nlb_tools)
* [neural-data-transformers](https://github.com/snel-repo/neural-data-transformers)