# ---- Imports ---- #
import numpy as np
import pandas as pd
import h5py
import pickle
from lfads_tf2.utils import load_posterior_averages
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, save_to_h5
import sys

# ---- Run params ---- #
dataset_name = 'mc_rtt'
bin_size = 5
suf = '' if bin_size == 5 else '_20'

# ---- Data locations ---- #
datapath_dict = {
    'mc_maze': '~/data/000128/sub-Jenkins/',
    'mc_rtt': '~/data/000129/sub-Indy/',
    'area2_bump': '~/data/000127/sub-Han/',
    'dmfc_rsg': '~/data/000130/sub-Haydn/',
    'mc_maze_large': '~/data/000138/sub-Jenkins/',
    'mc_maze_medium': '~/data/000139/sub-Jenkins/',
    'mc_maze_small': '~/data/000140/sub-Jenkins/',
}
prefix_dict = {
    'mc_maze': '*full',
    'mc_maze_large': '*large',
    'mc_maze_medium': '*medium',
    'mc_maze_small': '*small',
}
datapath = datapath_dict[dataset_name]
prefix = prefix_dict.get(dataset_name, '')
savepath = f'{dataset_name}{suf}_autolfads_submission.h5'

# ---- Load LFADS output ---- #
model_dir = f'~/autolfads/runs/{dataset_name}{suf}/best_model/'
train_rates, train_factors, *_ = load_posterior_averages(model_dir, merge_tv=True, ps_filename='posterior_samples.h5')
test_rates, test_factors, *_ = load_posterior_averages(model_dir, merge_tv=True, ps_filename='posterior_samples_test.h5')

# ---- Load data ---- #
dataset = NWBDataset(datapath, prefix)
dataset.resample(bin_size)

# ---- Find data shapes ---- #
data_dict = make_train_input_tensors(dataset, dataset_name, 'train', return_dict=True, save_file=False)
train_spikes_heldin = data_dict['train_spikes_heldin']
train_spikes_heldout = data_dict['train_spikes_heldout']
num_heldin = train_spikes_heldin.shape[2]
tlen = train_spikes_heldin.shape[1]

# ---- Split LFADS output ---- #
train_rates_heldin = train_rates[:, :tlen, :num_heldin]
train_rates_heldout = train_rates[:, :tlen, num_heldin:]
train_rates_heldin_forward = train_rates[:, tlen:, :num_heldin]
train_rates_heldout_forward = train_rates[:, tlen:, num_heldin:]
eval_rates_heldin = test_rates[:, :tlen, :num_heldin]
eval_rates_heldout = test_rates[:, :tlen, num_heldin:]
eval_rates_heldin_forward = test_rates[:, tlen:, :num_heldin]
eval_rates_heldout_forward = test_rates[:, tlen:, num_heldin:]

# ---- Save output ---- #
output_dict = {
    dataset_name + suf: {
        'train_rates_heldin': train_rates_heldin,
        'train_rates_heldout': train_rates_heldout,
        'eval_rates_heldin': eval_rates_heldin,
        'eval_rates_heldout': eval_rates_heldout,
        'eval_rates_heldin_forward': eval_rates_heldin_forward,
        'eval_rates_heldout_forward': eval_rates_heldout_forward,
    }
}
save_to_h5(output_dict, savepath, overwrite=True)