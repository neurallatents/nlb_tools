# ---- Imports ---- #
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors
import numpy as np
import h5py
import sys

# ---- Run params ---- #
dataset_name = 'mc_rtt'
valid_ratio = 0.2
bin_size_ms = 5
suf = '' if bin_size_ms == 5 else '_20'

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
savepath_train = f'~/data/lfads_input/{dataset_name}{suf}_train_lfads.h5'
savepath_test = f'~/data/lfads_input/{dataset_name}{suf}_test_lfads.h5'

# ---- Load data ---- #
dataset = NWBDataset(datapath, prefix)
dataset.resample(bin_size_ms)

# ---- Extract train data ---- #
data_dict = make_train_input_tensors(dataset, dataset_name, ['train', 'val'], save_file=False, include_forward_pred=True)

tlen = data_dict['train_spikes_heldin'].shape[1]
num_heldin = data_dict['train_spikes_heldin'].shape[2]
num_heldout = data_dict['train_spikes_heldout'].shape[2]
fp_steps = data_dict['train_spikes_heldin_forward'].shape[1]
spikes = np.hstack([
    np.dstack([data_dict['train_spikes_heldin'], data_dict['train_spikes_heldout']]),
    np.dstack([data_dict['train_spikes_heldin_forward'], data_dict['train_spikes_heldout_forward']]),
])

num_trials = len(spikes)
valid_inds = np.arange(0, num_trials, int(1./valid_ratio))
train_inds = np.delete(np.arange(num_trials), valid_inds)

with h5py.File(savepath_train, 'w') as h5file:
    h5file.create_dataset('train_inds', data=train_inds)
    h5file.create_dataset('valid_inds', data=valid_inds)
    h5file.create_dataset('train_data', data=spikes[train_inds])
    h5file.create_dataset('valid_data', data=spikes[valid_inds])

# ---- Extract test data ---- #
data_dict = make_eval_input_tensors(dataset, dataset_name, 'test', save_file=False, scramble_trials=('maze' not in dataset_name))
num_trials = len(data_dict['eval_spikes_heldin'])
spikes = np.hstack([
    np.dstack([data_dict['eval_spikes_heldin'], np.full((num_trials, tlen, num_heldout), 0.0)]),
    np.full((num_trials, fp_steps, num_heldin + num_heldout), 0.0),
])
valid_inds = np.arange(0, num_trials, int(1./valid_ratio))
train_inds = np.delete(np.arange(num_trials), valid_inds)

with h5py.File(savepath_test, 'w') as h5file:
    h5file.create_dataset('train_inds', data=train_inds)
    h5file.create_dataset('valid_inds', data=valid_inds)
    h5file.create_dataset('train_data', data=spikes[train_inds])
    h5file.create_dataset('valid_data', data=spikes[valid_inds])

# ---- Print summary ---- #
print(f'heldin: {num_heldin}')
print(f'heldout: {num_heldout}')
print(f'tlen: {tlen}')
print(f'fp_steps: {fp_steps}')
