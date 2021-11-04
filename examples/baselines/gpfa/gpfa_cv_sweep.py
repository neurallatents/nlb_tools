import numpy as np
import pandas as pd
import quantities as pq
import h5py
import neo
import gc
from elephant.gpfa import GPFA
from sklearn.linear_model import PoissonRegressor, Ridge
from itertools import product

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

# ---- Default sweep ranges ---- #
latent_dim_dict = {
    'mc_maze': np.linspace(32, 52, 6),
    'mc_maze_large': np.linspace(32, 52, 6),
    'mc_maze_medium': np.linspace(16, 36, 6),
    'mc_maze_small': np.linspace(12, 22, 6),
    'mc_rtt': np.linspace(24, 44, 6),
    'area2_bump': np.linspace(14, 26, 7),
    'dmfc_rsg': np.linspace(20, 40, 6),
}

# ---- Run Params ---- #
dataset_name = "mc_rtt" # one of {'area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_rtt', 
                            # 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
bin_size_ms = 5
cv_fold = 3
latent_dims = latent_dim_dict[dataset_name]
alpha1s = [0.0, 0.0001, 0.001, 0.01]
alpha2s = [0.0, 0.0001, 0.001, 0.01]

# ---- Useful variables ---- #
binsuf = '' if bin_size_ms == 5 else f'_{bin_size_ms}'
dskey = f'mc_maze_scaling{binsuf}_split' if 'maze_' in dataset_name else (dataset_name + binsuf + "_split")
pref_dict = {'mc_maze_small': '[100] ', 'mc_maze_medium': '[250] ', 'mc_maze_large': '[500] '}
bpskey = pref_dict.get(dataset_name, '') + 'co-bps'

# ---- Data locations ----#
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

# ---- Load data ---- #
dataset = NWBDataset(datapath, prefix, skip_fields=['hand_pos', 'cursor_pos', 'force', 'eye_pos', 'muscle_vel', 'muscle_len', 'joint_vel', 'joint_ang'])
dataset.resample(bin_size_ms)

# ---- Prepare n folds ---- #
all_mask = np.isin(dataset.trial_info.split, ['train', 'val'])
all_idx = np.arange(all_mask.shape[0])[all_mask]
train_masks = []
eval_masks = []
for i in range(cv_fold):
    eval_idx = all_idx[i::cv_fold] # take every n samples for each fold
    train_idx = all_idx[~np.isin(all_idx, eval_idx)]
    train_masks.append(np.isin(np.arange(all_mask.shape[0]), train_idx))
    eval_masks.append(np.isin(np.arange(all_mask.shape[0]), eval_idx))

# ---- Conversion helper ---- #
def array_to_spiketrains(array, bin_size):
    """Convert B x T x N spiking array to list of list of SpikeTrains"""
    stList = []
    for trial in range(len(array)):
        trialList = []
        for channel in range(array.shape[2]):
            times = np.nonzero(array[trial, :, channel])[0]
            counts = array[trial, times, channel].astype(int)
            times = np.repeat(times, counts)
            st = neo.SpikeTrain(times*bin_size*pq.ms, t_stop=array.shape[1]*bin_size*pq.ms)
            trialList.append(st)
        stList.append(trialList)
    return stList

# ---- Extract data for each fold ---- #
fold_data = []
for i in range(cv_fold):
    train_dict = make_train_input_tensors(dataset, dataset_name, train_masks[i], save_file=False)
    eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_masks[i], save_file=False)

    train_spikes_heldin = train_dict['train_spikes_heldin']
    train_spikes_heldout = train_dict['train_spikes_heldout']
    eval_spikes_heldin = eval_dict['eval_spikes_heldin']

    train_st_heldin = array_to_spiketrains(train_spikes_heldin, bin_size=bin_size_ms)
    eval_st_heldin = array_to_spiketrains(eval_spikes_heldin, bin_size=bin_size_ms)

    target_dict = make_eval_target_tensors(dataset, dataset_name, train_masks[i], eval_masks[i], save_file=False, include_psth=True)
    fold_data.append((train_spikes_heldin, train_spikes_heldout, eval_spikes_heldin, train_st_heldin, eval_st_heldin, target_dict))
del dataset
gc.collect()

# ---- Define helpers ---- #
flatten2d = lambda x: x.reshape(-1, x.shape[2])

def fit_poisson(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """Fit Poisson GLM from factors to spikes and return rate predictions"""
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    train_pred = []
    eval_pred = []
    for chan in range(train_out.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_in, train_out[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
            oldmax = pr.max_iter
            del pr
            pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
            pr.fit(train_in, train_out[:, chan])
        train_pred.append(pr.predict(train_factors_s))
        eval_pred.append(pr.predict(eval_factors_s))
    train_rates_s = np.vstack(train_pred).T
    eval_rates_s = np.vstack(eval_pred).T
    return train_rates_s, eval_rates_s

def fit_rectlin(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0, thresh=1e-10):
    """Fit linear regression from factors to spikes, rectify, and return rate predictions"""
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_in, train_out)
    train_rates_s = ridge.predict(train_factors_s)
    eval_rates_s = ridge.predict(eval_factors_s)
    rect_min = np.min([np.min(train_rates_s[train_rates_s > 0]), np.min(eval_rates_s[eval_rates_s > 0])])
    true_min = np.min([np.min(train_rates_s), np.min(eval_rates_s)])
    train_rates_s[train_rates_s < thresh] = thresh
    eval_rates_s[eval_rates_s < thresh] = thresh
    return train_rates_s, eval_rates_s

# ---- Sweep latent dims ---- #
results = []
for latent_dim in latent_dims:
    print(f"Evaluating latent_dim={latent_dim}")
    fold_gpfa = []
    # ---- n-fold gpfa ---- #
    for n, data in enumerate(fold_data):
        _, _, _, train_st_heldin, eval_st_heldin, _ = data
        gpfa = GPFA(bin_size=(bin_size_ms * pq.ms), x_dim=int(latent_dim))
        train_factors = gpfa.fit_transform(train_st_heldin)
        eval_factors = gpfa.transform(eval_st_heldin)
        train_factors = np.stack([train_factors[i].T for i in range(len(train_factors))])
        eval_factors = np.stack([eval_factors[i].T for i in range(len(eval_factors))])
        fold_gpfa.append((train_factors, eval_factors))

    # ---- Sweep alphas ---- #
    for alpha1, alpha2 in product(alpha1s, alpha2s):
        print(f"Evaluating alpha1={alpha1}, alpha2={alpha2}")
        res_list = []
        for n, (data, gpfa_res) in enumerate(zip(fold_data, fold_gpfa)):
            train_spikes_heldin, train_spikes_heldout, eval_spikes_heldin, train_st_heldin, eval_st_heldin, target_dict = data
            train_factors, eval_factors = gpfa_res

            train_spikes_heldin_s = flatten2d(train_spikes_heldin)
            train_spikes_heldout_s = flatten2d(train_spikes_heldout)
            eval_spikes_heldin_s = flatten2d(eval_spikes_heldin)
            train_factors_s = flatten2d(train_factors)
            eval_factors_s = flatten2d(eval_factors)

            train_rates_heldin_s, eval_rates_heldin_s = fit_rectlin(train_factors_s, eval_factors_s, train_spikes_heldin_s, eval_spikes_heldin_s, alpha=alpha1)
            train_rates_heldout_s, eval_rates_heldout_s = fit_poisson(train_rates_heldin_s, eval_rates_heldin_s, train_spikes_heldout_s, alpha=alpha2)

            train_rates_heldin = train_rates_heldin_s.reshape(train_spikes_heldin.shape)
            train_rates_heldout = train_rates_heldout_s.reshape(train_spikes_heldout.shape)
            eval_rates_heldin = eval_rates_heldin_s.reshape(eval_spikes_heldin.shape)
            eval_rates_heldout = eval_rates_heldout_s.reshape((eval_spikes_heldin.shape[0], eval_spikes_heldin.shape[1], train_spikes_heldout.shape[2]))

            submission = {
                dataset_name + binsuf: {
                    'train_rates_heldin': train_rates_heldin,
                    'train_rates_heldout': train_rates_heldout,
                    'eval_rates_heldin': eval_rates_heldin,
                    'eval_rates_heldout': eval_rates_heldout  
                }
            }

            res = evaluate(target_dict, submission)[0][dskey]
            res_list.append(res)
            print(f"    Fold {n}: " + str(res))
        res = pd.DataFrame(res_list).mean().to_dict()
        print("    Mean: " + str(res))
        res['latent_dim'] = latent_dim
        res['alpha1'] = alpha1
        res['alpha2'] = alpha2
        results.append(res)

# ---- Save results ---- #
results = pd.DataFrame(results)
results.to_csv(f'{dataset_name}{binsuf}_gpfa_cv_sweep.csv')

# ---- Find best parameters ---- #
best_combo = results[bpskey].argmax()
best_latent_dim = results.iloc[best_combo].latent_dim
best_alpha1 = results.iloc[best_combo].alpha1
best_alpha2 = results.iloc[best_combo].alpha2
print(f'Best params: latent_dim={best_latent_dim}, alpha1={alpha1}, alpha2={alpha2}')
