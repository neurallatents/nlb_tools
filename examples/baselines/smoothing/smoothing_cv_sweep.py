# ---- Imports ----- #
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, \
    make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate
import h5py
import sys, gc
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.linear_model import PoissonRegressor
from datetime import datetime

# ---- Run Params ---- #
dataset_name = "area2_bump" # one of {'area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_rtt', 
                            # 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
bin_size_ms = 5
kern_sds = np.linspace(30, 60, 4)
alphas = np.logspace(-3, 0, 4)
cv_fold = 5
log_offset = 1e-4 # amount to add before taking log to prevent log(0) error

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
dataset = NWBDataset(datapath, prefix, 
    skip_fields=['hand_pos', 'cursor_pos', 'eye_pos', 'force', 'muscle_vel', 'muscle_len', 'joint_vel', 'joint_ang'])
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

# ---- Extract data for each fold ---- #
fold_data = []
for i in range(cv_fold):
    train_dict = make_train_input_tensors(dataset, dataset_name, train_masks[i], save_file=False)
    eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_masks[i], save_file=False)

    train_spikes_heldin = train_dict['train_spikes_heldin']
    train_spikes_heldout = train_dict['train_spikes_heldout']
    eval_spikes_heldin = eval_dict['eval_spikes_heldin']

    target_dict = make_eval_target_tensors(dataset, dataset_name, train_masks[i], eval_masks[i], include_psth=True, save_file=False)
    fold_data.append((train_spikes_heldin, train_spikes_heldout, eval_spikes_heldin, target_dict))
del dataset
gc.collect()

# ---- Useful shape info ---- #
tlen = fold_data[0][0].shape[1]
num_heldin = fold_data[0][0].shape[2]
num_heldout = fold_data[0][1].shape[2]
results = []

# ---- Define helpers ---- #
flatten2d = lambda x: x.reshape(-1, x.shape[2]) # flattens 3d -> 2d array

def fit_poisson(train_factors_s, test_factors_s, train_spikes_s, test_spikes_s=None, alpha=0.0):
    """Fit Poisson GLM from factors to spikes and return rate predictions"""
    train_in = train_factors_s if test_spikes_s is None else np.vstack([train_factors_s, test_factors_s])
    train_out = train_spikes_s if test_spikes_s is None else np.vstack([train_spikes_s, test_spikes_s])
    train_pred = []
    test_pred = []
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
        test_pred.append(pr.predict(test_factors_s))
    train_rates_s = np.vstack(train_pred).T
    test_rates_s = np.vstack(test_pred).T
    return np.clip(train_rates_s, 1e-9, 1e20), np.clip(test_rates_s, 1e-9, 1e20)

# ---- Sweep kernel std ---- #
for ks in kern_sds:
    print(f"Evaluating kern_sd = {ks}")

    # ---- Prepare smoothing kernel ---- #
    window = signal.gaussian(int(6 * ks / bin_size_ms), int(ks / bin_size_ms), sym=True)
    window /=  np.sum(window)
    def filt(x):
        return np.convolve(x, window, 'same')

    # ---- Sweep GLM alpha ---- #
    for a in alphas:
        print(f"    Evaluating alpha = {a}")
        res_list = []

        # ---- Evaluate each fold ---- #
        for n, data in enumerate(fold_data):

            # ---- Smooth spikes ---- #
            train_spikes_heldin, train_spikes_heldout, eval_spikes_heldin, target_dict = data
            train_spksmth_heldin = np.apply_along_axis(filt, 1, train_spikes_heldin)
            eval_spksmth_heldin = np.apply_along_axis(filt, 1, eval_spikes_heldin)

            # ---- Reshape for regression ---- #
            train_spikes_heldin_s = flatten2d(train_spikes_heldin)
            train_spikes_heldout_s = flatten2d(train_spikes_heldout)
            train_spksmth_heldin_s = flatten2d(train_spksmth_heldin)
            eval_spikes_heldin_s = flatten2d(eval_spikes_heldin)
            eval_spksmth_heldin_s = flatten2d(eval_spksmth_heldin)
            
            # Taking log of smoothed spikes gives better results
            train_lograte_heldin_s = np.log(train_spksmth_heldin_s + log_offset)
            eval_lograte_heldin_s = np.log(eval_spksmth_heldin_s + log_offset)

            # ---- Predict rates ---- #
            train_spksmth_heldout_s, eval_spksmth_heldout_s = fit_poisson(train_lograte_heldin_s, eval_lograte_heldin_s, train_spikes_heldout_s, alpha=a)
            train_spksmth_heldout = train_spksmth_heldout_s.reshape((-1, tlen, num_heldout))
            eval_spksmth_heldout = eval_spksmth_heldout_s.reshape((-1, tlen, num_heldout))

            # OPTIONAL: Also use smoothed spikes for held-in rate predictions
            # train_spksmth_heldin_s, eval_spksmth_heldin_s = fit_poisson(train_lograte_heldin_s, eval_lograte_heldin_s, train_spikes_heldin_s, eval_spikes_heldin_s, alpha=0.0)
            # train_spksmth_heldin = train_spksmth_heldin_s.reshape((-1, tlen, num_heldin))
            # eval_spksmth_heldin = eval_spksmth_heldin_s.reshape((-1, tlen, num_heldin))

            # ---- Prepare output ---- #
            output_dict = {
                dataset_name + binsuf: {
                    'train_rates_heldin': train_spksmth_heldin,
                    'train_rates_heldout': train_spksmth_heldout,
                    'eval_rates_heldin': eval_spksmth_heldin,
                    'eval_rates_heldout': eval_spksmth_heldout
                }
            }

            # ---- Evaluate output ---- #
            res = evaluate(target_dict, output_dict)[0][dskey]
            res_list.append(res)
            print(f"        Fold {n}: " + str(res))

        # ---- Average across folds ---- #
        res = pd.DataFrame(res_list).mean().to_dict()
        print("        Mean: " + str(res))
        res['kern_sd'] = ks
        res['alpha'] = a
        results.append(res)

# ---- Save results ---- #
results = pd.DataFrame(results)
results.to_csv(f'{dataset_name}{binsuf}_smoothing_cv_sweep.csv')

# ---- Find best parameters ---- #
best_combo = results[bpskey].argmax()
best_kern_sd = results.iloc[best_combo].kern_sd
best_alpha = results.iloc[best_combo].alpha
print(f'Best params: kern_sd={best_kern_sd}, alpha={best_alpha}')