# ---- Imports ---- #
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate
import numpy as np
import h5py
import scipy.signal as signal
from sklearn.linear_model import PoissonRegressor

# ---- Default params ---- #
default_dict = { # [kern_sd, alpha]
    'mc_maze': [50, 0.01],
    'mc_rtt': [30, 0.1],
    'area2_bump': [30, 0.01],
    'dmfc_rsg': [60, 0.001],
    'mc_maze_large': [40, 0.1],
    'mc_maze_medium': [60, 0.1],
    'mc_maze_small': [60, 0.1],
}

# ---- Run Params ---- #
dataset_name = "mc_rtt" # one of {'area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_rtt', 
                            # 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
bin_size_ms = 5
kern_sd = default_dict[dataset_name][0]
alpha = default_dict[dataset_name][1]
phase = 'test' # one of {'test', 'val'}
log_offset = 1e-4 # amount to add before taking log to prevent log(0) error

# ---- Useful variables ---- #
binsuf = '' if bin_size_ms == 5 else f'_{bin_size_ms}'
dskey = f'mc_maze_scaling{binsuf}_split' if 'maze_' in dataset_name else (dataset_name + binsuf + "_split")
pref_dict = {'mc_maze_small': '[100] ', 'mc_maze_medium': '[250] ', 'mc_maze_large': '[500] '}
bpskey = pref_dict.get(dataset_name, '') + 'co-bps'

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
savepath = f'{dataset_name}{"" if bin_size_ms == 5 else f"_{bin_size_ms}"}_smoothing_output_{phase}.h5'

# ---- Load data ---- #
dataset = NWBDataset(datapath, prefix, skip_fields=['hand_pos', 'cursor_pos', 'eye_pos', 'muscle_vel', 'muscle_len', 'joint_vel', 'joint_ang', 'force'])
dataset.resample(bin_size_ms)

# ---- Extract data ---- #
if phase == 'val':
    train_split = 'train'
    eval_split = 'val'
else:
    train_split = ['train', 'val']
    eval_split = 'test'
train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False)
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
eval_spikes_heldin = eval_dict['eval_spikes_heldin']

# ---- Useful shape info ---- #
tlen = train_spikes_heldin.shape[1]
num_heldin = train_spikes_heldin.shape[2]
num_heldout = train_spikes_heldout.shape[2]

# ---- Define helpers ---- #
def fit_poisson(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """
    Fit Poisson GLM from factors to spikes and return rate predictions
    """
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
    return np.clip(train_rates_s, 1e-9, 1e20), np.clip(eval_rates_s, 1e-9, 1e20)

# ---- Smooth spikes ---- #
window = signal.gaussian(int(6 * kern_sd / bin_size_ms), int(kern_sd / bin_size_ms), sym=True)
window /=  np.sum(window)
def filt(x):
    return np.convolve(x, window, 'same')
train_spksmth_heldin = np.apply_along_axis(filt, 1, train_spikes_heldin)
eval_spksmth_heldin = np.apply_along_axis(filt, 1, eval_spikes_heldin)

# ---- Reshape for regression ---- #
flatten2d = lambda x: x.reshape(-1, x.shape[2])
train_spksmth_heldin_s = flatten2d(train_spksmth_heldin)
train_spikes_heldin_s = flatten2d(train_spikes_heldin)
train_spikes_heldout_s = flatten2d(train_spikes_heldout)
eval_spikes_heldin_s = flatten2d(eval_spikes_heldin)
eval_spksmth_heldin_s = flatten2d(eval_spksmth_heldin)

# Taking log of smoothed spikes gives better results
train_lograte_heldin_s = np.log(train_spksmth_heldin_s + log_offset)
eval_lograte_heldin_s = np.log(eval_spksmth_heldin_s + log_offset)

# ---- Predict rates ---- #
train_spksmth_heldout_s, eval_spksmth_heldout_s = fit_poisson(train_lograte_heldin_s, eval_lograte_heldin_s, train_spikes_heldout_s, alpha=alpha)
train_spksmth_heldout = train_spksmth_heldout_s.reshape((-1, tlen, num_heldout))
eval_spksmth_heldout = eval_spksmth_heldout_s.reshape((-1, tlen, num_heldout))

# OPTIONAL: Also use smoothed spikes + GLM for held-in rate predictions
# train_spksmth_heldin_s, eval_spksmth_heldin_s = fit_poisson(train_lograte_heldin_s, eval_lograte_heldin_s, train_spikes_heldin_s, eval_spikes_heldin_s, alpha=alpha)
# train_spksmth_heldin = train_spksmth_heldin_s.reshape((-1, tlen, num_heldin))
# eval_spksmth_heldin = eval_spksmth_heldin_s.reshape((-1, tlen, num_heldin))

# ---- Prepare/save output ---- #
output_dict = {
    dataset_name + binsuf: {
        'train_rates_heldin': train_spksmth_heldin,
        'train_rates_heldout': train_spksmth_heldout,
        'eval_rates_heldin': eval_spksmth_heldin,
        'eval_rates_heldout': eval_spksmth_heldout,
    }
}
save_to_h5(output_dict, savepath, overwrite=True)

# ---- Evaluate locally ---- #
if phase == 'val':
    target_dict = make_eval_target_tensors(dataset, dataset_name, train_split, eval_split, save_file=False, include_psth=True)
    print(evaluate(target_dict, output_dict))
