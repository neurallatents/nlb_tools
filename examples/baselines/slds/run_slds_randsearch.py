# ---- Imports ---- #
from ssm.lds import SLDS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
from sklearn.linear_model import PoissonRegressor
from itertools import product
from datetime import datetime
import time
import json
import traceback
import gc
import pickle

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

np.random.seed(1234)

# ---- Default ranges ---- #
default_dict = { # [K range, D range, l2_A range, l2_b range]
    'mc_maze': [('int', 3, 7), ('int', 30, 45), ('log', 3.25, 4.25), ('log', -8, -4)],
    'mc_rtt': [('int', 4, 10), ('int', 20, 32), ('log', 3.25, 4.5), ('log', -8, -4)],
    'area2_bump': [('int', 3, 7), ('int', 12, 24), ('log', 3.25, 4.25), ('log', -8, -4)],
    'dmfc_rsg': [('int', 5, 10), ('int', 20, 35), ('log', 3.75, 5.0), ('log', -8, -4)],
    'mc_maze_large': [('int', 3, 7), ('int', 25, 40), ('log', 3.25, 4.25), ('log', -8, -4)],
    'mc_maze_medium': [('int', 3, 7), ('int', 20, 35), ('log', 3.25, 4.25), ('log', -8, -4)],
    'mc_maze_small': [('int', 3, 7), ('int', 15, 25), ('log', 3.25, 4.25), ('log', -8, -4)],
}

# ---- Constants ---- #
dynamics = "gaussian"
transitions = "standard"
emissions = "poisson"
emission_kwargs = dict(link="softplus")

# ---- Run Params ---- #
dataset_name = 'mc_rtt'
bin_size_ms = 5

n_runs = 20

train_subset_size = 100
eval_subset_size = 100
num_init_iters = 50
num_train_iters = 50
num_repeats = 2

init_param_values = {
    'K': default_dict[dataset_name][0], # num states
    'D': default_dict[dataset_name][1], # num factors
    'dynamics_kwargs.l2_penalty_A': default_dict[dataset_name][2],
    'dynamics_kwargs.l2_penalty_b': default_dict[dataset_name][3],
}

fit_param_values = {
    'alpha': ('float', 0.2, 0.2),
}

# ---- Full sweep matrix ---- #
def unpack_nested(param_dict):
    new_dict = param_dict.copy()
    for k, v in param_dict.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                new_dict[k + '.' + nk] = nv
            new_dict.pop(k)
    return new_dict

def dict_sample(param_dict, num_samples):
    keys = list(param_dict.keys())
    vals = list(param_dict.values())

    def sample(tup):
        if tup[0] == 'float':
            return np.random.uniform(low=tup[1], high=tup[2], size=num_samples).tolist()
        elif tup[0] == 'int':
            return np.random.randint(low=tup[1], high=tup[2], size=num_samples).tolist()
        elif tup[0] == 'log':
            return np.power(10, np.random.uniform(low=tup[1], high=tup[2], size=num_samples)).tolist()
        else:
            raise ValueError("Unsupported sampling method")

    vals = [sample(val) for val in vals]
    combs = zip(*vals)

    def make_dict(keys, vals):
        d = {}
        for k, v in zip(keys, vals):
            if '.' in k:
                assert k.count('.') == 1, 'cannot handle nesting of depth 2'
                k_p, k_c = k.split('.')
                if k_p not in d:
                    d[k_p] = {}
                d[k_p][k_c] = v
            else:
                d[k] = v
        return d
    
    prod = [make_dict(keys, vals) for vals in combs]
    return prod

init_param_list = dict_sample(init_param_values, n_runs)
fit_param_list = dict_sample(fit_param_values, n_runs)

# ---- Load data ---- #
print("Loading dataset...")
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

dataset = NWBDataset(datapath, prefix)
dataset.resample(bin_size_ms)

binsuf = '' if bin_size_ms == 5 else '_20'
scaling_tdict = {
    'mc_maze_small': '[100] ',
    'mc_maze_medium': '[250] ',
    'mc_maze_large': '[500] ',
}
dskey = f'mc_maze_scaling{binsuf}_split' if 'maze_' in dataset_name else dataset_name + binsuf + "_split"
bpskey = scaling_tdict[dataset_name] + 'co-bps' if 'maze_' in dataset_name else 'co-bps'
deckey = scaling_tdict[dataset_name] + 'vel R2' if 'maze_' in dataset_name else 'tp Corr' if 'dmfc' in dataset_name else 'vel R2'

# ---- Prep Input ---- #
print("Preparing input...")

valid_mask = (dataset.trial_info.split != 'none').to_numpy()
good_trials = valid_mask.nonzero()[0]

trial_sels = [np.random.choice(good_trials, train_subset_size + eval_subset_size, replace=False) for _ in range(num_repeats)]
train_splits = [np.isin(np.arange(len(valid_mask)), ts[:train_subset_size]) for ts in trial_sels]
eval_splits = [np.isin(np.arange(len(valid_mask)), ts[train_subset_size:]) for ts in trial_sels]

train_datas = []
eval_datas = []
target_datas = []
for ts, es in zip(train_splits, eval_splits):
    train_dict = make_train_input_tensors(dataset, dataset_name, ts, save_file=False, include_forward_pred=True)
    eval_dict = make_eval_input_tensors(dataset, dataset_name, es, save_file=False)
    target_dict = make_eval_target_tensors(dataset, dataset_name, ts, es, save_file=False, include_psth=('rtt' not in dataset_name))

    train_spikes_heldin = train_dict['train_spikes_heldin']
    train_spikes_heldout = train_dict['train_spikes_heldout']
    train_spikes_heldin_fp = train_dict['train_spikes_heldin_forward']
    train_spikes_heldout_fp = train_dict['train_spikes_heldout_forward']
    train_spikes = np.concatenate([
        np.concatenate([train_spikes_heldin, train_spikes_heldin_fp], axis=1),
        np.concatenate([train_spikes_heldout, train_spikes_heldout_fp], axis=1),
    ], axis=2)

    eval_spikes_heldin = eval_dict['eval_spikes_heldin']
    eval_spikes = np.full((eval_spikes_heldin.shape[0], train_spikes.shape[1], train_spikes.shape[2]), 0.0)
    masks = np.full((eval_spikes_heldin.shape[0], train_spikes.shape[1], train_spikes.shape[2]), False)
    eval_spikes[:, :eval_spikes_heldin.shape[1], :eval_spikes_heldin.shape[2]] = eval_spikes_heldin
    masks[:, :eval_spikes_heldin.shape[1], :eval_spikes_heldin.shape[2]] = True

    train_spklist = [train_spikes[i, :, :].astype(int) for i in range(len(train_spikes))]
    eval_spklist = [eval_spikes[i, :, :].astype(int) for i in range(len(eval_spikes))]
    eval_masks = [masks[i, :, :] for i in range(len(masks))]

    train_datas.append((train_spklist, None))
    eval_datas.append((eval_spklist, eval_masks))
    target_datas.append(target_dict)

numheldin = train_spikes_heldin.shape[2]
tlen = train_spikes_heldin.shape[1]

def make_inputs(slds, datas):
    datas, inputs, masks, tags = slds.prep_inputs(datas=datas[0], masks=datas[1])
    tensors = {
        'datas': datas,
        'inputs': inputs,
        'masks': masks,
        'tags': tags
    }
    return tensors

temp_slds = SLDS(2, 1, 1)

train_tensors = [make_inputs(temp_slds, d) for d in train_datas]
eval_tensors = [make_inputs(temp_slds, d) for d in eval_datas]

del dataset, temp_slds
del train_dict, train_spikes_heldin, train_spikes_heldout, train_spikes_heldin_fp, train_spikes_heldout_fp
del eval_dict, eval_spikes_heldin, eval_spikes
del train_datas, eval_datas
del masks, eval_masks, train_spklist, eval_spklist
gc.collect()

# ---- Define slds Wrapper ---- #
def run_slds(init_params, fit_params, train_datas, eval_datas):
    N = train_datas['datas'].shape[2]
    slds = SLDS(N=N,
        transitions=transitions,
        emissions=emissions,
        emission_kwargs=emission_kwargs,
        **init_params,
    )
    
    slds.initialize(
        verbose=2,
        num_init_iters=num_init_iters,
        **train_datas,
    )

    q_elbos_lem_train, q_lem_train, *_ = slds.fit(
        method="laplace_em",
        variational_posterior="structured_meanfield",
        initialize=False,
        num_iters=num_train_iters, # score=True,
        **train_datas,
        **fit_params,
    )

    q_elbos_lem_eval, q_lem_eval, *_ = slds.approximate_posterior(
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=num_train_iters,
        **eval_datas,
        **fit_params,
    )

    train_rates = slds.smooth_3d(q_lem_train.mean_continuous_states, **train_datas).cpu().numpy()
    eval_rates = slds.smooth_3d(q_lem_eval.mean_continuous_states, **eval_datas).cpu().numpy()

    train_factors = q_lem_train.mean_continuous_states.cpu().numpy()
    eval_factors = q_lem_eval.mean_continuous_states.cpu().numpy()

    del slds
    del q_lem_train
    del q_lem_eval
    gc.collect()

    return (train_rates, eval_rates)

def dict_mean(dict_list):
    num_dicts = len(dict_list)
    if num_dicts == 0:
        return []
    if num_dicts == 1:
        return dict_list[0]
    mean_dict = {}
    for d in dict_list:
        for key, val in d.items():
            prev = mean_dict.get(key, 0)
            mean_dict[key] = prev + val / num_dicts
    return mean_dict

# ---- Run Sweep ---- # 
res_list = []

search_name = f"./{dataset_name}_runs/search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f'Starting {n_runs} runs...')

i = 0
num_restarts = 0
best_slds = None
best_bps = 0

while i < n_runs:
    init_params = dict_sample(init_param_values, 1)[0]
    fit_params = dict_sample(fit_param_values, 1)[0]
    print(f"Run {i}:\n  init_params: {init_params}\n  fit_params: {fit_params}")
    sub_list = []
    for n in range(num_repeats):
        try:
            (train_rates, eval_rates) = run_slds(init_params, fit_params, train_tensors[n], eval_tensors[n])
        except:
            print('Run failed!')
            continue

        # Reshape output
        train_rates_heldin = train_rates[:, :tlen, :numheldin]
        train_rates_heldout = train_rates[:, :tlen, numheldin:]
        eval_rates_heldin = eval_rates[:, :tlen, :numheldin]
        eval_rates_heldout = eval_rates[:, :tlen, numheldin:]
        eval_rates_heldin_forward = eval_rates[:, tlen:, :numheldin]
        eval_rates_heldout_forward = eval_rates[:, tlen:, numheldin:]

        submission_dict = {
            dataset_name + binsuf: {
                'train_rates_heldin': train_rates_heldin,
                'train_rates_heldout': train_rates_heldout,
                'eval_rates_heldin': eval_rates_heldin,
                'eval_rates_heldout': eval_rates_heldout,
                'eval_rates_heldin_forward': eval_rates_heldin_forward,
                'eval_rates_heldout_forward': eval_rates_heldout_forward,
            }
        }

        res = evaluate(target_datas[n], submission_dict)[0][dskey]
        sub_list.append(res)
    
    if not sub_list:
        i += 1
        continue
    res = dict_mean(sub_list)
    res['run_idx'] = i
    res.update(unpack_nested(fit_params))
    res.update(unpack_nested(init_params))
    res_list.append(res)

    del submission_dict, train_rates, eval_rates
    del train_rates_heldin, train_rates_heldout, eval_rates_heldin, eval_rates_heldout, eval_rates_heldin_forward, eval_rates_heldout_forward
    gc.collect()

    print('')
    time.sleep(10) # rest between models
    i += 1

del train_tensors, eval_tensors

# ---- Save results ---- #
results = pd.DataFrame(res_list)
results.to_csv(search_name + '_results.csv')

