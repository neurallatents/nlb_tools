
from ssm.lds import SLDS
import numpy as np
import h5py
from sklearn.linear_model import PoissonRegressor
from datetime import datetime
import gc

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

# ---- Default params ---- #
default_dict = { # [states, factors, dynamics_kwargs]
    'mc_maze': [6, 38, {'l2_penalty_A': 30000.0, 'l2_penalty_b': 6.264734351046042e-05}],
    'mc_rtt': [8, 20, {'l2_penalty_A': 5088.303423769022, 'l2_penalty_b': 2.0595034155496943e-07}],
    'area2_bump': [4, 15, {'l2_penalty_A': 10582.770724811768, 'l2_penalty_b': 3.982037833098992e-05}],
    'dmfc_rsg': [10, 30, {'l2_penalty_A': 30000.0, 'l2_penalty_b': 1e-05}],
    'mc_maze_large': [4, 28, {'l2_penalty_A': 5462.032425984561, 'l2_penalty_b': 2.1670446099229413e-05}],
    'mc_maze_medium': [3, 20, {'l2_penalty_A': 2391.229442269956, 'l2_penalty_b': 1.258022745020434e-05}],
    'mc_maze_small': [5, 15, {'l2_penalty_A': 5837.898552701826, 'l2_penalty_b': 1.2150060110686535e-08}],
}

# ---- Run Params ---- #
dataset_name = "area2_bump" # one of {'area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_rtt', 
                            # 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
bin_size_ms = 5
# replace defaults with other values if desired
# defaults are not optimal for 20 ms resolution
states = default_dict[dataset_name][0]
factors = default_dict[dataset_name][1]
dynamics_kwargs = default_dict[dataset_name][2]
alpha = 0.2
num_iters = 50
phase = 'test' # one of {'test', 'val'}

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

# ---- Load data ---- #
dataset = NWBDataset(datapath, prefix,
    skip_fields=['hand_pos', 'cursor_pos', 'eye_pos', 'muscle_vel', 'muscle_len', 'joint_vel', 'joint_ang', 'force'])
dataset.resample(bin_size_ms)

# ---- Extract data ---- #
if phase == 'val':
    train_split = 'train'
    eval_split = 'val'
else:
    train_split = ['train', 'val']
    eval_split = 'test'
train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False, include_forward_pred=True)
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
eval_spikes_heldin = eval_dict['eval_spikes_heldin']

train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
train_spikes_heldin_fp = train_dict['train_spikes_heldin_forward']
train_spikes_heldout_fp = train_dict['train_spikes_heldout_forward']

train_spikes = np.concatenate([
    np.concatenate([train_spikes_heldin, train_spikes_heldin_fp], axis=1),
    np.concatenate([train_spikes_heldout, train_spikes_heldout_fp], axis=1),
], axis=2)

eval_spikes_heldin = eval_dict['eval_spikes_heldin']
eval_spikes = np.full((eval_spikes_heldin.shape[0], train_spikes_heldin.shape[1] + train_spikes_heldin_fp.shape[1], train_spikes.shape[2]), 0.0)
masks = np.full((eval_spikes_heldin.shape[0], train_spikes_heldin.shape[1] + train_spikes_heldin_fp.shape[1], train_spikes.shape[2]), False)
eval_spikes[:, :eval_spikes_heldin.shape[1], :eval_spikes_heldin.shape[2]] = eval_spikes_heldin
masks[:, :eval_spikes_heldin.shape[1], :eval_spikes_heldin.shape[2]] = True

numheldin = train_spikes_heldin.shape[2]
tlen = train_spikes_heldin.shape[1]

# ---- Prepare run ---- #
T = train_spikes.shape[1]
K = states
D = factors
N = train_spikes.shape[2]
transitions = "standard"
emissions = "poisson"

train_datas = [train_spikes[i, :, :].astype(int) for i in range(len(train_spikes))]
eval_datas = [eval_spikes[i, :, :].astype(int) for i in range(len(eval_spikes))]
train_masks = [np.full(masks[0, :, :].shape, True) for _ in range(len(train_datas))]
eval_masks = [masks[i, :, :] for i in range(len(masks))]

numtrain = len(train_datas)
numeval = len(eval_datas)

# ---- Run SLDS ---- #
slds = SLDS(N, K, D,
    transitions=transitions,
    emissions=emissions,
    emission_kwargs=dict(link="softplus"),
    dynamics_kwargs=dynamics_kwargs,
)

train_datas, train_inputs, train_masks, train_tags = slds.prep_inputs(datas=train_datas)
eval_datas, eval_inputs, eval_masks, eval_tags = slds.prep_inputs(datas=eval_datas, masks=eval_masks)
gc.collect()

q_elbos_lem_train, q_lem_train, *_ = slds.fit(
    datas=train_datas,
    inputs=train_inputs,
    masks=train_masks,
    tags=train_tags,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    initialize=True,
    num_init_iters=50, num_iters=num_iters, alpha=alpha
)

q_elbos_lem_eval, q_lem_eval, *_ = slds.approximate_posterior(
    datas=eval_datas, 
    inputs=eval_inputs,
    masks=eval_masks,
    tags=eval_tags,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    num_iters=num_iters, alpha=alpha,
)

train_rates = slds.smooth_3d(q_lem_train.mean_continuous_states, train_datas, train_inputs, train_masks, train_tags).cpu().numpy()
eval_rates = slds.smooth_3d(q_lem_eval.mean_continuous_states, eval_datas, eval_inputs, eval_masks, eval_tags).cpu().numpy()

# ---- Format output ---- #
train_rates_heldin = train_rates[:, :tlen, :num_heldin]
train_rates_heldout = train_rates[:, :tlen, num_heldin:]
eval_rates_heldin = eval_rates[:, :tlen, :numheldin]
eval_rates_heldout = eval_rates[:, :tlen, numheldin:]
eval_rates_heldin_forward = eval_rates[:, tlen:, :numheldin]
eval_rates_heldout_forward = eval_rates[:, tlen:, numheldin:]

# ---- Save output ---- #
output_dict = {
    dataset_name + binsuf: {
        'train_rates_heldin': train_rates_heldin,
        'train_rates_heldout': train_rates_heldout,
        'eval_rates_heldin': eval_rates_heldin,
        'eval_rates_heldout': eval_rates_heldout,
        'eval_rates_heldin_forward': eval_rates_heldin_forward,
        'eval_rates_heldout_forward': eval_rates_heldout_forward,
    }
}
save_to_h5(output_dict, f'slds_output_{dataset_name}{binsuf}.h5')

# ---- Evaluate ---- #
if phase == 'val':
    target_dict = make_eval_target_tensors(dataset, dataset_name, train_split, eval_split, save_file=False, include_psth=True)
    print(evaluate(target_dict, output_dict))
