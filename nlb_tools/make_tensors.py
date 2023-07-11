
from .nwb_interface import NWBDataset
from .chop import ChopInterface, chop_data, merge_chops
from itertools import product
import numpy as np
import pandas as pd
import h5py
import sys
import os
import logging

logger = logging.getLogger(__name__)

PARAMS = {
    'mc_maze': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 100,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 70,
        },
    },
    'mc_rtt': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'finger_vel',
        'lag': 140,
        'make_params': {
            'align_field': 'start_time',
            'align_range': (0, 600),
            'allow_overlap': True,
        },
        'eval_make_params': {
            'align_field': 'start_time',
            'align_range': (0, 600),
            'allow_overlap': True,
        },
        'fp_len': 200,
    },
    'area2_bump': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'decode_masks': lambda x: np.stack([x.ctr_hold_bump == 0, x.ctr_hold_bump == 1]).T,
        'lag': -20,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-100, 500),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-100, 500),
            'allow_overlap': True,
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['cond_dir', 'ctr_hold_bump'],
            'make_params': {
                    'align_field': 'move_onset_time',
                    'align_range': (-100, 500),
            },
            'kern_sd': 40,
        },
    },
    'dmfc_rsg': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'behavior_source': 'trial_info',
        'behavior_mask': lambda x: x.is_outlier == 0,
        'behavior_field': ['is_eye', 'theta', 'is_short', 'ts', 'tp'],
        'jitter': lambda x: np.stack([
            np.zeros(len(x)),
            np.where(x.split == 'test', np.zeros(len(x)), 
                     np.clip(1500.0 - x.get('tp', pd.Series(np.nan)).to_numpy(), 0.0, 300.0))
        ]).T,
        'make_params': {
            'align_field': 'go_time',
            'align_range': (-1500, 0),
            'allow_overlap': True,
        },
        'eval_make_params': {
            'start_field': 'set_time',
            'end_field': 'go_time',
            'align_field': 'go_time',
        },
        'eval_tensor_params': {
            'seg_len': 1500,
            'pad': 'front'
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['is_eye', 'theta', 'is_short', 'ts'],
            'make_params': {
                'start_field': 'set_time',
                'end_field': 'go_time',
                'align_field': 'go_time',
            },
            'kern_sd': 70,
            'pad': 'front',
            'seg_len': 1500,
            'skip_mask': lambda x: x.is_outlier == 1,
        },
    },
    'mc_maze_large': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 120,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 50,
        },
    },
    'mc_maze_medium': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 120,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 50,
        },
    },
    'mc_maze_small': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 120,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 50,
        },
    },
}


def make_train_input_tensors(dataset, dataset_name,
                             trial_split='train',
                             update_params=None,
                             save_file=True,
                             return_dict=True,
                             save_path="train_input.h5",
                             include_behavior=False,
                             include_forward_pred=False,
                             seed=0):
    """Makes model training input tensors.
    Creates 3d arrays containing heldin and heldout spikes
    for train trials (and other data if indicated)
    and saves them as .h5 files and/or returns them 
    in a dict

    Parameters
    ----------
    dataset : NWBDataset
        An instance of NWBDataset to make tensors from
    dataset_name : {'mc_maze', 'mc_rtt', 'area2_bum', 'dmfc_rsg',
                    'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
        Name of dataset. Used to select default
        parameters from PARAMS
    trial_split : {'train', 'val'}, array-like, or list, optional
        The selection of trials to make the tensors with.
        It can be the predefined trial splits 'train'
        or 'val', an array-like boolean mask (see the 
        include_trials argument of `NWBDataset.make_trial_data`),
        or a list containing the previous two types, which 
        will include trials that are in any of the splits
        in the list. By default 'train'
    update_params : dict, optional
        New parameters with which to update default
        dict from PARAMS
    save_file : bool, optional
        Whether to save the reshaped data to an
        h5 file, by default True
    return_dict : bool, optional
        Whether to return the reshaped data in a
        data dict with the same keys as the h5 files,
        by default True
    save_path : str, optional
        Path to where the h5 output file should be saved
    include_behavior : bool, optional
        Whether to include behavioral data in the
        returned tensors, by default False
    include_forward_pred : bool, optional
        Whether to include forward-prediction spiking
        data in the returned tensors, by default False
    seed : int, optional
        Seed for random generator used for jitter
    
    Returns
    -------
    dict of np.array
        A dict containing 3d numpy arrays of
        spiking data for indicated trials, and possibly
        additional data based on provided arguments
    """
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"
    assert dataset_name in PARAMS.keys(), f"`dataset_name` must be one of {list(PARAMS.keys())}"
    assert isinstance(trial_split, (pd.Series, np.ndarray, list)) or trial_split in ['train', 'val'], \
        "Invalid `trial_split` argument. Please refer to the documentation for valid choices"

    # Fetch and update params
    params = PARAMS[dataset_name].copy()
    if update_params is not None:
        params.update(update_params)
    # Add filename extension if necessary
    if not save_path.endswith('.h5'):
        save_path = save_path + '.h5'

    # unpack params
    spk_field = params['spk_field']
    hospk_field = params['hospk_field']
    make_params = params['make_params'].copy()
    jitter = params.get('jitter', None)
    
    # Prep mask
    trial_mask = _prep_mask(dataset, trial_split)

    # Prep jitter if necessary
    if jitter is not None:
        np.random.seed(seed)
        jitter_vals = _prep_jitter(dataset, trial_mask, jitter)
        align_field = make_params.get('align_field', make_params.get('start_field', 'start_time'))
        align_vals = dataset.trial_info[trial_mask][align_field]
        align_jit = align_vals + pd.to_timedelta(jitter_vals, unit='ms')
        align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info = pd.concat([dataset.trial_info, align_jit], axis=1)
        if 'align_field' in make_params:
            make_params['align_field'] = align_jit.name
        else:
            make_params['start_field'] = align_jit.name

    # Make output spiking arrays and put into data_dict
    train_dict = make_stacked_array(dataset, [spk_field, hospk_field], make_params, trial_mask)
    data_dict = {
        'train_spikes_heldin': train_dict[spk_field],
        'train_spikes_heldout': train_dict[hospk_field],
    }

    # Add behavior data if necessary
    if include_behavior:
        behavior_source = params['behavior_source']
        behavior_field = params['behavior_field']
        behavior_make_params = _prep_behavior(dataset, params.get('lag', None), make_params)
        # Retrieve behavior data from indicated source
        if behavior_source == 'data':
            train_behavior = make_jagged_array(dataset, [behavior_field], behavior_make_params, trial_mask)[0][behavior_field]
        else:
            train_behavior = (
                dataset.trial_info[trial_mask][behavior_field]
                .apply(lambda x: x.dt.total_seconds() if hasattr(x, "dt") else x)
                .to_numpy()
                .astype('float')
            )
        # Filter out behavior on certain trials if necessary
        if 'behavior_mask' in params:
            if callable(params['behavior_mask']):
                behavior_mask = params['behavior_mask'](dataset.trial_info[trial_mask])
            else:
                behavior_mask, _ = params['behavior_mask']
            train_behavior[~behavior_mask] = np.nan
        data_dict['train_behavior'] = train_behavior
    
    # Add forward prediction data if necessary
    if include_forward_pred:
        fp_len = params['fp_len']
        fp_steps = fp_len / dataset.bin_width
        fp_make_params = _prep_fp(make_params, fp_steps, dataset.bin_width)
        fp_dict = make_stacked_array(dataset, [spk_field, hospk_field], fp_make_params, trial_mask)
        data_dict['train_spikes_heldin_forward'] = fp_dict[spk_field]
        data_dict['train_spikes_heldout_forward'] = fp_dict[hospk_field]

    # Delete jitter column
    if jitter is not None:
        dataset.trial_info.drop(align_jit.name, axis=1, inplace=True)

    # Save and return data
    if save_file:
        save_to_h5(data_dict, save_path, overwrite=True)
    if return_dict:
        return data_dict

def make_eval_input_tensors(dataset, dataset_name, 
                            trial_split='val',
                            update_params=None,
                            save_file=True,
                            return_dict=True,
                            save_path="eval_input.h5",
                            seed=0):
    """Makes model evaluation input tensors.
    Creates 3d arrays containing heldin spiking for 
    eval trials (and heldout spiking if available)
    and saves them as .h5 files and/or returns them 
    in a dict
    
    Parameters
    ----------
    dataset : NWBDataset
        An instance of NWBDataset to make tensors from
    dataset_name : {'mc_maze', 'mc_rtt', 'area2_bum', 'dmfc_rsg',
                    'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
        Name of dataset. Used to select default
        parameters from PARAMS
    trial_split : {'train', 'val', 'test'}, array-like, or list, optional
        The selection of trials to make the tensors with.
        It can be the predefined trial splits 'train'
        'val', or 'test', an array-like boolean mask (see the 
        include_trials argument of `NWBDataset.make_trial_data`),
        or a list containing the previous two types, which 
        will include trials that are in any of the splits
        in the list. By default 'val'
    update_params : dict, optional
        New parameters with which to update default
        dict from PARAMS
    save_file : bool, optional
        Whether to save the reshaped data to an
        h5 file, by default True
    return_dict : bool, optional
        Whether to return the reshaped data in a
        data dict with the same keys as the h5 files,
        by default True
    save_path : str, optional
        Path to where the h5 output file should be saved
    seed : int, optional
        Seed for random generator used for jitter
    
    Returns
    -------
    dict of np.array
        A dict containing 3d numpy arrays of
        spiking data for indicated trials
    """
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"
    assert dataset_name in PARAMS.keys(), f"`dataset_name` must be one of {list(PARAMS.keys())}"
    assert isinstance(trial_split, (pd.Series, np.ndarray, list)) or trial_split in ['train', 'val', 'test'], \
        "Invalid `trial_split` argument. Please refer to the documentation for valid choices"

    # Fetch and update params
    params = PARAMS[dataset_name].copy()
    if update_params is not None:
        params.update(update_params)
    # Add filename extension if necessary
    if not save_path.endswith('.h5'):
        save_path = save_path + '.h5'

    # Unpack params
    spk_field = params['spk_field']
    hospk_field = params['hospk_field']
    make_params = params['make_params'].copy()
    make_params['allow_nans'] = True
    jitter = params.get('jitter', None)
    
    # Prep mask
    trial_mask = _prep_mask(dataset, trial_split)

    # Prep jitter if necessary
    if jitter is not None:
        np.random.seed(seed)
        jitter_vals = _prep_jitter(dataset, trial_mask, jitter)
        align_field = make_params.get('align_field', make_params.get('start_field', 'start_time'))
        align_vals = dataset.trial_info[trial_mask][align_field]
        align_jit = align_vals + pd.to_timedelta(jitter_vals, unit='ms')
        align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info = pd.concat([dataset.trial_info, align_jit], axis=1)
        if 'align_field' in make_params:
            make_params['align_field'] = align_jit.name
        else:
            make_params['start_field'] = align_jit.name

    # Make output spiking arrays and put into data_dict
    if not np.any(dataset.trial_info[trial_mask].split == 'test'):
        eval_dict = make_stacked_array(dataset, [spk_field, hospk_field], make_params, trial_mask)
        data_dict = {
            'eval_spikes_heldin': eval_dict[spk_field],
            'eval_spikes_heldout': eval_dict[hospk_field],
        }
    else:
        eval_dict = make_stacked_array(dataset, [spk_field], make_params, trial_mask)
        data_dict = {
            'eval_spikes_heldin': eval_dict[spk_field],
        }

    # Delete jitter column
    if jitter is not None:
        dataset.trial_info.drop(align_jit.name, axis=1, inplace=True)

    # Save and return data
    if save_file:
        save_to_h5(data_dict, save_path, overwrite=True)
    if return_dict:
        return data_dict

def make_eval_target_tensors(dataset, dataset_name, 
                     train_trial_split='train',
                     eval_trial_split='val',
                     update_params=None,
                     save_file=True,
                     return_dict=True,
                     save_path="target_data.h5",
                     include_psth=False,
                     seed=0):
    """Makes tensors containing target data used to evaluate model predictions.
    Creates 3d arrays containing true heldout spiking data
    for eval trials and other arrays for model evaluation and saves them
    as .h5 files and/or returns them in a dict. Because heldout
    data is not available in the 'test' split, this function can not
    be used on the 'test' split, though it is what we used to generate
    the EvalAI evaluation data
    
    Parameters
    ----------
    dataset : NWBDataset
        An instance of NWBDataset to make tensors from
    dataset_name : {'mc_maze', 'mc_rtt', 'area2_bum', 'dmfc_rsg',
                    'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
        Name of dataset. Used to select default
        parameters from PARAMS
    train_trial_split : {'train', 'val'}, array-like, or list, optional
        The selection of trials used for training.
        It can be the predefined trial splits 'train'
        or 'val', an array-like boolean mask (see the 
        include_trials argument of `NWBDataset.make_trial_data`),
        or a list containing the previous two types, which 
        will include trials that are in any of the splits
        in the list. By default 'train'
    eval_trial_split : {'train', 'val'}, array-like, or list, optional
        The selection of trials used for evaluation.
        It follows the same format as train_trial_split
        described above. By default 'val'
    update_params : dict, optional
        New parameters with which to update default
        dict from PARAMS
    save_file : bool, optional
        Whether to save the reshaped data to an
        h5 file, by default True
    return_dict : bool, optional
        Whether to return the reshaped data in a
        data dict with the same keys as the h5 files,
        by default True
    save_path : str, optional
        Path to where the h5 output file should be saved
    include_psth : bool, optional
        Whether to make PSTHs for evaluation of match
        to PSTH, by default False. Since PSTH calculation
        is memory and cpu-intensive in its current implementation,
        it may be desirable to skip this step
    seed : int, optional
        Seed for random generator used for jitter
    
    Returns
    -------
    nested dict of np.array
        Dict containing data for evaluation, including
        held-out spiking activity for eval trials 
        and behavioral correlates
    """
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"
    assert dataset_name in PARAMS.keys(), f"`dataset_name` must be one of {list(PARAMS.keys())}"
    assert isinstance(train_trial_split, (pd.Series, np.ndarray, list)) or train_trial_split in ['train', 'val', 'test'], \
        "Invalid `train_trial_split` argument. Please refer to the documentation for valid choices"
    assert isinstance(eval_trial_split, (pd.Series, np.ndarray, list)) or eval_trial_split in ['train', 'val', 'test'], \
        "Invalid `eval_trial_split` argument. Please refer to the documentation for valid choices"
    
    # Fetch and update params
    params = PARAMS[dataset_name].copy()
    if update_params is not None:
        params.update(update_params)
    # Add filename extension if necessary
    if not save_path.endswith('.h5'):
        save_path = save_path + '.h5'

    # unpack params
    spk_field = params['spk_field']
    hospk_field = params['hospk_field']
    make_params = params['eval_make_params'].copy()
    behavior_source = params['behavior_source']
    behavior_field = params['behavior_field']
    jitter = params.get('jitter', None)
    eval_tensor_params = params.get('eval_tensor_params', {}).copy()
    fp_len = params['fp_len']
    fp_steps = fp_len / dataset.bin_width

    # Properly name output fields based on submission bin width
    suf = '' if (dataset.bin_width == 5) else f'_{dataset.bin_width}'
    
    # Prep masks
    train_mask = _prep_mask(dataset, train_trial_split)
    eval_mask = _prep_mask(dataset, eval_trial_split)
    if isinstance(eval_trial_split, str) and eval_trial_split == 'test':
        ignore_mask = dataset.trial_info.split == 'none'
    else:
        ignore_mask = ~(train_mask | eval_mask)
    
    # Prep jitter if necessary
    if jitter is not None:
        align_field = make_params.get('align_field', make_params.get('start_field', 'start_time'))

        np.random.seed(seed)
        train_jitter_vals = _prep_jitter(dataset, train_mask, jitter)
        train_align_vals = dataset.trial_info[train_mask][align_field]
        train_align_jit = train_align_vals + pd.to_timedelta(train_jitter_vals, unit='ms')
        train_align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info = pd.concat([dataset.trial_info, train_align_jit], axis=1)

        np.random.seed(seed)
        eval_jitter_vals = _prep_jitter(dataset, eval_mask, jitter)
        eval_align_vals = dataset.trial_info[eval_mask][align_field]
        eval_align_jit = eval_align_vals + pd.to_timedelta(eval_jitter_vals, unit='ms')
        eval_align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info.loc[eval_mask, eval_align_jit.name] = eval_align_jit
        if 'align_field' in make_params:
            make_params['align_field'] = eval_align_jit.name
        else:
            make_params['start_field'] = eval_align_jit.name
    else:
        train_jitter_vals = None
        eval_jitter_vals = None
    
    behavior_make_params = _prep_behavior(dataset, params.get('lag', None), make_params)
    if not ('align_range' in make_params):
        # Stack jagged arrays by padding with NaNs if uneven trials
        train_dict = make_jagged_array(dataset, [hospk_field], make_params, train_mask, jitter=train_jitter_vals, **eval_tensor_params)[0]
        eval_dict = make_jagged_array(dataset, [hospk_field], make_params, eval_mask, jitter=eval_jitter_vals, **eval_tensor_params)[0]
    else:
        # Make standard 3d arrays
        eval_dict = make_stacked_array(dataset, [hospk_field], make_params, eval_mask)
        if behavior_source == 'data':
            # Use `make_jagged_arrays` for RTT, in case some data is cut short at edges
            btrain_dict = make_jagged_array(dataset, [behavior_field], behavior_make_params, train_mask)[0]
            beval_dict = make_jagged_array(dataset, [behavior_field], behavior_make_params, eval_mask)[0]
    
    # Retrieve behavioral data
    if behavior_source == 'trial_info':
        train_behavior = (
            dataset.trial_info[train_mask][behavior_field]
            .apply(lambda x: x.dt.total_seconds() if hasattr(x, "dt") else x)
            .to_numpy()
            .astype('float')
        )
        eval_behavior = (
            dataset.trial_info[eval_mask][behavior_field]
            .apply(lambda x: x.dt.total_seconds() if hasattr(x, "dt") else x)
            .to_numpy()
            .astype('float')
        )
    else:
        train_behavior = btrain_dict[behavior_field]
        eval_behavior = beval_dict[behavior_field]
    # Mask some behavioral data if desired
    if 'behavior_mask' in params:
        if callable(params['behavior_mask']):
            train_behavior_mask = params['behavior_mask'](dataset.trial_info[train_mask])
            eval_behavior_mask = params['behavior_mask'](dataset.trial_info[eval_mask])
        else:
            train_behavior_mask, eval_behavior_mask = params['behavior_mask']
        train_behavior[~train_behavior_mask] = np.nan
        eval_behavior[~eval_behavior_mask] = np.nan
    
    # Prepare forward prediction spiking data
    fp_make_params = _prep_fp(make_params, fp_steps, dataset.bin_width)
    fp_dict = make_stacked_array(dataset, [spk_field, hospk_field], fp_make_params, eval_mask)
    
    # Construct data dict
    data_dict = {
        dataset_name + suf: {
            'eval_spikes_heldout': eval_dict[hospk_field],
            'train_behavior': train_behavior,
            'eval_behavior': eval_behavior,
            'eval_spikes_heldin_forward': fp_dict[spk_field],
            'eval_spikes_heldout_forward': fp_dict[hospk_field],
        }
    }

    # Include `decode_masks` to train separate decoders for different data
    if 'decode_masks' in params:
        if callable(params['decode_masks']):
            train_decode_mask = params['decode_masks'](dataset.trial_info[train_mask])
            eval_decode_mask = params['decode_masks'](dataset.trial_info[eval_mask])
        else:
            train_decode_mask, eval_decode_mask = params['decode_masks']
        data_dict[dataset_name + suf]['train_decode_mask'] = train_decode_mask
        data_dict[dataset_name + suf]['eval_decode_mask'] = eval_decode_mask
        
    # Calculate PSTHs if desired
    if include_psth:
        psth_params = params.get('psth_params', None)
        if psth_params is None:
            logger.warning("PSTHs are not supported for this dataset, skipping...")
        else:
            (train_cond_idx, eval_cond_idx), psths, comb = _make_psth(dataset, train_mask, eval_mask, ignore_mask, **psth_params)
            
            data_dict[dataset_name + suf]['eval_cond_idx'] = eval_cond_idx
            data_dict[dataset_name + suf]['train_cond_idx'] = train_cond_idx
            data_dict[dataset_name + suf]['psth'] = psths
            if jitter is not None:
                data_dict[dataset_name + suf]['eval_jitter'] = (eval_jitter_vals / dataset.bin_width).round().astype(int)
                data_dict[dataset_name + suf]['train_jitter'] = (train_jitter_vals / dataset.bin_width).round().astype(int)

    # Delete jitter column
    if jitter is not None:
        dataset.trial_info.drop(eval_align_jit.name, axis=1, inplace=True)

    # Save and return data
    if save_file:
        save_to_h5(data_dict, save_path, overwrite=True)
    if return_dict:
        return data_dict


''' Array creation helper functions '''
def make_stacked_array(dataset, fields, make_params, include_trials):
    """Generates 3d trial x time x channel arrays for each given field
    
    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized data. Provided arguments
        must result in fixed-length trials
    include_trials : array-like
        Boolean array to select trials to extract
    
    Returns
    -------
    dict of np.array
        Dict mapping each field in `fields`
        to a 3d trial x time x channel numpy array
    """
    if 'ignored_trials' in make_params:
        logger.warning("`ignored_trials` found in `make_params`. Deleting and overriding with `include_trials`")
        make_params.pop('ignored_trials')
    if type(fields) != list:
        fields = [fields]
    trial_data = dataset.make_trial_data(ignored_trials=~include_trials, **make_params)
    grouped = list(trial_data.groupby('trial_id', sort=False))
    array_dict = {}
    for field in fields:
        array_dict[field] = np.stack([trial[field].to_numpy() for _, trial in grouped])
    return array_dict

def make_jagged_array(dataset, fields, make_params, include_trials, jitter=None, pad='back', seg_len=None):
    """Generates 3d trial x time x channel arrays for each given field for uneven trial lengths

    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized data
    include_trials : array-like
        Boolean array to select trials to extract
    pad : {'back', 'front'}, optional
        Whether to pad shorter trials with NaNs
        at the end ('back') or beginning ('front')
        of the array, by default 'back'
    seg_len : int, optional
        Trial length to limit arrays to, by default
        None does not enforce a limit
    
    Returns
    -------
    dict of np.array
        Dict mapping each field in `fields`
        to a 3d trial x time x channel numpy array
    """
    if 'ignored_trials' in make_params:
        logger.warning("`ignored_trials` found in `make_params`. Overriding with `include_trials`")
        make_params.pop('ignored_trials')
    if type(fields) != list:
        fields = [fields]
    if type(include_trials) != list:
        include_trials = [include_trials]
    if jitter is None:
        jitter = [np.zeros(it.sum()) for it in include_trials]
    elif type(jitter) != list:
        jitter = [jitter]
    trial_data = dataset.make_trial_data(ignored_trials=~np.any(include_trials, axis=0), **make_params)
    grouped = dict(list(trial_data.groupby('trial_id', sort=False)))
    max_len = np.max([trial.shape[0] for _, trial in grouped.items()]) if seg_len is None else int(round(seg_len / dataset.bin_width))

    dict_list = []
    for trial_sel, jitter_vals in zip(include_trials, jitter):
        trial_ixs = dataset.trial_info[trial_sel].index.to_numpy()
        array_dict = {}
        for field in fields:
            arr = np.full((len(trial_ixs), max_len, dataset.data[field].shape[1]), np.nan)
            for i in range(len(trial_ixs)):
                jit = int(round(jitter_vals[i] / dataset.bin_width))
                data = grouped[trial_ixs[i]][field].to_numpy()
                if pad == 'front':
                    if jit == 0:
                        data = data[-max_len:]
                        arr[i, -data.shape[0]:, :] = data
                    elif jit > 0:
                        data = data[-(max_len - jit):]
                        arr[i, -(data.shape[0] + jit):-jit, :] = data
                    else:
                        data = data[-(max_len - jit):jit]
                        arr[i, -data.shape[0]:, :] = data
                elif pad == 'back':
                    if jit == 0:
                        data = data[:max_len]
                        arr[i, :data.shape[0], :] = data
                    if jit > 0:
                        data = data[jit:(max_len + jit)]
                        arr[i, :data.shape[0], :] = data
                    else:
                        data = data[:(max_len - jit)]
                        arr[i, -jit:(data.shape[0] - jit)] = data
            array_dict[field] = arr
        dict_list.append(array_dict)
    return dict_list

def make_seg_chopped_array(dataset, fields, make_params, chop_params, include_trials):
    """Generates chopped 3d arrays from trial segments using ChopInterface for given fields.
    Note that this function has not been used extensively and may not be perfect

    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized data
    chop_params : dict
        Arguments for `ChopInterface` initialization
    include_trials : array-like
        Boolean array to select trials to extract
    
    Returns
    -------
    tuple
        Tuple containing dict mapping each field in `fields`
        to a 3d segment x time x channel numpy array and
        the ChopInterface that records chop information
        required for reconstruction
    """
    if 'ignored_trials' in make_params:
        logger.warning("`ignored_trials` found in `make_params`. Overriding with `include_trials`")
        make_params.pop('ignored_trials')
    if type(fields) != list:
        fields = [fields]
    trial_data = dataset.make_trial_data(ignored_trials=~include_trials, **make_params)
    ci = ChopInterface(**chop_params)
    array_dict = ci.chop(trial_data, fields)
    return array_dict, ci

def make_cont_chopped_array(dataset, fields, chop_params, lag=0):
    """Generates 3d chopped arrays from continuous data using ChopInterface for given fields.
    Note that this function has not been used extensively and may not be perfect

    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    chop_params : dict
        Arguments for `ChopInterface` initialization
    lag : int, optional
        Amount of initial offset for continuous data
        before chopping, by default 0. Must be
        non-negative
    
    Returns
    -------
    tuple
        Tuple containing dict mapping each field in `fields`
        to a 3d segment x time x channel numpy array and
        the ChopInterface that records chop information
        required for reconstruction
    """
    if type(fields) != list:
        fields = [fields]

    ci = ChopInterface(**chop_params)
    if lag > 0:
        data = pd.concat([
            dataset.data.iloc[lag:], 
            pd.DataFrame(np.full((lag, dataset.data.shape[1]), np.nan), 
                index=(dataset.data.index[-lag:] + pd.to_timedelta(lag, unit='ms')))
            ], axis=0)
    else:
        data = dataset.data
    array_dict = ci.chop(data, fields)
    return array_dict, ci


''' Chop merging helper functions '''
def merge_seg_chops_to_df(dataset, data_dicts, cis):
    """Merges segment chopped 3d arrays back to main continuous dataframe.
    Note that this function has not been used extensively and may not be perfect

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset from which chopped data was generated
        and to which merged data will be added
    data_dicts : dict of np.array or list of dicts
        Dict (or list of dicts) mapping field names to 
        3d segment x time x channel chopped arrays
    cis : ChopInterface or list of ChopInterfaces
        ChopInterface(s) corresponding to chopped
        data in `data_dicts`
    """
    if type(data_dicts) != list:
        data_dicts = [data_dicts]
    if type(cis) != list:
        cis = [cis]
    assert len(data_dicts) == len(cis), "`data_dicts` and `cis` must be the same length"
    fields = list(data_dicts[0].keys())
    logger.info(f"Merging {fields} into dataframe")
    merged_list = []
    for data_dict, ci in zip(data_dicts, cis):
        merged_df = ci.merge(data_dict)
        merged_list.append(merged_df)
    merged = pd.concat(merged_list, axis=0).reset_index()
    if merged.clock_time.duplicated().sum() != 0:
        logger.warning("Duplicate time indices found. Merging by averaging")
        merged = merged.groupby('clock_time', sort=False).mean().reset_index()
    dataset.data = pd.concat([dataset.data, merged.set_index('clock_time')], axis=1)

def merge_cont_chops_to_df(dataset, data_dicts, ci, masks):
    """Merges continuous chopped 3d arrays back to main continuous dataframe.
    Note that this function has not been used extensively and may not be perfect

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset from which chopped data was generated
        and to which merged data will be added
    data_dicts : dict of np.array or list of dicts
        Dict (or list of dicts) mapping field names to 
        3d segment x time x channel chopped arrays
    ci : ChopInterface or list of ChopInterfaces
        ChopInterface used to chop continuous data
    masks : array-like or list of array-like
        Boolean mask (or list of masks) indicating which
        segments of original chops were not dropped,
        as dropping may be necessary if the original
        chopped segment contained NaNs
    """
    if type(data_dicts) != list:
        data_dicts = [data_dicts]
    if type(masks) != list:
        masks = [masks]
    assert isinstance(ci, ChopInterface), "`ci` should be a single ChopInterface for merging continuous chops"
    assert len(data_dicts) == len(masks), "`data_dicts` and `masks` must be the same length"
    fields = list(data_dicts[0].keys())
    logger.info(f"Merging {fields} into dataframe")
    num_chops = len(masks[0])
    chop_len = data_dicts[0][fields[0]].shape[1]
    full_dict = {}
    for field in fields:
        num_chan = data_dicts[0][field].shape[2]
        full_arr = np.full((num_chops, chop_len, num_chan), np.nan)
        for data_dict, mask in zip(data_dicts, masks):
            full_arr[mask] = data_dict[field]
        full_dict[field] = full_arr
    merged = ci.merge(full_dict).reset_index()
    if merged.clock_time.duplicated().sum() != 0:
        logger.warning("Duplicate time indices found. Merging by averaging")
        merged = merged.groupby('clock_time', sort=False).mean().reset_index()
    dataset.data = pd.concat([dataset.data, merged.set_index('clock_time')], axis=1)

''' Miscellaneous helper functions '''
def combine_train_eval(dataset, train_dict, eval_dict, train_split, eval_split):
    """Function that combines dict of tensors from two splits
    into one tensor while preserving original order of trials

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset from which data was extracted from
    train_dict : dict
        Dict containing tensors from `train_split`. Format 
        expected to match outputs of `make_train_input_tensors`
        and `make_eval_input_tensors`, though exact field names may
        differ
    eval_dict : dict
        Dict containing tensors from `eval_split`, like `train_dict`
    train_split : {'train', 'val', 'test'}, array-like, or list, optional
        Trial splits contained in `train_dict`. See `make_train_input_tensors`
        for more information
    eval_split : {'train', 'val', 'test'}, array-like, or list, optional
        Trial splits contained in `eval_dict`, like `eval_split`
    
    Returns
    -------
    dict of np.array
        Dict of np.array with same keys as `train_dict` but
        with arrays containing data from both input dicts
    """
    train_mask = _prep_mask(dataset, train_split)
    eval_mask = _prep_mask(dataset, eval_split)
    assert not np.any(np.all([train_mask, eval_mask], axis=0)), \
        "Duplicate trial(s) found in both `train_split` and `eval_split`. Unable to merge..."
    tolist = lambda x: x if isinstance(x, list) else [x]
    train_eval_split = tolist(train_split) + tolist(eval_split)
    train_eval_mask = _prep_mask(dataset, train_eval_split)
    num_tot = train_eval_mask.sum()
    train_idx = np.arange(num_tot)[train_mask[train_eval_mask]]
    eval_idx = np.arange(num_tot)[eval_mask[train_eval_mask]]
    return _combine_dict(train_dict, eval_dict, train_idx, eval_idx)

def _combine_dict(train_dict, eval_dict, train_idx, eval_idx):
    """Recursive helper function that combines dict of tensors from two splits
    into one tensor using provided indices

    Parameters
    ----------
    train_dict : dict
        Dict containing tensors from `train_split`. Format 
        expected to match outputs of `make_train_input_tensors`
        and `make_eval_input_tensors`, though exact field names may
        differ.
    eval_dict : dict
        Dict containing tensors from `eval_split`.
    train_idx : array-like
        Indices of where trials in `train_dict` should be 
        in combined array
    eval_idx : array-like
        Indices of where trials in `eval_dict` should be 
        in combined array

    -------
    dict of np.array
        Dict of np.array with same keys as `train_dict` but
        with arrays containing data from both input dicts
    """
    combine_dict = {}
    for key, val in train_dict.items():
        if isinstance(val, dict):
            if key not in eval_dict:
                logger.warning(f'{key} not found in `eval_dict`, skipping...')
                continue
            combine_dict[key] = _combine_dict(train_dict[key], eval_dict[key], train_idx, eval_idx)
        else:
            if key.replace('train', 'eval') not in eval_dict:
                logger.warning(f"{key.replace('train', 'eval')} not found in `eval_dict`, skipping...")
                continue
            train_arr = val
            eval_arr = eval_dict[key.replace('train', 'eval')]
            assert train_arr.shape[1] == eval_arr.shape[1], f"Trial lengths for {key} and {key.replace('train', 'eval')} don't match"
            assert train_arr.shape[2] == eval_arr.shape[2], f"Number of channels for {key} and {key.replace('train', 'eval')} don't match"
            full_arr = np.empty((train_arr.shape[0] + eval_arr.shape[0], train_arr.shape[1], train_arr.shape[2]))
            full_arr[train_idx] = train_arr
            full_arr[eval_idx] = eval_arr
            combine_dict[key] = full_arr
    return combine_dict

def _prep_mask(dataset, trial_split):
    """Converts string trial split names to boolean array and combines
    multiple splits if a list is provided

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset that trial mask is made for
    trial_split : {'train', 'val', 'test'}, array-like, or list, optional
        Trial splits to include in mask. See `make_train_input_tensors` and
        related functions for more information
    
    Returns
    -------
    np.array
        Boolean array indicating which trials are within 
        provided split(s)
    """
    split_to_mask = lambda x: (dataset.trial_info.split == x) if isinstance(x, str) else x
    if isinstance(trial_split, list):
        trial_mask = np.any([split_to_mask(split) for split in trial_split], axis=0)
    else:
        trial_mask = split_to_mask(trial_split)
    return trial_mask

def _prep_behavior(dataset, lag, make_params):
    """Helper function that makes separate make_params 
    for behavioral data

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset that the make_params are made for
    lag : int
        Amount that behavioral data is lagged relative to
        trial alignment in `make_params`
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized data
    
    Returns
    -------
    dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized behavior data
    """
    behavior_make_params = make_params.copy()
    if lag is not None:
        behavior_make_params['allow_nans'] = True
        if 'align_range' in behavior_make_params:
            behavior_make_params['align_range'] = tuple([(t + lag) for t in make_params['align_range']])
        else:
            behavior_make_params['align_range'] = (lag, lag)
    else:
        behavior_make_params = None
    return behavior_make_params

def _prep_fp(make_params, fp_steps, bin_width_ms):
    """Helper function that makes separate make_params
    for forward prediction data

    Parameters
    ----------
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized data
    fp_steps : int
        Amount of time for which forward prediction
        spiking activity is extracted, in ms
    bin_width_ms : int
        Dataset bin width in ms
    
    Returns
    -------
    dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized forward prediction data
    """
    align_point = make_params.get('align_field', make_params.get('end_field', 'end_time'))
    align_start = make_params.get('align_range', (0,0))[1]
    align_window = (align_start, align_start + fp_steps * bin_width_ms)
    fp_make_params = {
        'align_field': align_point,
        'align_range': align_window,
        'allow_overlap': True,
    }
    return fp_make_params

def _prep_jitter(dataset, trial_mask, jitter):
    """Helper function that that randomly choose jitter
    values for each trial

    Parameters
    ----------
    dataset : NWBDataset
        Dataset to prepare jitter for
    trial_mask : array-like of bool
        Mask indicating which trials to prepare jitter for
    jitter : np.ndarray
        2d array with 2 columns. The first column
        should be the lower bound on possible jitter values
        per trial, and the second should be the upper 
        bound (exclusive)
    
    Returns
    -------
    np.ndarray
        Array containing jitter values for each trial in ms
    """
    trial_info = dataset.trial_info[trial_mask]
    if callable(jitter):
        jitter_range = jitter(trial_info)
    elif isinstance(jitter, (list, tuple)) and len(jitter) == 2:
        jitter_range = np.tile(np.array(jitter), (len(trial_info), 1))
    elif isinstance(jitter, np.ndarray):
        assert jitter.shape == (trial_info, 2), \
            f"Error: `jitter` array shape is incorrect; " \
            "provided shape: {jitter.shape}, expected shape: ({trial_info}, 2)"
        jitter_range = jitter
    else:
        logger.error("Unrecognized type for argument `jitter`")
    
    jitter_range = np.floor((jitter_range / dataset.bin_width).round(4))

    sample = lambda x: np.random.random() * (x[1] - x[0]) + x[0]
    jitter_vals = np.apply_along_axis(sample, 1, jitter_range).round()
    return jitter_vals * dataset.bin_width

def _make_psth(dataset, train_mask, eval_mask, ignore_mask, make_params, cond_fields, kern_sd, pad='back', psth_len=None, seg_len=None, skip_mask=None):
    """Function to generate PSTHs to evaluate eval trials
    This function is extremely slow and memory-intensive, as it completely
    loads the dataset again so that it can smooth spikes before resampling

    Parameters
    ----------
    dataset : NWBDataset
        Dataset object for which PSTHs are made
    train_mask : array-like
        Boolean mask indicating trials used for training
    eval_mask : array-like
        Boolean mask indicating trials to be evaluated
    ignore_mask : array-like
        Boolean mask indicating trials to be excluded 
        when computing PSTHs
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
        to extract trialized data for PSTH calculation
    cond_fields : list of str
        List of `dataset.trial_info` field names used to 
        define conditions
    kern_sd : int
        Standard deviation of Gaussian smoothing kernel
        in ms
    pad : {'back', 'front'}, optional
        Whether to pad shorter trials with NaNs
        at the end ('back') or beginning ('front')
        of the array, by default 'back'. Ignored
        if PSTHs are all the same length
    seg_len : int, optional
        Maximum length to limit PSTHs to, in ms.
        By default None, which enforces no limit
    skip_mask : array-like, optional
        Additional indicator like `ignore_mask` for
        trials to be excluded from PSTH calculation.
        Can be callable function to be called on 
        `dataset.trial_info`
    
    Returns
    -------
    tuple
        Tuple containing PSTHs and various related information
    """
    # Reload dataset and smooth spikes
    bin_width = dataset.bin_width
    ti = dataset.trial_info
    neur = dataset.data[['spikes', 'heldout_spikes']].columns
    dataset = NWBDataset(dataset.fpath, dataset.prefix, skip_fields=['force', 'hand_pos', 'hand_vel', 'finger_pos', 'finger_vel', 'eye_pos', 'cursor_pos', 'muscle_len', 'muscle_vel', 'joint_ang', 'joint_vel'])
    dataset.trial_info = ti
    dataset.data = dataset.data.loc[:, neur]
    dataset.smooth_spk(kern_sd, signal_type=['spikes', 'heldout_spikes'], overwrite=True, ignore_nans=True)
    if bin_width != 1:
        dataset.resample(bin_width)

    # Make mask for valid trials to skip for PSTH calculation
    if skip_mask is not None:
        if callable(skip_mask):
            skip_mask = skip_mask(dataset.trial_info)
    else:
        skip_mask = np.full(len(dataset.trial_info), False)

    # Find unique conditions based on trial info fields
    if type(cond_fields) == str:
        cond_fields = [cond_fields]
    combinations = sorted(dataset.trial_info[~ignore_mask][cond_fields].dropna().set_index(cond_fields).index.unique().tolist())
    
    # Get trial ids of train and test trials
    train_trial_ids = dataset.trial_info[train_mask][['trial_id', make_params.get('align_field', 'start_time')]].dropna().trial_id.to_numpy()
    eval_trial_ids = dataset.trial_info[eval_mask][['trial_id', make_params.get('align_field', 'start_time')]].dropna().trial_id.to_numpy()
    
    # Koop through conditions
    psth_list = []; train_ids_list = []; eval_ids_list = []; remove_combs = []
    for comb in combinations:
        # Find trials in condition
        mask = np.all(dataset.trial_info[cond_fields] == comb, axis=1)
        if not np.any(mask & (~ignore_mask) & (~skip_mask)):
            logger.warning(f"No matching trials found for {comb}. Dropping")
            remove_combs.append(comb)
            continue
        # Make trial data
        trial_data = dataset.make_trial_data(ignored_trials=(~mask | ignore_mask | skip_mask), allow_nans=True, **make_params)
        # Find length of shortest trial
        mean_len = np.mean([trial.shape[0] for tid, trial in trial_data.groupby('trial_id')])
        tlens = {tid: trial.shape[0] for tid, trial in trial_data.groupby('trial_id')}
        bad_tid = [tid for tid in tlens.keys() if tlens[tid] < (0.8 * mean_len)]
        trial_data = trial_data[~np.isin(trial_data.trial_id, bad_tid)]
        min_len = np.min([trial.shape[0] for tid, trial in trial_data.groupby('trial_id')])
        # Compute PSTH, shorten to min length
        psth = trial_data.groupby('align_time')[trial_data[['spikes', 'heldout_spikes']].columns].mean().to_numpy()
        psth = psth[:min_len] if pad == 'back' else psth[-min_len:]
        # Find indices of train and test trials in condition, for evaluation
        train_ids = np.sort(np.where(np.isin(train_trial_ids, trial_data.trial_id))[0])
        eval_ids = np.sort(np.where(np.isin(eval_trial_ids, trial_data.trial_id))[0])
        # Add to list
        psth_list.append(psth)
        train_ids_list.append(train_ids)
        eval_ids_list.append(eval_ids)
    
    # Stack PSTHs to 3d array
    max_len = np.max([psth.shape[0] for psth in psth_list]) if seg_len is None else int(round(seg_len / dataset.bin_width))
    psth = np.stack([psth if psth.shape[0] == max_len
                     else psth[:max_len] if (psth.shape[0] > max_len and pad == 'back')
                     else psth[max_len:] if (psth.shape[0] > max_len and pad == 'front')
                     else np.concatenate([psth, np.full((max_len - psth.shape[0], psth.shape[1]), np.nan)], axis=0) if pad == 'back'
                     else np.concatenate([np.full((max_len - psth.shape[0], psth.shape[1]), np.nan), psth], axis=0)
                     for psth in psth_list])
    good_comb = [comb for comb in combinations if comb not in remove_combs] 
    train_ids_list = np.array(train_ids_list, dtype='object')
    eval_ids_list = np.array(eval_ids_list, dtype='object')

    return (train_ids_list, eval_ids_list), psth, good_comb


''' Tensor saving functions '''
def save_to_h5(data_dict, save_path, overwrite=False, dlen=32, compression="gzip"):
    """Function that saves dict as .h5 file while preserving
    nested dict structure

    Parameters
    ----------
    data_dict : dict
        Dict containing data to be saved in HDF5 format
    save_path : str
        Path to location where data should be saved
    overwrite : bool, optional
        Whether to overwrite duplicate data found 
        at `save_path` if file already exists, by
        default False
    dlen : int, optional
        Byte length of data format to save numerical data,
        by default 32.
    compression : str, optional
        Compression to use when writing to HDF5, default "gzip"
    """
    h5file = h5py.File(save_path, 'a')
    good, dup_list = _check_h5_r(data_dict, h5file, overwrite)
    if good:
        if len(dup_list) > 0:
            logger.warning(f"{dup_list} already found in {save_path}. Overwriting...")
        _save_h5_r(data_dict, h5file, dlen, compression)
        logger.info(f"Saved data to {save_path}")
    else:
        logger.warning(f"{dup_list} already found in {save_path}. Save to file canceled. " \
            "Please set `overwrite=True` or specify a different file path.")
    h5file.close()

def _check_h5_r(data_dict, h5obj, overwrite):
    """Recursive helper function that finds duplicate keys and deletes them if `overwrite == True`
    
    Parameters
    ----------
    data_dict : dict
        Dict containing data to be saved in HDF5 format
    h5obj : h5py.File or h5py.Group
        h5py object to check for duplicates
    overwrite : bool, optional
        Whether to overwrite duplicate data found 
        at `save_path` if file already exists, by
        default False
    
    Returns
    -------
    tuple
        Tuple containing bool of whether `h5obj` passes
        checks and list of duplicate keys found
    """
    dup_list = []
    good = True
    for key in data_dict.keys():
        if key in h5obj.keys():
            if isinstance(h5obj[key], h5py.Group) and isinstance(data_dict[key], dict):
                rgood, rdup_list = _check_h5_r(data_dict[key], h5obj[key], overwrite)
                good = good and rgood
                dup_list += list(zip([key] * len(rdup_list), rdup_list))
            else:
                dup_list.append(key)
                if overwrite:
                    del h5obj[key]
                else:
                    good = False
    return good, dup_list

def _save_h5_r(data_dict, h5obj, dlen, compression="gzip"):
    """Recursive function that adds all the items in a dict to an h5py.File or h5py.Group object

    Parameters
    ----------
    data_dict : dict
        Dict containing data to be saved in HDF5 format
    h5obj : h5py.File or h5py.Group
        h5py object to save data to
    dlen : int, optional
        Byte length of data format to save numerical data,
        by default 32.
    compression : str, optional
        Compression to use when writing to HDF5, default "gzip"
    """
    for key, val in data_dict.items():
        if isinstance(val, dict):
            h5group = h5obj[key] if key in h5obj.keys() else h5obj.create_group(key)
            _save_h5_r(val, h5group, dlen)
        else:
            if val.dtype == 'object':
                sub_dtype = val[0].dtype
                if dlen is not None:
                    if np.issubdtype(sub_dtype, np.floating):
                        sub_dtype = f'float{dlen}'
                    elif np.issubdtype(sub_dtype, np.integer):
                        sub_dtype = f'int{dlen}'                    
                dtype = h5py.vlen_dtype(sub_dtype)
            else:
                dtype = val.dtype
                if dlen is not None:
                    if np.issubdtype(dtype, np.floating):
                        dtype = f'float{dlen}'
                    elif np.issubdtype(dtype, np.integer):
                        dtype = f'int{dlen}'
            h5obj.create_dataset(key, data=val, dtype=dtype, compression=compression)
            
def h5_to_dict(h5obj):
    """Recursive function that reads HDF5 file to dict

    Parameters
    ----------
    h5obj : h5py.File or h5py.Group
        File or group object to load into a dict
    
    Returns
    -------
    dict of np.array
        Dict mapping h5obj keys to arrays
        or other dicts
    """
    data_dict = {}
    for key in h5obj.keys():
        if isinstance(h5obj[key], h5py.Group):
            data_dict[key] = h5_to_dict(h5obj[key])
        else:
            data_dict[key] = h5obj[key][()]
    return data_dict

def combine_h5(file_paths, save_path=None):
    """Function that takes multiple .h5 files and combines them into one.
    May be particularly useful for combining MC_Maze scaling results for submission

    Parameters
    ----------
    file_paths : list
        List of paths to h5 files to combine
    save_path : str, optional
        Path to save combined results to. By
        default None saves to first path in 
        `file_paths`.
    """
    assert len(file_paths) > 1, "Must provide at least 2 files to combine"
    if save_path is None:
        save_path = file_paths[0]
    for fpath in file_paths:
        if fpath == save_path:
            continue
        with h5py.File(fpath, 'r') as h5file:
            data_dict = h5_to_dict(h5file)
        save_to_h5(data_dict, save_path)

''' Tensor chopping convenience functions '''
def chop_tensors(fpath, window, overlap, chop_fields=None, save_path=None):
    """Function that chops a tensor .h5 file directly without needing to load into NWBDataset.
    Note that this function has not been used extensively and may not be perfect
    
    Parameters
    ----------
    fpath : str
        Path to HDF5 file containing data to chop
    window : int
        Length of chop window
    overlap : int
        Overlap shared between chop windows
    chop_fields : list of str, optional
        Fields to chop. By default None chops all fields
    save_path : str, optional
        Path to save chopped data to. By default None
        overwrites data in the original file
    """
    h5file = h5py.File(fpath, 'r')
    if chop_fields is None:
        chop_fields = sorted(list(h5file.keys()))
    if save_path is None:
        save_path = fpath[::-1].split('.', 1)[-1][::-1] + '_chopped.h5'
    elif not save_path.endswith('.h5'):
        save_path += '.h5'
    data_list = []
    for field in chop_fields:
        arr = h5file[field][()]
        if len(arr.shape) < 3:
            arr = arr[:, :, None] if len(arr.shape) == 2 else arr[:, None, None]
        data_list.append(arr)
    h5file.close()
    splits = np.cumsum([arr.shape[2] for arr in data_list[:-1]])
    data = np.dstack(data_list)
    chop_list = []
    if (data.shape[1] - window) % (window - overlap) != 0:
        logger.warning("With the specified window and overlap, " \
            f"{(data.shape[1] - window) % (window - overlap)} samples will be discarded per trial.")
    for i in range(data.shape[0]):
        seg = data[i, :, :]
        chop_list.append(chop_data(seg, overlap, window))
    chopped_data = np.vstack(chop_list)
    chopped_data = np.split(chopped_data, splits, axis=2)
    data_dict = {chop_fields[i]: chopped_data[i] for i in range(len(chop_fields))}
    save_to_h5(data_dict, save_path, overwrite=True)

def merge_tensors(fpath, window, overlap, orig_len, merge_fields=None, save_path=None):
    """Function that merges a tensor .h5 file after chopping without needing to load into NWBDataset.
    Because SegmentRecords aren't saved, the original segment length (assumed to be a fixed value) must be provided.
    Note that this function has not been used extensively and may not be perfect
    
    Parameters
    ----------
    fpath : str
        Path to HDF5 file containing data to merge
    window : int
        Length of chop window
    overlap : int
        Overlap shared between chop windows
    orig_len : int
        Original length of trial segments before chopping
    merge_fields : list of str, optional
        Fields to merge. By default None merges all fields
    save_path : str, optional
        Path to save chopped data to. By default None
        overwrites data in the original file
    """
    h5file = h5py.File(fpath, 'r')
    if merge_fields is None:
        merge_fields = sorted(list(h5file.keys()))
    if save_path is None:
        save_path = fpath[::-1].split('.', 1)[-1][::-1] + '_merged.h5'
    elif not save_path.endswith('.h5'):
        save_path += '.h5'
    data_list = []
    for field in merge_fields:
        arr = h5file[field][()]
        data_list.append(arr)
    h5file.close()
    splits = np.cumsum([arr.shape[2] for arr in data_list[:-1]])
    data = np.dstack(data_list)
    chop_per_seg = int(round((orig_len - overlap)/(window - overlap)))
    merge_list = []
    for i in range(0, data.shape[0], chop_per_seg):
        chopped_seg = data[i:(i+chop_per_seg), :, :]
        merge_list.append(merge_chops(chopped_seg, overlap, orig_len))
    merge_data = np.stack(merge_list)
    merge_data = np.split(merge_data, splits, axis=2)
    data_dict = {merge_fields[i]: merge_data[i] for i in range(len(merge_fields))}
    save_to_h5(data_dict, save_path, overwrite=True)
