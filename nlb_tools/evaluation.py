import numpy as np
import h5py
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from scipy.special import gammaln
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV

import logging
logger = logging.getLogger(__name__)

def evaluate(test_annotation_file, user_submission_file):
    """
    Runs evaluation as it would be run on EvalAI servers

    Parameters
    ----------
    test_annotation_file : str or dict
        Path to the eval target .h5 file or dict of eval target 
        data to evaluate against
    user_submission_file : str
        Path to the .h5 file or dict with user 
        rate predictions

    Returns
    -------
    list
        List containing a dict for each dataset that was
        evaluated. Each dict contains the calculated metrics
    """
    logger.info("Starting Evaluation.....")

    # define prefixes for scaling metrics
    scaling_tcount = {
        'mc_maze_large': '[500]',
        'mc_maze_medium': '[250]',
        'mc_maze_small': '[100]',
    }

    # read data from files
    if type(test_annotation_file) == str:
        target_data = h5py.File(test_annotation_file, 'r')
    else:
        target_data = test_annotation_file
    if type(user_submission_file) == str:
        user_data = h5py.File(user_submission_file, 'r')
    else:
        user_data = user_submission_file
    
    result_list = []
    scaling_dict = {}
    scaling_dict_20 = {}
    # evaluate on datasets that are included in both submission and evaluation data
    for dataset in ['mc_maze', 'mc_rtt', 'area2_bump', 'dmfc_rsg', 'mc_maze_large', 'mc_maze_medium', 'mc_maze_small']:
        for bin_size_ms, suf in zip([5, 20], ['', '_20']):
            if (dataset + suf) not in user_data.keys():
                continue
            dataset_name = dataset + suf
            logger.info(f"Evaluating {dataset_name}")
            result_dict = {}
            # check that both submission and evaluation dicts have data for this dataset
            if 'train_rates_heldin' not in user_data[dataset_name].keys():
                continue
            elif (dataset_name) not in target_data.keys() or 'eval_spikes_heldout' not in target_data[dataset_name].keys():
                logger.warning(f"Evaluation data for {dataset_name} not found")
                continue

            # extract evaluation data
            eval_spikes_heldout = target_data[dataset_name]['eval_spikes_heldout'][()].astype('float')
            train_behavior = target_data[dataset_name]['train_behavior'][()].astype('float')
            eval_behavior = target_data[dataset_name]['eval_behavior'][()].astype('float')

            # extract submitted data
            eval_rates_heldin = user_data[dataset_name]['eval_rates_heldin'][()].astype('float')
            eval_rates_heldout = user_data[dataset_name]['eval_rates_heldout'][()].astype('float')
            eval_rates = np.concatenate([eval_rates_heldin, eval_rates_heldout], axis=-1)

            # calculate co-smoothing bits per spike
            result_dict['co-bps'] = float(bits_per_spike(eval_rates_heldout, eval_spikes_heldout))

            if dataset == 'dmfc_rsg':
                # find where data is outside of set-go period
                mask = np.hstack([np.isnan(eval_spikes_heldout[:, :, 0])])
                # calculate neural speed
                eval_speeds = np.array([np.mean(np.linalg.norm(np.diff(eval_rates[i, :, :][~mask[i, :]], axis=0), axis=1), axis=0) for i in range(len(eval_rates))])

                # calculate correlation within each condition
                decoding_rs = []
                # conditions based only off prior, response modality, and direction
                # because there aren't many trials for each t_s in the test split
                cond_cols = [0, 1, 2]
                conds = np.vstack(list({tuple(cond) for cond in eval_behavior[:, cond_cols] if not np.all(np.isnan(cond))})) # find conditions (behavior columns are in `make_tensors.py`)
                for cond in conds:
                    cmask = np.all(eval_behavior[:, cond_cols] == cond, axis=1)
                    cond_eval_speeds = eval_speeds[cmask]
                    cond_eval_behavior = eval_behavior[cmask][:, -1]
                    decoding_rs.append(pearsonr(cond_eval_speeds, cond_eval_behavior)[0])
                decoding_r = np.median(decoding_rs)
                result_dict["tp corr"] = decoding_r
            else:
                # extract train rates for regression
                train_rates_heldin = user_data[dataset_name]['train_rates_heldin'][()].astype('float')
                train_rates_heldout = user_data[dataset_name]['train_rates_heldout'][()].astype('float')
                train_rates = np.concatenate([train_rates_heldin, train_rates_heldout], axis=-1)
                flatten3d = lambda x: x.reshape(-1, x.shape[2]) if (len(x.shape) > 2) else x
                # make decode mask if not provided
                if 'train_decode_mask' in target_data[dataset_name].keys():
                    train_decode_mask = target_data[dataset_name]['train_decode_mask'][()]
                    eval_decode_mask = target_data[dataset_name]['eval_decode_mask'][()]
                else:
                    train_decode_mask = np.full(train_rates.shape[0], True)[:, None]
                    eval_decode_mask = np.full(eval_rates.shape[0], True)[:, None]
                decoding_r2s = []
                # train/evaluate regression for each mask
                for i in range(train_decode_mask.shape[1]):
                    decoding_r2 = fit_and_eval_decoder(
                        flatten3d(train_rates[train_decode_mask[:, i]]),
                        flatten3d(train_behavior[train_decode_mask[:, i]]),
                        flatten3d(eval_rates[eval_decode_mask[:, i]]),
                        flatten3d(eval_behavior[eval_decode_mask[:, i]])
                    )
                    decoding_r2s.append(decoding_r2)
                # average R2s across masks
                result_dict["vel R2"] = np.mean(decoding_r2s)
            
            if 'psth' in target_data[dataset_name].keys():
                # extract PSTHs and evaluate
                psth = target_data[dataset_name]['psth'][()].astype('float')
                psth_r2 = eval_psth(psth, eval_rates)              
                result_dict["psth R2"] = float(psth_r2)

            if 'eval_rates_heldin_forward' in user_data[dataset_name].keys() and 'eval_spikes_heldin_forward' in target_data[dataset_name].keys():
                # extract forward prediction data
                eval_spikes_heldin_forward = target_data[dataset_name]['eval_spikes_heldin_forward'][()].astype('float')
                eval_spikes_heldout_forward = target_data[dataset_name]['eval_spikes_heldout_forward'][()].astype('float')
                eval_rates_heldin_forward = user_data[dataset_name]['eval_rates_heldin_forward'][()].astype('float')
                eval_rates_heldout_forward = user_data[dataset_name]['eval_rates_heldout_forward'][()].astype('float')
                # combine held-in and held-out
                eval_spikes_forward = np.dstack([eval_spikes_heldin_forward, eval_spikes_heldout_forward])
                eval_rates_forward = np.dstack([eval_rates_heldin_forward, eval_rates_heldout_forward])
                # calculate forward prediction bits per spike
                result_dict['fp-bps'] = float(bits_per_spike(eval_rates_forward, eval_spikes_forward))

            if dataset in ['mc_maze_large', 'mc_maze_medium', 'mc_maze_small']:
                sd = scaling_dict if suf == '' else scaling_dict_20
                for key, val in result_dict.items():
                    sd[scaling_tcount[dataset] + " " + key] = val
            elif dataset in ['mc_maze', 'mc_rtt', 'area2_bump', 'dmfc_rsg']:
                result_list.append({f"{dataset_name}_split": result_dict})
    
    # put scaling data in proper split
    if len(scaling_dict) > 0:
        result_list.append({'mc_maze_scaling_split': scaling_dict})
    if len(scaling_dict_20) > 0:
        result_list.append({'mc_maze_scaling_20_split': scaling_dict_20})
    
    logger.info("Completed evaluation")

    try:
        target_data.close()
    except:
        pass
    try:
        user_data.close()
    except:
        pass

    return result_list

def neg_log_likelihood(rates, spikes):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)
    
    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    
    Returns
    -------
    float
        Total negative log-likelihood of the data
    """  
    assert spikes.shape == rates.shape, \
        f"Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]
    
    assert not np.any(np.isnan(rates)), \
        "NaN rate predictions found"

    assert np.all(rates >= 0), \
        "Negative rate predictions found"
    if (np.any(rates == 0)):
        logger.warning("Zero rate predictions found. Replacing zeros with 1e-9")
        rates[rates == 0] = 1e-9
    
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts
    
    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    nll_null = neg_log_likelihood(np.tile(np.nanmean(spikes, axis=(0,1), keepdims=True), (spikes.shape[0], spikes.shape[1], 1)), spikes)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

def fit_and_eval_decoder(train_rates, train_behavior, eval_rates, eval_behavior):
    """Fits ridge regression on train data passed
    in and evaluates on eval data

    Parameters
    ----------
    train_rates : np.ndarray
        2d array with 1st dimension being samples (time) and
        2nd dimension being input variables (units).
        Used to train regressor
    train_behavior : np.ndarray
        2d array with 1st dimension being samples (time) and
        2nd dimension being output variables (channels).
        Used to train regressor
    eval_rates : np.ndarray
        2d array with same dimension ordering as train_rates.
        Used to evaluate regressor
    eval_behavior : np.ndarray
        2d array with same dimension ordering as train_behavior.
        Used to evaluate regressor
    
    Returns
    -------
    float
        R2 score on eval data
    """
    if np.any(np.isnan(train_behavior)):
        train_rates = train_rates[~np.isnan(train_behavior)[:, 0]]
        train_behavior = train_behavior[~np.isnan(train_behavior)[:, 0]]
    if np.any(np.isnan(eval_behavior)):
        eval_rates = eval_rates[~np.isnan(eval_behavior)[:, 0]]
        eval_behavior = eval_behavior[~np.isnan(eval_behavior)[:, 0]]
    assert not np.any(np.isnan(train_rates)) and not np.any(np.isnan(eval_rates)), \
        "NaNs found in rate predictions within required trial times"

    ridge = Ridge()
    param_grid = {'alpha': np.logspace(-4, 0, 9)}
    gscv = GridSearchCV(ridge, param_grid)
    gscv.fit(train_rates, train_behavior)
    return gscv.score(eval_rates, eval_behavior)

def eval_psth(psth, eval_rates):
    """Evaluates match to PSTH

    Parameters
    ----------
    psth : np.ndarray
        2d array, with dimensions time x neuron,
        containing PSTHs for each unit for each
        trial in the eval data
    eval_rates : np.ndarray
        3d array, with dimensions trial x time x neuron,
        containing rate predictions for all eval trials
    
    Returns
    -------
    float
        R2 of rate predictions to true PSTHs, averaged
        across neurons
    """
    eval_rates = eval_rates.reshape(-1, eval_rates.shape[2])
    nan_mask = np.isnan(psth[:, 0])
    assert not np.any(np.isnan(eval_rates[~nan_mask])), \
        "NaNs found in rate predictions within required trial times"
    return r2_score(psth[~nan_mask], eval_rates[~nan_mask])
