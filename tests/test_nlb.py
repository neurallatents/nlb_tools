import pytest
import numpy as np
import scipy.signal as signal
from sklearn.linear_model import PoissonRegressor

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors,
)
from nlb_tools.evaluation import evaluate


def fit_ridge_regression(train_input, test_input, train_target, alpha=0.0):
    # closed-form for better reproducibility
    train_input = np.concatenate(
        [train_input, np.ones((train_input.shape[0], 1))], axis=-1
    )
    test_input = np.concatenate(
        [test_input, np.ones((test_input.shape[0], 1))], axis=-1
    )
    w = np.linalg.pinv(
        train_input.T @ train_input + alpha * np.eye(train_input.shape[1])
    ) @ (train_input.T @ train_target)
    train_pred = train_input @ w
    test_pred = test_input @ w
    return train_pred, test_pred


def test_nlb_5ms(nlb_dataset):
    # Prepare data
    nlb_dataset.resample(5)
    # Make input tensors
    train_dict = make_train_input_tensors(
        nlb_dataset, "mc_maze_large", "train", save_file=False
    )
    eval_dict = make_eval_input_tensors(
        nlb_dataset, "mc_maze_large", "val", save_file=False
    )
    # Extract spikes
    train_spikes_heldin = train_dict["train_spikes_heldin"]
    train_spikes_heldout = train_dict["train_spikes_heldout"]
    eval_spikes_heldin = eval_dict["eval_spikes_heldin"]
    # Make target tensors
    target_dict = make_eval_target_tensors(
        nlb_dataset, "mc_maze_large", "train", "val", include_psth=True, save_file=False
    )
    # 50ms std kernel
    window = signal.windows.gaussian(int(6 * 50 / 5), int(50 / 5), sym=True)
    window /= np.sum(window)

    def filt(x):
        return np.convolve(x, window, "same")

    # Prep useful things
    flatten2d = lambda x: x.reshape(-1, x.shape[-1])
    log_offset = 1e-4
    tlen = train_spikes_heldin.shape[1]
    num_heldin = train_spikes_heldin.shape[2]
    num_heldout = train_spikes_heldout.shape[2]
    # Smooth spikes
    train_spksmth_heldin = np.apply_along_axis(filt, 1, train_spikes_heldin)
    eval_spksmth_heldin = np.apply_along_axis(filt, 1, eval_spikes_heldin)
    train_spksmth_heldout = np.apply_along_axis(filt, 1, train_spikes_heldout)
    # Prep for regression
    train_spksmth_heldin_s = flatten2d(train_spksmth_heldin)
    train_spksmth_heldout_s = flatten2d(train_spksmth_heldout)
    eval_spksmth_heldin_s = flatten2d(eval_spksmth_heldin)
    # Make lograte input
    train_lograte_heldin_s = np.log(train_spksmth_heldin_s + log_offset)
    eval_lograte_heldin_s = np.log(eval_spksmth_heldin_s + log_offset)
    # Regress
    def rectify(arr):
        return np.clip(arr, 1e-9, 1e20)

    train_spksmth_heldout_s, eval_spksmth_heldout_s = fit_ridge_regression(
        train_lograte_heldin_s,
        eval_lograte_heldin_s,
        train_spksmth_heldout_s,
        alpha=1e6,
    )
    train_spksmth_heldout = rectify(
        train_spksmth_heldout_s.reshape((-1, tlen, num_heldout))
    )
    eval_spksmth_heldout = rectify(
        eval_spksmth_heldout_s.reshape((-1, tlen, num_heldout))
    )

    output_dict = {
        "mc_maze_large": {
            "train_rates_heldin": train_spksmth_heldin,
            "train_rates_heldout": train_spksmth_heldout,
            "eval_rates_heldin": eval_spksmth_heldin,
            "eval_rates_heldout": eval_spksmth_heldout,
        }
    }

    res = evaluate(target_dict, output_dict)[0]["mc_maze_scaling_split"]

    assert np.abs(res["[500] co-bps"] - 0.1528) < 1e-4, (
        "Co-smoothing bits/spike does not match expected value. "
        + f"Expected: 0.1528. Result: {res['[500] co-bps']}"
    )
    assert np.abs(res["[500] vel R2"] - 0.5224) < 1e-4, (
        "Velocity decoding R^2 does not match expected value. "
        + f"Expected: 0.5224. Result: {res['[500] vel R2']}"
    )
    assert np.abs(res["[500] psth R2"] - 0.3871) < 1e-4, (
        "PSTH R^2 does not match expected value. "
        + f"Expected: 0.3871. Result: {res['[500] psth R2']}"
    )
