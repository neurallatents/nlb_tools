import pytest
import numpy as np
from scipy.special import gammaln

from nlb_tools.evaluation import (
    evaluate,
    neg_log_likelihood,
    bits_per_spike,
    fit_and_eval_decoder,
    eval_psth,
    speed_tp_correlation,
    velocity_decoding,
)


# -- NLL and bits/spike ----------


def test_neg_log_likelihood():
    """Test that NLL computation is correct"""
    # randomized test
    for _ in range(20):
        spikes = np.random.randint(low=0, high=5, size=(10, 100, 10)).astype(float)
        rates = np.random.exponential(scale=1.0, size=(10, 100, 10))

        expected_nll = np.sum(rates - spikes * np.log(rates) + gammaln(spikes + 1.0))
        actual_nll = neg_log_likelihood(rates, spikes)
        assert np.isclose(expected_nll, actual_nll)


def test_neg_log_likelihood_mismatched_shapes():
    """Test that NLL computation fails when shapes don't match"""
    # randomized test
    spikes = np.random.randint(low=0, high=5, size=(10, 100, 8)).astype(float)
    rates = np.random.exponential(scale=1.0, size=(10, 100, 10))

    with pytest.raises(AssertionError):
        neg_log_likelihood(rates, spikes)


def test_neg_log_likelihood_negative_rates():
    """Test that NLL computation fials when rates are negative"""
    # randomized test
    spikes = np.random.randint(low=0, high=5, size=(10, 100, 8)).astype(float)
    rates = np.random.exponential(scale=1.0, size=(10, 100, 10))
    rates -= np.min(rates) + 5  # guarantee negative rates

    with pytest.raises(AssertionError):
        neg_log_likelihood(rates, spikes)


def test_neg_log_likelihood_drop_nans():
    """Test that NLL computation is correct when there are nans in either rates or spikes"""
    # randomized test
    for _ in range(20):
        spikes = np.random.randint(low=0, high=5, size=(10, 100, 10)).astype(float)
        rates = np.random.exponential(scale=1.0, size=(10, 100, 10))
        mask = np.random.rand(10, 100, 10) > 0.9
        spikes[mask] = np.nan
        if np.random.rand() > 0.5:  # rates does not have to have nans
            rates[mask] = np.nan

        expected_nll = np.sum(
            rates[~mask]
            - spikes[~mask] * np.log(rates[~mask])
            + gammaln(spikes[~mask] + 1.0)
        )
        actual_nll = neg_log_likelihood(rates, spikes)
        assert np.isclose(expected_nll, actual_nll)


def test_neg_log_likelihood_mismatched_nans():
    """Test that NLL computation is correct"""
    # randomized test
    spikes = np.random.randint(low=0, high=5, size=(10, 100, 10)).astype(float)
    rates = np.random.exponential(scale=1.0, size=(10, 100, 10))
    mask = np.random.rand(10, 100, 10)
    # make sure spikes and rates have different nans
    spikes[mask < 0.1] = np.nan
    rates[mask > 0.9] = np.nan

    with pytest.raises(AssertionError):
        neg_log_likelihood(rates, spikes)


def test_bits_per_spike():
    for _ in range(20):
        spikes = np.random.randint(low=0, high=5, size=(10, 100, 10)).astype(float)
        rates = np.random.exponential(scale=1.0, size=(10, 100, 10))
        null_rates = np.tile(
            spikes.mean(axis=(0, 1), keepdims=True),
            (spikes.shape[0], spikes.shape[1], 1),
        ).squeeze()

        expected_rate_nll = np.sum(
            rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
        )
        expected_null_nll = np.sum(
            null_rates - spikes * np.log(null_rates) + gammaln(spikes + 1.0)
        )
        expected_bps = (
            (expected_null_nll - expected_rate_nll) / np.sum(spikes) / np.log(2)
        )
        actual_bps = bits_per_spike(rates, spikes)
        assert np.isclose(expected_bps, actual_bps)


def test_bits_per_spike_drop_nans():
    for _ in range(20):
        spikes = np.random.randint(low=0, high=5, size=(10, 100, 10)).astype(float)
        rates = np.random.exponential(scale=1.0, size=(10, 100, 10))
        mask = np.random.rand(10, 100, 10) > 0.9
        spikes[mask] = np.nan
        if np.random.rand() > 0.5:  # rates does not have to have nans
            rates[mask] = np.nan
        null_rates = np.tile(
            np.nanmean(spikes, axis=(0, 1), keepdims=True),
            (spikes.shape[0], spikes.shape[1], 1),
        ).squeeze()

        expected_rate_nll = np.sum(
            rates[~mask]
            - spikes[~mask] * np.log(rates[~mask])
            + gammaln(spikes[~mask] + 1.0)
        )
        expected_null_nll = np.sum(
            null_rates[~mask]
            - spikes[~mask] * np.log(null_rates[~mask])
            + gammaln(spikes[~mask] + 1.0)
        )
        expected_bps = (
            (expected_null_nll - expected_rate_nll) / np.nansum(spikes) / np.log(2)
        )
        actual_bps = bits_per_spike(rates, spikes)
        assert np.isclose(expected_bps, actual_bps)


# -- Ridge regression ---------------


def test_fit_and_eval_decoder():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(size=(1000, 10))
    y = x @ rng.standard_normal(size=(10, 2))

    # noiseless should have high R^2
    score = fit_and_eval_decoder(
        train_rates=x[:800],
        train_behavior=y[:800],
        eval_rates=x[800:],
        eval_behavior=y[800:],
    )
    assert score > 0.95

    # with noise should still have decent R^2
    y += rng.standard_normal(size=(1000, 2)) * 0.1
    score = fit_and_eval_decoder(
        train_rates=x[:800],
        train_behavior=y[:800],
        eval_rates=x[800:],
        eval_behavior=y[800:],
    )
    assert score > 0.25  # arbitrary heuristic

    # regressing on noise should have poor R^2
    y = rng.standard_normal(size=(1000, 2))
    score = fit_and_eval_decoder(
        train_rates=x[:800],
        train_behavior=y[:800],
        eval_rates=x[800:],
        eval_behavior=y[800:],
    )
    assert score < 0.95  # arbitrary heuristic


# -- PSTH evaluation

# def test_eval_psth():
#     return
