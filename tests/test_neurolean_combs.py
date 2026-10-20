import numpy as np

from nlb_tools import (
    betti_refractory_orbit_purification,
    koopman_5d_spike_jet_observables,
    schur_feshbach_lfp_spike_dress,
)
from nlb_tools.evaluation import bits_per_spike


def test_schur_feshbach_lfp_spike_dress():
    rng = np.random.default_rng(2026)
    fs = 19531.25
    n_samples = 4096
    t = np.arange(n_samples, dtype=np.float64) / fs

    # Slow LFP (20 Hz) + High-Gamma PING (120 Hz) + Fast Spike Shells (600 Hz, 1800 Hz)
    lfp = 2.0 * np.sin(2.0 * np.pi * 20.0 * t) + 1.2 * np.cos(2.0 * np.pi * 120.0 * t)
    spikes_fast = (1.0 + 0.5 * np.cos(2.0 * np.pi * 120.0 * t)) * (
        np.sin(2.0 * np.pi * 600.0 * t) + 0.8 * np.sin(2.0 * np.pi * 1800.0 * t)
    )
    signal = lfp + spikes_fast + rng.normal(0.0, 0.2, size=n_samples)

    res = schur_feshbach_lfp_spike_dress(signal, fs=fs)

    assert res["H_eff"].shape == (2, 2)
    assert res["Sigma_E"].shape == (2, 2)
    assert res["contraction_verified"] is True
    assert res["feshbach_banach_L_FB"] < 1.0
    assert res["cond_eff"] < 100.0
    assert res["conditioning_reduction_x"] > 1000.0
    assert np.isclose(res["n1_reference_conditioning_reduction_x"], 711349.6)

    # Exact block-inversion identity: H_PP u_P + H_PQ u_Q == H_eff u_P when u_Q = (E I - H_QQ)^(-1) H_QP u_P
    u_p = res["dressed_slow_lfp"]
    u_q = np.linalg.solve(res["probe_energy_E"] * np.eye(4) - res["H_QQ"], res["H_QP"] @ u_p)
    lhs = res["H_PP"] @ u_p + res["H_PQ"] @ u_q
    rhs = res["H_eff"] @ u_p
    np.testing.assert_allclose(lhs, rhs, atol=1e-13)


def test_koopman_5d_spike_jet_observables():
    rng = np.random.default_rng(77)
    n_samples = 2000
    t = np.linspace(0.0, 40.0, n_samples)
    # Nonlinear excitable spike-like trajectory with strong quadratic / phase-velocity coupling
    v = np.sin(t) - 0.65 * (np.sin(t) ** 3) + 0.45 * np.cos(2.0 * t) + rng.normal(0.0, 0.03, size=n_samples)

    res = koopman_5d_spike_jet_observables(v, step_k=2, active_threshold_sigma=0.5)

    assert res["observables_5d"].shape == (5, n_samples)
    assert res["koopman_matrix"].shape == (5, 5)
    assert res["r2_koopman_5d"] > res["r2_linear_1d"]
    assert res["r2_lift_ratio_x"] > 1.25
    assert np.isclose(res["n1_reference_r2_lift_ratio_x"], 1.9614)

    # Also test 3D trial tensor lifting (N, T, C) -> (N, T, 5*C)
    tensor_3d = rng.normal(0.0, 1.0, size=(4, 60, 6))
    res_3d = koopman_5d_spike_jet_observables(tensor_3d)
    assert res_3d["observables_5d"].shape == (4, 60, 30)


def test_betti_refractory_orbit_purification_and_per_neuron_bps():
    # Two distinct single-unit limit-cycle orbits interleaved on one electrode with sub-1.5ms cross-ISIs
    orbit_a = np.array([0.0, 2.2, 4.5, 7.1, 9.8, 12.4], dtype=np.float64)
    orbit_b = np.array([0.8, 3.1, 5.2, 8.0, 10.6, 13.1], dtype=np.float64)  # 0.8 ms after orbit_a (< 1.5 ms)
    combined_times = np.concatenate([orbit_a, orbit_b])

    jet_a = np.tile(np.array([-6.5, 0.35, 1.2]), (len(orbit_a), 1))
    jet_b = np.tile(np.array([-4.2, 0.75, 0.5]), (len(orbit_b), 1))
    combined_jets = np.vstack([jet_a, jet_b])

    res = betti_refractory_orbit_purification(combined_times, combined_jets, tau_ref_ms=1.500)

    assert res["raw_refractory_violations"] > 0
    assert res["raw_grassmann_compliance_pct"] < 100.0
    assert res["betti_1_orbits_allocated"] == 2
    assert res["purified_single_unit_violations"] == 0
    assert np.isclose(res["purified_grassmann_compliance_pct"], 100.0)
    assert res["nilpotent_grassmann_compliant"] is True

    # Verify per_neuron bits_per_spike option alongside standard population bits_per_spike
    rng = np.random.default_rng(12)
    spikes = rng.poisson(0.4, size=(5, 40, 4)).astype(np.float64)
    rates = np.clip(spikes * 0.6 + 0.15, 1e-4, None)
    bps_pop = bits_per_spike(rates, spikes, per_neuron=False)
    bps_pn = bits_per_spike(rates, spikes, per_neuron=True)
    assert np.isfinite(bps_pop) and bps_pop > 0.0
    assert np.isfinite(bps_pn) and bps_pn > 0.0
