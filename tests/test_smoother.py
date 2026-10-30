import numpy as np

from perception.smoother import rts_smooth, savgol_rate

AMP, W = 0.6, 2 * np.pi * 0.7  # 0.6 rad swing at 0.7 Hz
SIGMA = 0.01


def _take(seed=0, n=600):
    rng = np.random.default_rng(seed)
    # phones record at variable frame rate: 60 fps with jitter
    t = np.cumsum(rng.uniform(0.8, 1.2, n) / 60.0)
    theta = AMP * np.sin(W * t)
    omega = AMP * W * np.cos(W * t)
    return t, theta + SIGMA * rng.standard_normal(n), omega


def test_rts_recovers_rate_on_jittered_timestamps():
    t, z, omega = _take()
    res = rts_smooth(t, z, sigma=SIGMA, q_tilde=50.0)
    err = res.states[50:-50, 1] - omega[50:-50]
    assert np.sqrt(np.mean(err**2)) < 0.05 * AMP * W


def test_smoother_beats_raw_position_noise():
    t, z, _ = _take(1)
    res = rts_smooth(t, z, sigma=SIGMA, q_tilde=50.0)
    truth = AMP * np.sin(W * t)
    assert np.std(res.states[:, 0] - truth) < 0.7 * SIGMA


def test_missing_measurements_are_bridged():
    t, z, omega = _take(2)
    z[200:230] = np.nan  # e.g. theta skipped when elongation < 0.15
    res = rts_smooth(t, z, sigma=SIGMA, q_tilde=50.0)
    assert np.all(np.isfinite(res.states))
    # uncertainty grows inside the gap
    assert res.covs[215, 0, 0] > 5 * res.covs[100, 0, 0]


def test_innovation_nll_prefers_reasonable_process_noise():
    t, z, _ = _take(3)
    nll = {q: rts_smooth(t, z, sigma=SIGMA, q_tilde=q).nll for q in (1e-4, 50.0, 1e8)}
    assert nll[50.0] < nll[1e-4] and nll[50.0] < nll[1e8]


def test_savgol_cross_check_is_correct_on_clean_data():
    # SG is diagnostic only: window 7 amplifies noise, so check correctness without noise
    t, _, omega = _take(4)
    sg = savgol_rate(t, AMP * np.sin(W * t))
    assert np.max(np.abs(sg - omega)[10:-10]) < 0.02 * AMP * W


def test_decimate_removes_codec_period_and_keeps_dynamics():
    from perception.smoother import decimate_uniform
    rng = np.random.default_rng(5)
    t = np.cumsum(rng.uniform(0.99, 1.01, 600) / 60.0)  # phone VFR jitter is ~1%
    clean = AMP * np.sin(W * t)
    codec = 0.002 * np.tile([1.0, -0.5, -0.5], t.size // 3 + 1)[: t.size]  # period-3 frame-type artifact
    td, zd = decimate_uniform(t, clean + codec, 4)
    truth = AMP * np.sin(W * td)
    assert np.max(np.abs(zd - truth)[5:-5]) < 3e-4  # artifact gone, swing intact
