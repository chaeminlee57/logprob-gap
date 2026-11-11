import numpy as np
import pytest

from gap import metrics as m


def test_k3_zero_when_identical():
    lp = np.log([0.5, 0.1, 0.9])
    assert m.k3_kl(lp, lp) == 0.0


def test_k3_hand_computed():
    # one token, sampler says 0.5, trainer says 0.25 -> r = 0.5
    # k3 = (0.5 - 1) - log(0.5) = -0.5 + 0.693147...
    got = m.k3_kl(np.log([0.5]), np.log([0.25]))
    assert got == pytest.approx(-0.5 + np.log(2), rel=1e-12)


def test_k3_nonnegative_and_matches_kl_on_full_support():
    # If we draw tokens exactly in proportion to the sampler's distribution,
    # the mean of k3 is the true KL(sampler || trainer).
    mu = np.array([0.7, 0.2, 0.1])
    pi = np.array([0.6, 0.3, 0.1])
    counts = (mu * 1000).astype(int)
    lp_s = np.repeat(np.log(mu), counts)
    lp_t = np.repeat(np.log(pi), counts)
    true_kl = float(np.sum(mu * np.log(mu / pi)))
    assert m.k3_kl(lp_s, lp_t) == pytest.approx(true_kl, rel=1e-9)
    assert m.k3_kl(lp_t, lp_s) >= 0


def test_k3_tiny_differences_not_lost():
    # at |lr| ~ 1e-8, (exp(lr)-1)-lr in float64 would round to 0 or noise;
    # the true value is about lr^2 / 2 = 5e-17
    lr = 1e-8
    got = m.k3_kl(np.array([0.0]), np.array([lr]))
    assert got == pytest.approx(lr**2 / 2, rel=1e-3)


def test_frac_outside_bounds_are_inclusive():
    r = np.array([0.5, 0.79, 0.8, 1.0, 1.25, 1.26, 2.0, 2.1])
    assert m.frac_outside(r, 0.8, 1.25) == pytest.approx(5 / 8)
    assert m.frac_outside(r, 0.5, 2.0) == pytest.approx(1 / 8)


def test_tis_caps_only_from_above():
    r = np.array([0.1, 1.0, 2.0, 3.0, 50.0])
    w, frac = m.tis(r, cap=2.0)
    assert w.tolist() == [0.1, 1.0, 2.0, 2.0, 2.0]
    # exactly at the cap is not "clipped"
    assert frac == pytest.approx(2 / 5)


def test_prob_bin_edges():
    p = np.array([0.0, 0.0099, 0.01, 0.0999, 0.1, 0.4999, 0.5, 1.0])
    assert m.prob_bin(p).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_by_prob_bin_shares():
    p_ref = np.array([0.005, 0.05, 0.3, 0.9])
    lp_s = np.log(p_ref)
    # only the rare token disagrees, by a factor of 3 (outside [0.5, 2])
    lp_t = lp_s.copy()
    lp_t[0] += np.log(3)
    rows = m.by_prob_bin(p_ref, lp_s, lp_t)
    assert [r["bin"] for r in rows] == list(m.PROB_LABELS)
    assert [r["token_share"] for r in rows] == [0.25] * 4
    assert rows[0]["share_of_abs_dlogp"] == pytest.approx(1.0)
    assert rows[0]["share_of_w_outside_0.5_2"] == pytest.approx(1.0)
    assert rows[0]["mean_signed_log_w"] == pytest.approx(np.log(3))
    assert rows[0]["mean_abs_dlogp_over_mean_1mp"] == pytest.approx(np.log(3) / 0.995)
    assert rows[3]["mean_abs_dlogp"] == 0.0
    assert rows[3]["mean_signed_log_w"] == 0.0


def test_by_prob_bin_empty_bin_is_nan():
    p_ref = np.array([0.9, 0.95])
    rows = m.by_prob_bin(p_ref, np.log(p_ref), np.log(p_ref) + 0.01)
    assert np.isnan(rows[0]["mean_abs_dlogp"])
    assert rows[0]["token_share"] == 0.0


def test_seq_log_ratio_groups_and_sums():
    seq = np.array([3, 3, 1, 1, 1])
    lp_s = np.zeros(5)
    lp_t = np.array([0.1, 0.2, -0.1, -0.1, 0.05])
    ids, sums, lengths = m.seq_log_ratio(seq, lp_s, lp_t)
    assert ids.tolist() == [1, 3]
    assert sums == pytest.approx([-0.15, 0.3])
    assert lengths.tolist() == [3, 2]


def test_spread_by_length_skips_tiny_buckets():
    sums = np.array([0.1, -0.1, 0.5, -0.5, 9.0])
    lengths = np.array([10, 20, 300, 400, 900])
    rows = m.spread_by_length(sums, lengths, edges=(0, 128, 512, 1025))
    assert [(r["len_lo"], r["seqs"]) for r in rows] == [(0, 2), (128, 2)]
    assert rows[1]["std"] == pytest.approx(np.std([0.5, -0.5], ddof=1))
    assert rows[1]["std_over_sqrt_len"] == pytest.approx(np.std([0.5 / np.sqrt(300), -0.5 / np.sqrt(400)], ddof=1))
    assert rows[1]["frac_seq_w_outside_0.5_2"] == 0.0


def test_by_position_buckets():
    pos = np.arange(300)
    lp_s = np.zeros(300)
    lp_t = np.where(pos >= 256, 0.5, 0.0)
    rows = m.by_position(pos, lp_s, lp_t, width=128)
    assert [r["start"] for r in rows] == [0, 128, 256]
    assert [r["tokens"] for r in rows] == [128, 128, 44]
    assert rows[2]["mean_abs_dlogp"] == pytest.approx(0.5)


def test_bootstrap_ratio_collapses_when_sequences_identical():
    lo, hi = m.bootstrap_ratio(np.full(20, 3.0), np.full(20, 10.0))
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


def test_bootstrap_ratio_brackets_point_estimate():
    rng = np.random.default_rng(1)
    den = rng.integers(50, 500, size=100).astype(float)
    num = den * rng.uniform(0.01, 0.03, size=100)
    lo, hi = m.bootstrap_ratio(num, den)
    point = num.sum() / den.sum()
    assert lo < point < hi


def test_token_summary_keys_and_values():
    lp_s = np.log([0.5, 0.5, 0.5, 0.5])
    lp_t = np.log([0.5, 0.5, 0.5, 1.5 * 0.5])
    s = m.token_summary(lp_s, lp_t)
    assert s["tokens"] == 4
    assert s["max_abs_dlogp"] == pytest.approx(np.log(1.5))
    assert s["max_abs_dprob"] == pytest.approx(0.25)
    assert s["frac_w_outside_0.8_1.25"] == pytest.approx(0.25)
    assert s["frac_w_outside_0.5_2"] == 0.0
    assert s["frac_clipped_tis_c2"] == 0.0
