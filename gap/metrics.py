"""Mismatch metrics between a sampler's logprobs and a trainer's logprobs.

Everything here takes plain numpy arrays of per-token logprobs for the
*sampled* tokens. `lp_s` is what the sampler recorded, `lp_t` is what the
trainer computes for the same token. The importance weight is
r = pi_trainer / mu_sampler = exp(lp_t - lp_s).
"""

import numpy as np

PROB_EDGES = (0.0, 0.01, 0.1, 0.5, 1.0)
PROB_LABELS = ("<0.01", "0.01-0.1", "0.1-0.5", ">=0.5")


def k3_kl(lp_s, lp_t):
    """k3 estimator of KL(sampler || trainer) from tokens drawn by the sampler.

    k3 = (r - 1) - log r, always >= 0. expm1 keeps it accurate when r is
    within ~1e-7 of 1, which is most tokens at fp32.
    """
    lr = np.asarray(lp_t, dtype=np.float64) - np.asarray(lp_s, dtype=np.float64)
    if lr.size == 0:
        return float("nan")
    return float(np.mean(np.expm1(lr) - lr))


def frac_outside(r, lo, hi):
    r = np.asarray(r)
    return float(np.mean((r < lo) | (r > hi))) if r.size else float("nan")


def tis(r, cap=2.0):
    """Truncated importance sampling: cap each token's weight from above.

    Returns (clipped weights, fraction of tokens that hit the cap). Weights
    below 1 are left alone, as in the usual TIS / veRL rollout correction.
    """
    r = np.asarray(r, dtype=np.float64)
    hit = r > cap
    return np.minimum(r, cap), float(hit.mean()) if r.size else float("nan")


def token_summary(lp_s, lp_t):
    lp_s = np.asarray(lp_s, dtype=np.float64)
    lp_t = np.asarray(lp_t, dtype=np.float64)
    d = lp_t - lp_s
    ad = np.abs(d)
    r = np.exp(d)
    _, clipped = tis(r, 2.0)
    return {
        "tokens": int(d.size),
        "mean_abs_dlogp": float(ad.mean()),
        "p99_abs_dlogp": float(np.quantile(ad, 0.99)),
        "max_abs_dlogp": float(ad.max()),
        "max_abs_dprob": float(np.abs(np.exp(lp_t) - np.exp(lp_s)).max()),
        "k3_kl": k3_kl(lp_s, lp_t),
        "frac_w_outside_0.8_1.25": frac_outside(r, 0.8, 1.25),
        "frac_w_outside_0.5_2": frac_outside(r, 0.5, 2.0),
        "frac_clipped_tis_c2": clipped,
    }


def prob_bin(p):
    """Bin index for each probability: [0,.01) [.01,.1) [.1,.5) [.5,1]."""
    return np.searchsorted(PROB_EDGES[1:-1], np.asarray(p), side="right")


def by_prob_bin(p_ref, lp_s, lp_t):
    """How the gap splits across token-probability bins.

    `p_ref` is the probability used for binning. I bin by a reference policy
    (fp32), not by either side being compared, so a token doesn't land in a
    lower bin just because one side underestimated it.
    """
    ad = np.abs(np.asarray(lp_t, dtype=np.float64) - np.asarray(lp_s, dtype=np.float64))
    r = np.exp(np.asarray(lp_t, dtype=np.float64) - np.asarray(lp_s, dtype=np.float64))
    lr = np.log(r)
    p_ref = np.asarray(p_ref, dtype=np.float64)
    b = prob_bin(p_ref)
    total = ad.sum()
    n_out = ((r < 0.5) | (r > 2.0)).sum()
    rows = []
    for i, label in enumerate(PROB_LABELS):
        m = b == i
        rows.append({
            "bin": label,
            "token_share": float(m.mean()),
            "mean_abs_dlogp": float(ad[m].mean()) if m.any() else float("nan"),
            # the vocabulary-pruning bound says |dlogp| scales with (1 - p)
            "mean_abs_dlogp_over_mean_1mp": float(ad[m].mean() / (1 - p_ref[m]).mean()) if m.any() else float("nan"),
            "mean_signed_log_w": float(lr[m].mean()) if m.any() else float("nan"),
            "share_of_abs_dlogp": float(ad[m].sum() / total) if total > 0 else float("nan"),
            "share_of_w_outside_0.5_2": float(((r[m] < 0.5) | (r[m] > 2.0)).sum() / n_out) if n_out else float("nan"),
        })
    return rows


def by_position(pos, lp_s, lp_t, width=128):
    pos = np.asarray(pos)
    ad = np.abs(np.asarray(lp_t, dtype=np.float64) - np.asarray(lp_s, dtype=np.float64))
    bucket = pos // width
    rows = []
    for b in np.unique(bucket):
        m = bucket == b
        rows.append({"start": int(b * width), "tokens": int(m.sum()), "mean_abs_dlogp": float(ad[m].mean())})
    return rows


def seq_log_ratio(seq_id, lp_s, lp_t):
    """Sum of per-token log ratios for each sequence: log of the sequence-level weight.

    Returns (ids, sums, lengths) with ids sorted.
    """
    seq_id = np.asarray(seq_id)
    lr = np.asarray(lp_t, dtype=np.float64) - np.asarray(lp_s, dtype=np.float64)
    ids, inv = np.unique(seq_id, return_inverse=True)
    sums = np.bincount(inv, weights=lr)
    lengths = np.bincount(inv)
    return ids, sums, lengths


def spread_by_length(sums, lengths, edges=(0, 128, 256, 512, 1025)):
    """Std and mean of the sequence log ratio within each length bucket."""
    sums = np.asarray(sums)
    lengths = np.asarray(lengths)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lengths >= lo) & (lengths < hi)
        if m.sum() < 2:
            continue
        rows.append({
            "len_lo": int(lo), "len_hi": int(hi), "seqs": int(m.sum()),
            "mean": float(sums[m].mean()), "std": float(sums[m].std(ddof=1)),
            # flat across buckets if every token adds an independent error of the same size
            "std_over_sqrt_len": float((sums[m] / np.sqrt(lengths[m])).std(ddof=1)),
            "frac_seq_w_outside_0.5_2": float(np.mean(np.abs(sums[m]) > np.log(2.0))),
        })
    return rows


def bootstrap_ratio(num, den, n_boot=2000, seed=0, q=(0.025, 0.975)):
    """CI for sum(num) / sum(den) when whole sequences are resampled.

    Tokens inside one answer are not independent, so the unit of resampling
    is the sequence: num[i] is a per-sequence total (e.g. sum of |dlogp|),
    den[i] its token count.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.quantile(stats, q)
    return float(lo), float(hi)
