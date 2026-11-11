"""Tables (summary.json, summary.md) and charts from tokens.parquet."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gap import metrics as M

# which trainer precision each sampler is compared against
MATCH = {"fp32": "fp32", "bf16": "bf16", "fp16": "fp16", "q8": "bf16", "q4": "bf16"}
LABEL = {"fp32": "fp32", "bf16": "bf16", "fp16": "fp16", "q8": "8-bit", "q4": "4-bit"}
COLOR = {"fp32": "#2a78d6", "bf16": "#eb6834", "fp16": "#1baf7a", "q8": "#eda100", "q4": "#e87ba4"}
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "axes.axisbelow": True,
})


def comparisons(cfg):
    """(name, side A column, side B column) pairs for one sampler config."""
    t = f"lp_torch_{MATCH[cfg]}"
    pairs = [
        ("sampler vs trainer", "lp_sampler", t),
        ("sampler vs fp32 trainer", "lp_sampler", "lp_torch_fp32"),
        ("KV cache vs no cache (MLX)", "lp_sampler", "lp_mlx_nocache"),
        ("MLX vs PyTorch (no cache)", "lp_mlx_nocache", t),
    ]
    if cfg != "fp32":
        pairs.append(("trainer vs fp32 trainer", t, "lp_torch_fp32"))
    return pairs


def _mean_abs(df, a, b):
    d = (df[a] - df[b]).dropna()
    return float(d.abs().mean()) if len(d) else None


def precision_same_tokens(tok):
    """PyTorch bf16 and fp16, each against PyTorch fp32, on the same tokens.

    bf16 keeps 8 significant bits and fp16 keeps 11, so if rounding is the
    whole story the ratio should be near 2^3 = 8.
    """
    rows = {}
    for cfg, df in list(tok.groupby("config")) + [("all", tok)]:
        b, f = _mean_abs(df, "lp_torch_bf16", "lp_torch_fp32"), _mean_abs(df, "lp_torch_fp16", "lp_torch_fp32")
        rows[cfg] = {"bf16_vs_fp32": b, "fp16_vs_fp32": f, "ratio": b / f}
    return rows


def noise_model(df, a, b):
    """Treat each token's log ratio as noise of size sigma * (1 - p).

    sigma comes from tokens with p < 0.9. If the model holds, a whole answer's
    log weight divided by sqrt(sum (1 - p)^2) has the same spread at every
    length, while dividing by sqrt(length) doesn't account for how uncertain
    the answer was.
    """
    p = np.exp(df.lp_torch_fp32.to_numpy())
    lr = (df[b] - df[a]).to_numpy()
    keep = p < 0.9
    sigma = float(np.sqrt(np.mean((lr[keep] / (1 - p[keep])) ** 2)))
    per = pd.DataFrame({"prompt_idx": df.prompt_idx.to_numpy(), "lr": lr, "s2": (1 - p) ** 2,
                        "surprisal": -df.lp_torch_fp32.to_numpy(), "ad": np.abs(lr)})
    g = per.groupby("prompt_idx").agg(lw=("lr", "sum"), s2=("s2", "sum"), n=("lr", "size"),
                                      surprisal=("surprisal", "mean"), ad=("ad", "mean"))
    lag = [(x[:-1], x[1:]) for _, x in per.groupby("prompt_idx").lr if len(x) > 2]
    a1 = np.concatenate([u for u, _ in lag])
    a2 = np.concatenate([v for _, v in lag])
    sq = per.lr ** 2
    out = {
        "sigma": sigma,
        "lag1_autocorr": float(np.corrcoef(a1, a2)[0, 1]),
        "share_of_sq_from_p_below_0.5": float(sq[p < 0.5].sum() / sq.sum()),
        "corr_abs_log_w_vs_sqrt_sum_1mp2": float(np.corrcoef(g.lw.abs(), np.sqrt(g.s2))[0, 1]),
        "corr_abs_log_w_vs_len": float(np.corrcoef(g.lw.abs(), g.n)[0, 1]),
        "by_length": [],
    }
    for lo, hi in ((0, 256), (256, 512), (512, 100000)):
        m = (g.n >= lo) & (g.n < hi)
        if m.sum() < 2:
            continue
        out["by_length"].append({
            "len_lo": lo, "len_hi": hi, "seqs": int(m.sum()),
            "mean_surprisal": float(g.surprisal[m].mean()), "mean_abs_dlogp": float(g.ad[m].mean()),
            "std_log_w_over_sqrt_len": float((g.lw[m] / np.sqrt(g.n[m])).std(ddof=1)),
            "std_log_w_over_sqrt_sum_1mp2": float((g.lw[m] / np.sqrt(g.s2[m])).std(ddof=1)),
        })
    return out, g


def path_errors(df, prec):
    """Each path's distance from PyTorch fp32, and how correlated their errors are."""
    cols = {"sampler (MLX, KV cache)": "lp_sampler", "MLX, no cache": "lp_mlx_nocache",
            f"PyTorch {prec}": f"lp_torch_{prec}"}
    err = {k: (df[c] - df.lp_torch_fp32).to_numpy() for k, c in cols.items()}
    names = list(err)
    return {
        "mean_abs_from_fp32": {k: float(np.abs(v).mean()) for k, v in err.items()},
        "error_corr": {f"{names[i]} / {names[j]}": float(np.corrcoef(err[names[i]], err[names[j]])[0, 1])
                       for i in range(3) for j in range(i + 1, 3)},
    }


def head_ablation(tok):
    """Same comparisons with the output layer in fp32 on both sides."""
    out = {}
    for prec in ("bf16", "fp16"):
        df = tok[tok.config == prec]
        h = f"lp_torch_{prec}_fp32head"
        if h not in df or df[h].isna().all():
            continue
        r = {
            "MLX vs PyTorch, no cache": [_mean_abs(df, "lp_mlx_nocache", f"lp_torch_{prec}"),
                                         _mean_abs(df, "lp_mlx_nocache_fp32head", h)],
            "trainer vs fp32 trainer": [_mean_abs(df, f"lp_torch_{prec}", "lp_torch_fp32"), _mean_abs(df, h, "lp_torch_fp32")],
        }
        if not df.lp_replay_fp32head.isna().all():
            r["sampler vs trainer"] = [_mean_abs(df, "lp_sampler", f"lp_torch_{prec}"), _mean_abs(df, "lp_replay_fp32head", h)]
            r["sampler vs fp32 trainer"] = [_mean_abs(df, "lp_sampler", "lp_torch_fp32"), _mean_abs(df, "lp_replay_fp32head", "lp_torch_fp32")]
            lr = (df[h] - df.lp_replay_fp32head)
            ids, sums, _ = M.seq_log_ratio(df.prompt_idx.to_numpy(), df.lp_replay_fp32head.to_numpy(), df[h].to_numpy())
            r["answers_w_outside_0.5_2"] = [None, float(np.mean(np.abs(sums) > np.log(2)))]
            r["k3_kl"] = [M.k3_kl(df.lp_sampler, df[f"lp_torch_{prec}"]), M.k3_kl(df.lp_replay_fp32head, df[h])]
        out[prec] = {k: {"as_run": v[0], "fp32_output_layer": v[1]} for k, v in r.items()}
    return out


def controls(tok, out):
    """Batching and a known kernel change in the trainer, and the sampler replay check."""
    c = {}
    d = tok.dropna(subset=["lp_torch_bf16_batch8"])
    c["right_padded_batch_of_8"] = {"tokens": len(d), "identical": int((d.lp_torch_bf16 == d.lp_torch_bf16_batch8).sum()),
                                    "max_abs": float((d.lp_torch_bf16 - d.lp_torch_bf16_batch8).abs().max())}
    for name, col in (("left_padded_batch_of_8", "lp_torch_bf16_batch8_left"), ("eager_attention", "lp_torch_bf16_eager")):
        if col in tok:
            d = tok.dropna(subset=[col])
            c[name] = {"tokens": len(d), "identical": int((d.lp_torch_bf16 == d[col]).sum()),
                       "mean_abs": float((d.lp_torch_bf16 - d[col]).abs().mean()),
                       "max_abs": float((d.lp_torch_bf16 - d[col]).abs().max())}
    extra = Path(out) / "mlx_extra.json"
    if extra.exists():
        c["sampler_replay"] = json.loads(extra.read_text())
    return c


def _with_ci(df, a, b):
    s = M.token_summary(df[a].to_numpy(), df[b].to_numpy())
    g = df.assign(ad=(df[b] - df[a]).abs()).groupby("prompt_idx")["ad"].agg(["sum", "count"])
    s["mean_abs_dlogp_ci95"] = M.bootstrap_ratio(g["sum"].to_numpy(), g["count"].to_numpy())
    return s


def first_divergence(roll):
    """Token index where each config's rollout first differs from the fp32 rollout.

    Every config samples prompt i with the same seed, so until the numbers
    push some draw across a boundary, they produce the same text.
    """
    if "fp32" not in roll:
        return {}
    ref = roll["fp32"].set_index("prompt_idx").tokens
    out = {}
    for cfg, df in roll.items():
        if cfg == "fp32":
            continue
        first = []
        for i, toks in zip(df.prompt_idx, df.tokens):
            a, b = list(ref[i]), list(toks)
            n = min(len(a), len(b))
            diff = [k for k in range(n) if a[k] != b[k]]
            first.append(diff[0] if diff else n)
        first = np.array(first)
        out[cfg] = {"median": float(np.median(first)), "identical_rollouts": int(sum(
            list(ref[i]) == list(t) for i, t in zip(df.prompt_idx, df.tokens)))}
    return out


def write(out, configs):
    out = Path(out)
    tok = pd.read_parquet(out / "tokens.parquet")
    roll = {c: pd.read_parquet(out / f"rollouts_{c}.parquet") for c in configs}
    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    p_ref = np.exp(tok["lp_torch_fp32"])

    summary = {"run": json.loads((out / "run.json").read_text()), "configs": {}}
    for cfg in configs:
        df = tok[tok.config == cfg]
        r = roll[cfg]
        t = f"lp_torch_{MATCH[cfg]}"
        lengths = r.tokens.map(len)
        ids, sums, lens = M.seq_log_ratio(df.prompt_idx.to_numpy(), df.lp_sampler.to_numpy(), df[t].to_numpy())
        summary["configs"][cfg] = {
            "trainer": MATCH[cfg],
            "accuracy": float(r.correct.mean()),
            "finished": float(r.finished.mean()),
            "mean_len": float(lengths.mean()),
            "sampler_repeats_itself": bool(r.rerun_identical.all()),
            "compare": {name: _with_ci(df, a, b) for name, a, b in comparisons(cfg)},
            "by_prob": M.by_prob_bin(p_ref[df.index], df.lp_sampler, df[t]),
            "by_position": M.by_position(df.pos, df.lp_sampler, df[t]),
            "seq": {
                "std_log_w": float(sums.std(ddof=1)),
                "mean_log_w": float(sums.mean()),
                "frac_seq_w_outside_0.5_2": float(np.mean(np.abs(sums) > np.log(2))),
                "corr_abs_log_w_vs_len": float(np.corrcoef(np.abs(sums), lens)[0, 1]),
                "by_length": M.spread_by_length(sums, lens),
            },
        }
    summary["first_divergence_from_fp32"] = first_divergence(roll)
    summary["precision_same_tokens"] = precision_same_tokens(tok)
    summary["noise_model"] = {}
    groups = {}
    for cfg in [c for c in ("bf16", "fp16", "q8") if c in configs]:
        df = tok[tok.config == cfg]
        summary["noise_model"][cfg], groups[cfg] = noise_model(df, "lp_sampler", f"lp_torch_{MATCH[cfg]}")
    summary["path_errors"] = {c: path_errors(tok[tok.config == c], c) for c in ("bf16", "fp16") if c in configs}
    if "lp_torch_bf16_fp32head" in tok:
        summary["fp32_output_layer"] = head_ablation(tok)
    summary["controls"] = controls(tok, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(to_markdown(summary, configs))

    if {"fp32", "bf16", "fp16"} <= set(configs):
        chart_bf16_fp16(tok, charts / "bf16_vs_fp16.png")
    chart_tail(tok, configs, charts / "tail.png")
    chart_pieces(summary, configs, charts / "comparisons.png")
    if groups:
        chart_noise(tok, summary, groups, charts / "noise_model.png")
    chart_by_prob(summary, [c for c in ("bf16", "fp16", "q4") if c in configs], charts / "by_token_prob.png")
    chart_seq(tok, configs, charts / "sequence_weights.png")
    chart_position(summary, configs, charts / "by_position.png")


def _fmt(x):
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e4:
            return f"{x:.2e}"
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def to_markdown(s, configs):
    run = s["run"]
    lines = [f"# {run['model']}, {run['prompts']} GSM8K prompts, T=1.0, up to {run['max_new']} new tokens", ""]
    lines += ["## Sampler vs trainer", "",
              "| sampler | trainer | tokens | mean abs dlogp (95% CI) | p99 | max | max abs dprob | k3 KL | w outside [0.8,1.25] | w outside [0.5,2] | clipped at C=2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in configs:
        x = s["configs"][c]["compare"]["sampler vs trainer"]
        lo, hi = x["mean_abs_dlogp_ci95"]
        lines.append(f"| {LABEL[c]} | {s['configs'][c]['trainer']} | {x['tokens']} | {_fmt(x['mean_abs_dlogp'])} ({_fmt(lo)} to {_fmt(hi)}) "
                     f"| {_fmt(x['p99_abs_dlogp'])} | {_fmt(x['max_abs_dlogp'])} | {_fmt(x['max_abs_dprob'])} | {_fmt(x['k3_kl'])} "
                     f"| {_fmt(x['frac_w_outside_0.8_1.25'])} | {_fmt(x['frac_w_outside_0.5_2'])} | {_fmt(x['frac_clipped_tis_c2'])} |")
    lines += ["", "## Each comparison on its own (mean abs dlogp; these don't add up)", "",
              "| sampler | " + " | ".join(n for n, _, _ in comparisons("bf16")) + " |",
              "|---|" + "---|" * len(comparisons("bf16"))]
    for c in configs:
        cmp = s["configs"][c]["compare"]
        row = [_fmt(cmp[n]["mean_abs_dlogp"]) if n in cmp else "" for n, _, _ in comparisons("bf16")]
        lines.append(f"| {LABEL[c]} | " + " | ".join(row) + " |")
    lines += ["", "## By token probability (fp32 reference), sampler vs trainer", "",
              "For the fp32 sampler the trainer side is the fp32 reference itself.", "",
              "| sampler | bin | share of tokens | mean abs dlogp | mean abs dlogp / mean (1-p) | mean signed log w | share of total abs dlogp | share of w outside [0.5,2] |",
              "|---|---|---|---|---|---|---|---|"]
    for c in configs:
        for b in s["configs"][c]["by_prob"]:
            lines.append(f"| {LABEL[c]} | {b['bin']} | {_fmt(b['token_share'])} | {_fmt(b['mean_abs_dlogp'])} "
                         f"| {_fmt(b['mean_abs_dlogp_over_mean_1mp'])} | {_fmt(b['mean_signed_log_w'])} "
                         f"| {_fmt(b['share_of_abs_dlogp'])} | {_fmt(b['share_of_w_outside_0.5_2'])} |")
    lines += ["", "## Sequence-level log weight (sum of token log ratios)", "",
              "| sampler | std | mean | seqs with w outside [0.5,2] | corr(abs log w, length) |", "|---|---|---|---|---|"]
    for c in configs:
        q = s["configs"][c]["seq"]
        lines.append(f"| {LABEL[c]} | {_fmt(q['std_log_w'])} | {_fmt(q['mean_log_w'])} | {_fmt(q['frac_seq_w_outside_0.5_2'])} "
                     f"| {_fmt(q['corr_abs_log_w_vs_len'])} |")
    lines += ["", "| sampler | length | seqs | std of log w | std of log w / sqrt(length) | seqs with w outside [0.5,2] |", "|---|---|---|---|---|---|"]
    for c in configs:
        for b in s["configs"][c]["seq"]["by_length"]:
            lines.append(f"| {LABEL[c]} | {b['len_lo']}-{b['len_hi'] - 1} | {b['seqs']} | {_fmt(b['std'])} | {_fmt(b['std_over_sqrt_len'])} | {_fmt(b['frac_seq_w_outside_0.5_2'])} |")

    lines += ["", "## Same tokens, PyTorch bf16 and fp16 against PyTorch fp32", "",
              "bf16 keeps 8 significant bits, fp16 keeps 11, so pure rounding predicts a ratio near 2^3 = 8.", "",
              "| rollouts from | bf16 vs fp32 | fp16 vs fp32 | ratio |", "|---|---|---|---|"]
    for c, r in s["precision_same_tokens"].items():
        lines.append(f"| {LABEL.get(c, c)} | {_fmt(r['bf16_vs_fp32'])} | {_fmt(r['fp16_vs_fp32'])} | {r['ratio']:.2f} |")

    lines += ["", "## Noise model: each token's log ratio as sigma * (1 - p)", "",
              "| pair | sigma | lag-1 autocorr | share of sum of squares from p < 0.5 | corr(abs log w, sqrt sum (1-p)^2) | corr(abs log w, length) |",
              "|---|---|---|---|---|---|"]
    for c, q in s["noise_model"].items():
        lines.append(f"| {LABEL[c]} vs {MATCH[c]} | {_fmt(q['sigma'])} | {_fmt(q['lag1_autocorr'])} | {_fmt(q['share_of_sq_from_p_below_0.5'])} "
                     f"| {_fmt(q['corr_abs_log_w_vs_sqrt_sum_1mp2'])} | {_fmt(q['corr_abs_log_w_vs_len'])} |")
    lines += ["", "| pair | length | seqs | mean surprisal (nats/token, fp32) | mean abs dlogp | std log w / sqrt(length) | std log w / sqrt(sum (1-p)^2) |",
              "|---|---|---|---|---|---|---|"]
    for c, q in s["noise_model"].items():
        for b in q["by_length"]:
            hi = "" if b["len_hi"] > 10000 else b["len_hi"] - 1
            lines.append(f"| {LABEL[c]} vs {MATCH[c]} | {b['len_lo']}-{hi} | {b['seqs']} | {_fmt(b['mean_surprisal'])} | {_fmt(b['mean_abs_dlogp'])} "
                         f"| {_fmt(b['std_log_w_over_sqrt_len'])} | {_fmt(b['std_log_w_over_sqrt_sum_1mp2'])} |")

    lines += ["", "## Each path's error against PyTorch fp32", ""]
    for c, q in s["path_errors"].items():
        lines.append(f"- {c}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in q["mean_abs_from_fp32"].items())
                     + "; error correlations " + ", ".join(f"{k} {v:.2f}" for k, v in q["error_corr"].items()))

    if "fp32_output_layer" in s:
        lines += ["", "## Output layer in fp32 on both sides (mean abs dlogp unless noted)", "",
                  "| precision | comparison | as run | fp32 output layer |", "|---|---|---|---|"]
        for c, q in s["fp32_output_layer"].items():
            for k, v in q.items():
                lines.append(f"| {c} | {k} | {_fmt(v['as_run']) if v['as_run'] is not None else '-'} | {_fmt(v['fp32_output_layer'])} |")

    ctl = s["controls"]
    lines += ["", "## Controls", ""]
    for k, v in ctl.items():
        lines.append(f"- {k}: " + ", ".join(f"{a} {_fmt(b) if isinstance(b, float) else b}" for a, b in v.items()))
    lines += ["", "## Rollouts", "", "| sampler | GSM8K accuracy | finished | mean length | repeats itself with same seed | first token differing from fp32 rollout (median) | rollouts identical to fp32 |",
              "|---|---|---|---|---|---|---|"]
    div = s.get("first_divergence_from_fp32", {})
    for c in configs:
        q = s["configs"][c]
        d = div.get(c, {})
        lines.append(f"| {LABEL[c]} | {_fmt(q['accuracy'])} | {_fmt(q['finished'])} | {q['mean_len']:.0f} | {q['sampler_repeats_itself']} "
                     f"| {_fmt(d.get('median', '-'))} | {d.get('identical_rollouts', '-')} |")
    return "\n".join(lines) + "\n"


def _ccdf(ax, ad, color, label):
    """Share of tokens with |Δ| >= x. Exact zeros are counted but can't be drawn on a log axis."""
    ad = np.sort(ad)
    frac = 1 - np.arange(len(ad)) / len(ad)
    keep = ad > 0
    ax.loglog(ad[keep], frac[keep], color=color, lw=2, label=label)


def _w_lines(ax):
    for w, y in ((1.25, 0.98), (2, 0.90)):
        ax.axvline(np.log(w), color=INK2, lw=0.8, ls="--")
        ax.text(np.log(w), y, f" w={w:g}", transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, va="top")


def chart_bf16_fp16(tok, path):
    """The two numbers I'd show first: per token and per answer, bf16 against fp16."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})
    for c in ("fp32", "bf16", "fp16"):
        df = tok[tok.config == c]
        ad = (df[f"lp_torch_{c}"] - df.lp_sampler).abs().to_numpy()
        _ccdf(a1, ad, COLOR[c], f"{c} sampler vs {c} trainer")
    _w_lines(a1)
    a1.set_xlim(1e-7, 1)
    a1.set_xlabel("|Δ logprob| of a sampled token")
    a1.set_ylabel("share of tokens at least this far off")
    a1.set_title("Per token")
    a2.axhspan(-np.log(2), np.log(2), color=GRID, alpha=0.7, lw=0)
    a2.text(0.99, np.log(2), "w = 2 ", transform=a2.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=INK2)
    a2.text(0.99, -np.log(2), "w = 0.5 ", transform=a2.get_yaxis_transform(), ha="right", va="top", fontsize=8, color=INK2)
    for c in ("bf16", "fp16"):
        df = tok[tok.config == c]
        _, sums, lens = M.seq_log_ratio(df.prompt_idx.to_numpy(), df.lp_sampler.to_numpy(), df[f"lp_torch_{c}"].to_numpy())
        a2.scatter(lens, sums, s=14, color=COLOR[c], edgecolor=BG, linewidth=0.5)
    a2.axhline(0, color=INK2, lw=0.6)
    a2.set_xlabel("answer length (tokens)")
    a2.set_ylabel("log of the whole answer's weight")
    a2.set_title("Per answer (one dot per GSM8K answer)")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_tail(tok, configs, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in configs:
        df = tok[tok.config == c]
        _ccdf(ax, (df[f"lp_torch_{MATCH[c]}"] - df.lp_sampler).abs().to_numpy(), COLOR[c], f"{LABEL[c]} sampler vs {MATCH[c]} trainer")
    _w_lines(ax)
    ax.set_xlim(1e-8, 20)
    ax.set_xlabel("|Δ logprob| of a sampled token")
    ax.set_ylabel("share of tokens at least this far off")
    ax.set_title("How often a token is off by at least x")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_pieces(s, configs, path):
    shown = [c for c in ("bf16", "fp16") if c in configs]
    fig, axes = plt.subplots(1, len(shown) + 1, figsize=(4.2 * len(shown) + 3.4, 4.2),
                             gridspec_kw={"width_ratios": [1] * len(shown) + [0.8]})

    def bars(ax, rows, color, title, labels):
        ys = np.arange(len(rows))[::-1]
        for y, (name, v) in zip(ys, rows):
            if v is None:
                ax.text(2e-6, y, " not run", va="center", fontsize=8, color=INK2)
            elif v == 0:
                ax.text(2e-6, y, " 0 (bit-identical)", va="center", fontsize=8, color=INK2)
            else:
                ax.barh(y, v, height=0.55, color=INK2 if name.startswith("sampler vs trainer") else color)
                ax.text(v, y, f" {v:.1e}", va="center", fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(2e-6, 0.3)
        ax.set_yticks(ys, [n for n, _ in rows] if labels else [""] * len(rows))
        ax.set_title(title)
        ax.set_xlabel("mean |Δ logprob|")

    for k, c in enumerate(shown):
        cmp = s["configs"][c]["compare"]
        h = s.get("fp32_output_layer", {}).get(c, {})
        get = lambda name: h.get(name, {}).get("fp32_output_layer")
        rows = [
            ("sampler vs trainer", cmp["sampler vs trainer"]["mean_abs_dlogp"]),
            ("  same, fp32 output layer", get("sampler vs trainer")),
            ("MLX vs PyTorch, no cache", cmp["MLX vs PyTorch (no cache)"]["mean_abs_dlogp"]),
            ("  same, fp32 output layer", get("MLX vs PyTorch, no cache")),
            ("KV cache vs no cache (MLX)", cmp["KV cache vs no cache (MLX)"]["mean_abs_dlogp"]),
            ("trainer vs fp32 trainer", cmp["trainer vs fp32 trainer"]["mean_abs_dlogp"]),
            ("  same, fp32 output layer", get("trainer vs fp32 trainer")),
        ]
        bars(axes[k], rows, COLOR[c], c, k == 0)
    ctl = s["controls"]
    rows = [
        ("padded batch of 8 (right)", ctl["right_padded_batch_of_8"]["max_abs"]),
        ("padded batch of 8 (left)", ctl.get("left_padded_batch_of_8", {}).get("mean_abs")),
        ("eager vs SDPA attention", ctl.get("eager_attention", {}).get("mean_abs")),
    ]
    bars(axes[-1], rows, COLOR["bf16"], "bf16 trainer, vs scored alone", True)
    axes[-1].yaxis.tick_right()
    fig.suptitle("Each comparison measured on its own (they don't add up)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_noise(tok, s, groups, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    edges = np.logspace(-6, 0, 19)
    for c in [c for c in ("bf16", "fp16") if c in groups]:
        df = tok[tok.config == c]
        q = 1 - np.exp(df.lp_torch_fp32.to_numpy())
        ad = (df[f"lp_torch_{MATCH[c]}"] - df.lp_sampler).abs().to_numpy()
        b = np.digitize(q, edges)
        xs, ys = [], []
        for i in np.unique(b):
            m = b == i
            if m.sum() >= 30 and 0 < i < len(edges):
                xs.append(np.sqrt(edges[i - 1] * edges[i]))
                ys.append(ad[m].mean())
        a1.loglog(xs, ys, "o-", color=COLOR[c], lw=2, ms=5, label=f"{c} sampler vs {c} trainer")
        sig = s["noise_model"][c]["sigma"]
        a1.loglog(edges, sig * np.sqrt(2 / np.pi) * edges, "--", color=COLOR[c], lw=1)
        g = groups[c]
        a2.loglog(np.sqrt(g.s2), g.lw.abs(), "o", color=COLOR[c], ms=4, mec=BG, mew=0.5, label=c)
        x = np.array([np.sqrt(g.s2).min(), np.sqrt(g.s2).max()])
        a2.loglog(x, sig * np.sqrt(2 / np.pi) * x, "--", color=COLOR[c], lw=1)
    a1.set_xlabel("1 − p   (p = the token's probability under fp32)")
    a1.set_ylabel("mean |Δ logprob|")
    a1.set_title("Per token: the gap grows with 1 − p")
    a1.legend(fontsize=8, loc="upper left")
    a2.axhline(np.log(2), color=INK2, lw=0.8, ls=":")
    a2.text(0.98, np.log(2), "w = 2 or 0.5", transform=a2.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=INK2)
    a2.set_xlabel("√ Σ (1 − p)²  over the answer")
    a2.set_ylabel("|log of the whole answer's weight|")
    a2.set_title("Per answer: drift follows the uncertain tokens")
    a2.legend(fontsize=8, loc="lower right")
    fig.text(0.5, 0.01, "dashed lines: one noise size σ per precision, fitted on tokens, times (1 − p)", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_by_prob(s, configs, path):
    if not configs:
        return
    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(M.PROB_LABELS))
    for ax, c in zip(axes, configs):
        rows = s["configs"][c]["by_prob"]
        series = [("share of tokens", "token_share", "#2a78d6"),
                  ("share of total |Δ logprob|", "share_of_abs_dlogp", "#eb6834"),
                  ("share of weights outside [0.5, 2]", "share_of_w_outside_0.5_2", "#1baf7a")]
        for k, (lab, key, col) in enumerate(series):
            ax.bar(x + (k - 1) * 0.27, [r[key] * 100 for r in rows], width=0.25, color=col, label=lab)
        ax.set_xticks(x, M.PROB_LABELS)
        ax.set_xlabel("token probability under fp32")
        ax.set_title(f"{LABEL[c]} sampler vs {MATCH[c]} trainer")
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_seq(tok, configs, path):
    fig, axes = plt.subplots(1, len(configs), figsize=(3.2 * len(configs), 3.3))
    axes = np.atleast_1d(axes)
    data = {}
    for c in configs:
        df = tok[tok.config == c]
        _, sums, lens = M.seq_log_ratio(df.prompt_idx.to_numpy(), df.lp_sampler.to_numpy(), df[f"lp_torch_{MATCH[c]}"].to_numpy())
        data[c] = (sums, lens)
    # 4-bit is two orders of magnitude off the others, so it gets its own scale
    shared = [c for c in configs if c != "q4"]
    lim = max(np.abs(data[c][0]).max() for c in shared) * 1.1 if shared else 1
    for ax, c in zip(axes, configs):
        sums, lens = data[c]
        ax.axhspan(-np.log(2), np.log(2), color=GRID, alpha=0.7, lw=0)
        ax.scatter(lens, sums, s=12, color=COLOR[c], edgecolor=BG, linewidth=0.5)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{LABEL[c]} vs {MATCH[c]}" + (" (own scale)" if c == "q4" else ""))
        ax.set_xlabel("answer length (tokens)")
        if c != "q4":
            ax.set_ylim(-lim, lim)
            if c != configs[0]:
                ax.set_yticklabels([])
    axes[0].set_ylabel("log of the whole answer's weight")
    fig.suptitle("Whole-answer importance weights; gray band is w in [0.5, 2]", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_position(s, configs, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in configs:
        rows = [r for r in s["configs"][c]["by_position"] if r["tokens"] >= 200]
        ax.plot([r["start"] for r in rows], [r["mean_abs_dlogp"] for r in rows], marker="o", ms=4,
                color=COLOR[c], lw=2, label=f"{LABEL[c]} vs {MATCH[c]}")
    ax.set_yscale("log")
    ax.set_xlabel("position in the answer (start of each 128-token bucket; buckets under 200 tokens left out)")
    ax.set_ylabel("mean |Δ logprob|")
    ax.set_title("Does the gap grow later in the answer?")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
