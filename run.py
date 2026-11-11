"""Sample with MLX, rescore with MLX (no cache) and PyTorch, then report.

    uv run python run.py                       # 1.5B, 128 GSM8K prompts
    uv run python run.py --model Qwen/Qwen2.5-0.5B-Instruct
    uv run python run.py --only report         # redo tables and charts

Stages: sample (MLX), ablate (MLX replay and fp32 output layer checks),
score (PyTorch), report.

Each stage skips work whose output file already exists, so a stopped run
picks up where it left off. Delete the results folder to start clean.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd

from gap import prompts as P

SAMPLER_CONFIGS = ["fp32", "bf16", "fp16", "q8", "q4"]
TRAINER_DTYPES = ["fp32", "bf16", "fp16"]
BATCH = 8


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stage_sample(args, out):
    import mlx.core as mx
    from transformers import AutoTokenizer

    from gap import sampler

    tok = AutoTokenizer.from_pretrained(args.model)
    qs = P.gsm8k(args.prompts)
    prompts = [(P.prompt_ids(tok, q), gold) for q, gold in qs]

    for cfg in args.configs:
        path = out / f"rollouts_{cfg}.parquet"
        if path.exists():
            continue
        log(f"sampling {cfg}")
        t0 = time.time()
        rows, model = sampler.run(args.model, cfg, prompts, args.max_new, seed=args.seed, log=log)
        log(f"  {cfg}: sampled in {time.time() - t0:.0f}s, rescoring without cache")
        for r in rows:
            r["lp_mlx_nocache"] = sampler.score(model, r["prompt_ids"], r["tokens"])
            r["correct"] = P.is_correct(r["text"], r["gold"])

        # Same seed, same model: the sampler should repeat itself exactly.
        # If it didn't, run-to-run noise would be mixed into every number below.
        again = []
        for r in rows[:4]:
            t, lp = sampler.sample(model, r["prompt_ids"], sampler.eos_ids(tok), args.max_new,
                                   args.seed + r["prompt_idx"])
            again.append(t == r["tokens"] and lp == r["lp_sampler"])
        for r in rows:
            r["config"] = cfg
            r["rerun_identical"] = all(again)
        pd.DataFrame(rows).to_parquet(path)
        del model
        gc.collect()
        mx.clear_cache()


def stage_ablate(args, out):
    """Two checks on the MLX side, for bf16 and fp16 rollouts.

    1. Replay the first prompts through the cached decode path with the token
       forced: it should give back the recorded logprobs exactly.
    2. Redo the scoring with the output layer in fp32, to see how much of the
       gap is just the logits being rounded to 16 bits. bf16 gets this on the
       cached path too (a full replay, as slow as sampling); fp16 only without
       the cache.
    """
    import mlx.core as mx

    from gap import sampler

    path = out / "mlx_extra.parquet"
    if path.exists():
        return
    parts, checks = [], {}
    for cfg in [c for c in ("bf16", "fp16") if c in args.configs]:
        log(f"mlx extras for {cfg}")
        model, _ = sampler.load_model(args.model, cfg)
        w32 = sampler.head_fp32(model)
        roll = pd.read_parquet(out / f"rollouts_{cfg}.parquet")
        same = [sampler.replay(model, r.prompt_ids.tolist(), r.tokens.tolist()) == r.lp_sampler.tolist()
                for r in roll.head(8).itertuples()]
        checks[cfg] = {"replayed": len(same), "identical": int(sum(same))}
        for r in roll.itertuples():
            p, t = r.prompt_ids.tolist(), r.tokens.tolist()
            parts.append(pd.DataFrame({
                "config": cfg, "prompt_idx": r.prompt_idx, "pos": range(len(t)),
                "lp_mlx_nocache_fp32head": sampler.score(model, p, t, w32),
                "lp_replay_fp32head": sampler.replay(model, p, t, w32) if cfg == "bf16" else float("nan"),
            }))
        del model, w32
        gc.collect()
        mx.clear_cache()
    (out / "mlx_extra.json").write_text(json.dumps(checks, indent=2))
    pd.concat(parts, ignore_index=True).to_parquet(path)


def stage_score(args, out):
    import torch
    from transformers import AutoTokenizer

    from gap import trainer

    path = out / "tokens.parquet"
    if path.exists():
        return
    roll = pd.concat([pd.read_parquet(out / f"rollouts_{c}.parquet") for c in args.configs], ignore_index=True)
    pad_id = AutoTokenizer.from_pretrained(args.model).pad_token_id
    seqs = [(list(p), list(t)) for p, t in zip(roll.prompt_ids, roll.tokens)]
    n = len(seqs)
    scored = {}

    def batches(idx, **kw):
        res = {}
        for j in range(0, len(idx), BATCH):
            chunk = idx[j : j + BATCH]
            res.update(zip(chunk, trainer.score(model, [seqs[k] for k in chunk], pad_id, **kw)))
        return [res.get(k) for k in range(n)]

    all_idx = list(range(n))
    own = {c: roll.index[roll.config == c].tolist() for c in args.configs}
    mine = {c: set(v) for c, v in own.items()}
    for dt in TRAINER_DTYPES:
        log(f"scoring with torch {dt}")
        model = trainer.load_model(args.model, dt)
        scored[f"lp_torch_{dt}"] = [trainer.score(model, [s], pad_id)[0] for s in seqs]
        if dt in ("bf16", "fp16") and dt in own:
            # same body, output layer in fp32, on the matching rollouts only
            w32 = model.lm_head.weight.float()
            scored[f"lp_torch_{dt}_fp32head"] = [
                trainer.score(model, [seqs[k]], pad_id, w32=w32)[0] if k in mine[dt] else None for k in all_idx]
            del w32
        if dt == "bf16":
            log(f"scoring with torch bf16 in padded batches of {BATCH}")
            scored["lp_torch_bf16_batch8"] = batches([k for c in args.configs for k in own[c]])
            if "bf16" in own:
                scored["lp_torch_bf16_batch8_left"] = batches(own["bf16"], left_pad=True)
        del model
        gc.collect()
        torch.mps.empty_cache()
    if "bf16" in own:
        # a known kernel change in the trainer, so a zero from batching means something
        log("scoring with torch bf16, eager attention")
        model = trainer.load_model(args.model, "bf16", attn="eager")
        scored["lp_torch_bf16_eager"] = [
            trainer.score(model, [seqs[k]], pad_id)[0] if k in mine["bf16"] else None for k in all_idx]
        del model
        gc.collect()
        torch.mps.empty_cache()

    parts = []
    for i, r in roll.iterrows():
        part = pd.DataFrame({
            "config": r.config, "prompt_idx": r.prompt_idx, "pos": range(len(r.tokens)),
            "token": r.tokens, "lp_sampler": r.lp_sampler, "lp_mlx_nocache": r.lp_mlx_nocache,
        })
        for col, vals in scored.items():
            part[col] = vals[i] if vals[i] is not None else float("nan")
        parts.append(part)
    tok = pd.concat(parts, ignore_index=True)
    extra = out / "mlx_extra.parquet"
    if extra.exists():
        tok = tok.merge(pd.read_parquet(extra), on=["config", "prompt_idx", "pos"], how="left")
    tok.to_parquet(path)
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=SAMPLER_CONFIGS)
    ap.add_argument("--only", choices=["sample", "ablate", "score", "report"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or Path("results") / args.model.split("/")[-1].lower()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "model": args.model, "prompts": args.prompts, "max_new": args.max_new,
        "seed": args.seed, "configs": args.configs, "temperature": 1.0,
    }, indent=2))

    if args.only in (None, "sample"):
        stage_sample(args, out)
    if args.only in (None, "ablate"):
        stage_ablate(args, out)
    if args.only in (None, "score"):
        stage_score(args, out)
    if args.only in (None, "report"):
        from gap import report
        report.write(out, args.configs)


if __name__ == "__main__":
    main()
