"""The "inference engine": MLX, one token at a time with a KV cache.

It records log mu(token) for each token it samples, the way an RL rollout
worker hands its logprobs to the trainer.
"""

import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# quantized configs start from the bf16 weights, so activations stay bf16
CONFIGS = {
    "fp32": (mx.float32, None),
    "bf16": (mx.bfloat16, None),
    "fp16": (mx.float16, None),
    "q8": (mx.bfloat16, 8),
    "q4": (mx.bfloat16, 4),
}


def load_model(model_id, config):
    dtype, bits = CONFIGS[config]
    model, tokenizer = load(model_id)
    model.set_dtype(dtype)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model, tokenizer


def sample(model, prompt, eos_ids, max_new, seed):
    """Sample one completion at temperature 1. Returns (tokens, logprobs)."""
    key = mx.random.key(seed)
    cache = make_prompt_cache(model)
    x = mx.array(prompt)[None]
    tokens, logprobs = [], []
    for _ in range(max_new):
        logits = model(x, cache=cache)[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        key, sub = mx.random.split(key)
        tok = mx.random.categorical(lp, key=sub)
        chosen = lp[tok]
        mx.eval(tok, chosen)
        t = tok.item()
        tokens.append(t)
        logprobs.append(chosen.item())
        if t in eos_ids:
            break
        x = mx.array([[t]])
    return tokens, logprobs


def head_fp32(model):
    """The output layer's weight in fp32 (dequantized for 8/4-bit models)."""
    layer = model.model.embed_tokens if model.args.tie_word_embeddings else model.lm_head
    w = layer.weight
    if hasattr(layer, "scales"):
        w = mx.dequantize(w, layer.scales, layer.biases, layer.group_size, layer.bits)
    w = w.astype(mx.float32)
    mx.eval(w)
    return w


def score(model, prompt, tokens, w32=None):
    """Same tokens, one full forward pass with no cache (how a trainer would see them).

    With w32 (from head_fp32) the output layer runs in fp32 instead of the
    model's precision.
    """
    ids = mx.array(prompt + tokens[:-1])[None]
    h = model.model(ids)[0, len(prompt) - 1:]
    # only the answer positions go through the output layer
    if w32 is not None:
        logits = h.astype(mx.float32) @ w32.T
    elif model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    logits = logits.astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(lp, mx.array(tokens)[:, None], axis=-1)[:, 0]
    mx.eval(picked)
    return picked.tolist()


def replay(model, prompt, tokens, w32=None):
    """Feed given tokens through the same cached decode path sample() uses.

    Without w32 this repeats sample()'s arithmetic exactly, only with the
    token forced instead of drawn, so it should give back the recorded
    logprobs bit for bit.
    """
    cache = make_prompt_cache(model)
    x = mx.array(prompt)[None]
    out = []
    for t in tokens:
        if w32 is None:
            logits = model(x, cache=cache)[0, -1].astype(mx.float32)
        else:
            logits = model.model(x, cache=cache)[0, -1].astype(mx.float32) @ w32.T
        lp = logits - mx.logsumexp(logits)
        chosen = lp[t]
        mx.eval(chosen)
        out.append(chosen.item())
        x = mx.array([[t]])
    return out


def eos_ids(tokenizer):
    return {tokenizer.eos_token_id, *tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])}


def run(model_id, config, prompts, max_new, seed=0, log=print):
    model, tokenizer = load_model(model_id, config)
    eos = eos_ids(tokenizer)
    rows = []
    t0, n_tok = time.time(), 0
    for i, (prompt, text_gold) in enumerate(prompts):
        toks, lps = sample(model, prompt, eos, max_new, seed + i)
        n_tok += len(toks)
        rows.append({
            "prompt_idx": i, "prompt_ids": prompt, "tokens": toks, "lp_sampler": lps,
            "finished": toks[-1] in eos, "text": tokenizer.decode(toks), "gold": text_gold,
        })
        if (i + 1) % 16 == 0:
            log(f"  {config}: {i + 1}/{len(prompts)} prompts, {n_tok / (time.time() - t0):.1f} tok/s")
    return rows, model
