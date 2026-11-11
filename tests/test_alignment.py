"""The two scorers have to line up position i with token i. These check that on
tiny random models (no download, runs on CPU in a few seconds)."""

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gap import trainer

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from gap import sampler  # noqa: E402


@pytest.fixture(scope="module")
def tiny_torch():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).eval()


def test_trainer_picks_the_right_positions(tiny_torch):
    prompt, toks = [5, 9, 2, 40], [7, 7, 31, 0, 12]
    got = trainer.score(tiny_torch, [(prompt, toks)])[0]
    with torch.no_grad():
        logits = tiny_torch(torch.tensor([prompt + toks])).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    want = [lp[len(prompt) - 1 + i, t].item() for i, t in enumerate(toks)]
    assert got == pytest.approx(want, abs=1e-6)


def test_trainer_padded_batch_matches_single(tiny_torch):
    seqs = [([1, 2, 3], [4, 5, 6, 7, 8, 9]), ([10, 11], [12]), ([13, 14, 15, 16, 17], [18, 19])]
    single = [trainer.score(tiny_torch, [s], pad_id=0)[0] for s in seqs]
    batch = trainer.score(tiny_torch, seqs, pad_id=0)
    for a, b in zip(single, batch):
        assert len(a) == len(b)
        assert a == pytest.approx(b, abs=1e-5)


@pytest.fixture(scope="module")
def tiny_mlx():
    mx.random.seed(0)
    args = ModelArgs(model_type="qwen2", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                     num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=2,
                     tie_word_embeddings=True)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_sampler_cache_and_no_cache_agree(tiny_mlx):
    # in fp32 the KV-cache logprobs and the one-pass logprobs should match closely;
    # if score() were off by one position they wouldn't come close
    toks, lps = sampler.sample(tiny_mlx, [3, 1, 4, 1, 5], eos_ids={63}, max_new=30, seed=0)
    again = sampler.score(tiny_mlx, [3, 1, 4, 1, 5], toks)
    assert len(again) == len(toks) == len(lps)
    assert np.allclose(lps, again, atol=1e-4)


def test_sampler_repeats_itself_with_same_seed(tiny_mlx):
    a = sampler.sample(tiny_mlx, [2, 7, 1], eos_ids=set(), max_new=20, seed=3)
    b = sampler.sample(tiny_mlx, [2, 7, 1], eos_ids=set(), max_new=20, seed=3)
    c = sampler.sample(tiny_mlx, [2, 7, 1], eos_ids=set(), max_new=20, seed=4)
    assert a == b
    assert a[0] != c[0]


def test_sampler_stops_at_eos(tiny_mlx):
    toks, _ = sampler.sample(tiny_mlx, [2, 7, 1], eos_ids=set(range(64)), max_new=20, seed=0)
    assert len(toks) == 1


def test_trainer_left_padded_batch_matches_single(tiny_torch):
    seqs = [([1, 2, 3], [4, 5, 6, 7, 8, 9]), ([10, 11], [12]), ([13, 14, 15, 16, 17], [18, 19])]
    single = [trainer.score(tiny_torch, [s], pad_id=0)[0] for s in seqs]
    batch = trainer.score(tiny_torch, seqs, pad_id=0, left_pad=True)
    for a, b in zip(single, batch):
        assert a == pytest.approx(b, abs=1e-5)


def test_trainer_fp32_head_is_a_no_op_for_an_fp32_model(tiny_torch):
    seq = ([1, 2, 3], [4, 5, 6])
    w32 = tiny_torch.lm_head.weight.float()
    assert trainer.score(tiny_torch, [seq], w32=w32)[0] == pytest.approx(trainer.score(tiny_torch, [seq])[0], abs=1e-6)


def test_replay_gives_back_the_sampled_logprobs(tiny_mlx):
    toks, lps = sampler.sample(tiny_mlx, [3, 1, 4], eos_ids={63}, max_new=25, seed=1)
    assert sampler.replay(tiny_mlx, [3, 1, 4], toks) == lps
    w32 = sampler.head_fp32(tiny_mlx)
    assert np.allclose(sampler.replay(tiny_mlx, [3, 1, 4], toks, w32), lps, atol=1e-5)
    assert np.allclose(sampler.score(tiny_mlx, [3, 1, 4], toks, w32), lps, atol=1e-4)
