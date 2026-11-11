"""The "trainer": PyTorch on MPS, teacher-forced over the whole sequence."""

import torch
from transformers import AutoModelForCausalLM

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_model(model_id, dtype, attn="sdpa"):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPES[dtype], attn_implementation=attn)
    return model.to("mps").eval()


@torch.no_grad()
def score(model, seqs, pad_id=0, left_pad=False, w32=None):
    """Logprobs of the answer tokens for a batch of (prompt, tokens) pairs.

    One pair is a plain forward pass with no mask. Several are padded and
    masked into one batch, on the right by default. Left padding passes
    explicit positions so each sequence still starts at position 0.
    With w32 (lm_head.weight in fp32) the output layer runs in fp32.
    """
    device = next(model.parameters()).device
    full = [p + t[:-1] for p, t in seqs]
    width = max(len(f) for f in full)
    ids = torch.full((len(full), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(full), width), dtype=torch.long)
    offset = [width - len(f) if left_pad else 0 for f in full]
    for i, f in enumerate(full):
        ids[i, offset[i] : offset[i] + len(f)] = torch.tensor(f)
        mask[i, offset[i] : offset[i] + len(f)] = 1
    kwargs = {}
    if len(seqs) > 1:
        kwargs["attention_mask"] = mask.to(device)
    if left_pad:
        kwargs["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0).to(device)
    h = model.model(input_ids=ids.to(device), **kwargs).last_hidden_state

    out = []
    for i, (p, t) in enumerate(seqs):
        # position len(p)-1 is the last prompt token, which predicts t[0]
        rows = h[i, offset[i] + len(p) - 1 : offset[i] + len(p) - 1 + len(t)]
        logits = (rows.float() @ w32.T) if w32 is not None else model.lm_head(rows).float()
        lp = torch.log_softmax(logits, dim=-1)
        picked = lp.gather(-1, torch.tensor(t, device=device)[:, None])[:, 0]
        out.append(picked.cpu().double().numpy())
    return out
