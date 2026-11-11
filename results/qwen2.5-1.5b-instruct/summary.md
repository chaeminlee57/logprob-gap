# Qwen/Qwen2.5-1.5B-Instruct, 128 GSM8K prompts, T=1.0, up to 1024 new tokens

## Sampler vs trainer

| sampler | trainer | tokens | mean abs dlogp (95% CI) | p99 | max | max abs dprob | k3 KL | w outside [0.8,1.25] | w outside [0.5,2] | clipped at C=2 |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp32 | 39451 | 3.13e-06 (2.92e-06 to 3.36e-06) | 3.11e-05 | 8.23e-05 | 2.54e-05 | 2.43e-11 | 0 | 0 | 0 |
| bf16 | bf16 | 38798 | 0.0097 (0.0087 to 0.0108) | 0.1189 | 0.3646 | 0.1336 | 3.74e-04 | 8.76e-04 | 0 | 0 |
| fp16 | fp16 | 39403 | 0.0011 (0.0011 to 0.0012) | 0.0145 | 0.0464 | 0.0232 | 5.42e-06 | 0 | 0 | 0 |
| 8-bit | bf16 | 38960 | 0.013 (0.012 to 0.0141) | 0.1658 | 0.7306 | 0.1818 | 6.53e-04 | 0.0035 | 2.57e-05 | 0 |
| 4-bit | bf16 | 37362 | 0.1639 (0.1526 to 0.1755) | 1.7366 | 9.6195 | 0.9943 | 0.0628 | 0.2006 | 0.0726 | 0.0171 |

## Each comparison on its own (mean abs dlogp; these don't add up)

| sampler | sampler vs trainer | sampler vs fp32 trainer | KV cache vs no cache (MLX) | MLX vs PyTorch (no cache) | trainer vs fp32 trainer |
|---|---|---|---|---|---|
| fp32 | 3.13e-06 | 3.13e-06 | 2.65e-06 | 2.66e-06 |  |
| bf16 | 0.0097 | 0.0085 | 0.0093 | 0.0095 | 0.0085 |
| fp16 | 0.0011 | 0.001 | 0.0012 | 0.0011 | 0.001 |
| 8-bit | 0.013 | 0.0117 | 0.0091 | 0.013 | 0.0081 |
| 4-bit | 0.1639 | 0.1634 | 0.0122 | 0.1634 | 0.0112 |

## By token probability (fp32 reference), sampler vs trainer

For the fp32 sampler the trainer side is the fp32 reference itself.

| sampler | bin | share of tokens | mean abs dlogp | mean abs dlogp / mean (1-p) | mean signed log w | share of total abs dlogp | share of w outside [0.5,2] |
|---|---|---|---|---|---|---|---|
| fp32 | <0.01 | 0.0134 | 1.53e-05 | 1.54e-05 | 2.37e-06 | 0.066 | nan |
| fp32 | 0.01-0.1 | 0.0319 | 1.63e-05 | 1.71e-05 | 3.61e-07 | 0.166 | nan |
| fp32 | 0.1-0.5 | 0.0949 | 1.20e-05 | 1.72e-05 | 7.30e-07 | 0.3656 | nan |
| fp32 | >=0.5 | 0.8598 | 1.46e-06 | 2.72e-05 | -2.72e-09 | 0.4025 | nan |
| bf16 | <0.01 | 0.014 | 0.0569 | 0.0571 | 0.0015 | 0.0823 | nan |
| bf16 | 0.01-0.1 | 0.034 | 0.0565 | 0.0594 | -0.0019 | 0.198 | nan |
| bf16 | 0.1-0.5 | 0.0973 | 0.042 | 0.06 | -0.0017 | 0.4209 | nan |
| bf16 | >=0.5 | 0.8547 | 0.0034 | 0.0631 | -4.76e-05 | 0.2988 | nan |
| fp16 | <0.01 | 0.0126 | 0.0071 | 0.0071 | -4.05e-04 | 0.0779 | nan |
| fp16 | 0.01-0.1 | 0.0317 | 0.007 | 0.0074 | 3.32e-04 | 0.1953 | nan |
| fp16 | 0.1-0.5 | 0.0952 | 0.005 | 0.0071 | 6.13e-05 | 0.4162 | nan |
| fp16 | >=0.5 | 0.8604 | 4.13e-04 | 0.0076 | -2.23e-06 | 0.3107 | nan |
| 8-bit | <0.01 | 0.0112 | 0.079 | 0.0793 | -0.002 | 0.0683 | 0 |
| 8-bit | 0.01-0.1 | 0.0297 | 0.0853 | 0.0898 | -0.0092 | 0.1959 | 1 |
| 8-bit | 0.1-0.5 | 0.0906 | 0.0608 | 0.087 | -0.0025 | 0.4254 | 0 |
| 8-bit | >=0.5 | 0.8685 | 0.0046 | 0.0882 | 2.54e-05 | 0.3104 | 0 |
| 4-bit | <0.01 | 0.03 | 1.0184 | 1.0215 | -0.7832 | 0.1866 | 0.2269 |
| 4-bit | 0.01-0.1 | 0.0566 | 0.7863 | 0.8262 | -0.5475 | 0.2714 | 0.3818 |
| 4-bit | 0.1-0.5 | 0.1202 | 0.4397 | 0.6169 | -0.1627 | 0.3224 | 0.3309 |
| 4-bit | >=0.5 | 0.7932 | 0.0454 | 0.7203 | 0.0147 | 0.2196 | 0.0605 |

## Sequence-level log weight (sum of token log ratios)

| sampler | std | mean | seqs with w outside [0.5,2] | corr(abs log w, length) |
|---|---|---|---|---|
| fp32 | 1.34e-04 | 3.40e-05 | 0 | 0.2767 |
| bf16 | 0.5025 | -0.0753 | 0.1641 | 0.394 |
| fp16 | 0.0579 | 0.0029 | 0 | 0.1824 |
| 8-bit | 0.6152 | -0.1527 | 0.1875 | 0.3917 |
| 4-bit | 14.033 | -18.2216 | 0.9844 | 0.6656 |

| sampler | length | seqs | std of log w | std of log w / sqrt(length) | seqs with w outside [0.5,2] |
|---|---|---|---|---|---|
| fp32 | 0-127 | 3 | 3.92e-05 | 3.56e-06 | 0 |
| fp32 | 128-255 | 45 | 7.95e-05 | 5.75e-06 | 0 |
| fp32 | 256-511 | 72 | 1.44e-04 | 7.54e-06 | 0 |
| fp32 | 512-1024 | 8 | 2.18e-04 | 8.82e-06 | 0 |
| bf16 | 0-127 | 3 | 0.1596 | 0.0154 | 0 |
| bf16 | 128-255 | 46 | 0.4068 | 0.0286 | 0.1087 |
| bf16 | 256-511 | 70 | 0.4809 | 0.0269 | 0.1286 |
| bf16 | 512-1024 | 9 | 1.0345 | 0.0411 | 0.7778 |
| fp16 | 0-127 | 3 | 0.0535 | 0.0048 | 0 |
| fp16 | 128-255 | 43 | 0.044 | 0.0031 | 0 |
| fp16 | 256-511 | 75 | 0.0628 | 0.0033 | 0 |
| fp16 | 512-1024 | 7 | 0.0785 | 0.0032 | 0 |
| 8-bit | 0-127 | 6 | 0.3639 | 0.0339 | 0 |
| 8-bit | 128-255 | 43 | 0.4182 | 0.0302 | 0.1163 |
| 8-bit | 256-511 | 68 | 0.6276 | 0.0322 | 0.2059 |
| 8-bit | 512-1024 | 11 | 1.0756 | 0.0439 | 0.4545 |
| 4-bit | 0-127 | 2 | 4.0487 | 0.4035 | 1 |
| 4-bit | 128-255 | 48 | 5.845 | 0.4043 | 0.9792 |
| 4-bit | 256-511 | 72 | 14.238 | 0.708 | 0.9861 |
| 4-bit | 512-1024 | 6 | 18.8922 | 0.7541 | 1 |

## Same tokens, PyTorch bf16 and fp16 against PyTorch fp32

bf16 keeps 8 significant bits, fp16 keeps 11, so pure rounding predicts a ratio near 2^3 = 8.

| rollouts from | bf16 vs fp32 | fp16 vs fp32 | ratio |
|---|---|---|---|
| bf16 | 0.0085 | 0.0011 | 8.11 |
| fp16 | 0.0084 | 0.001 | 8.06 |
| fp32 | 0.0084 | 0.001 | 8.06 |
| 4-bit | 0.0112 | 0.0014 | 7.85 |
| 8-bit | 0.0081 | 9.98e-04 | 8.10 |
| all | 0.0089 | 0.0011 | 8.02 |

## Noise model: each token's log ratio as sigma * (1 - p)

| pair | sigma | lag-1 autocorr | share of sum of squares from p < 0.5 | corr(abs log w, sqrt sum (1-p)^2) | corr(abs log w, length) |
|---|---|---|---|---|---|
| bf16 vs bf16 | 0.0861 | -0.0119 | 0.865 | 0.4853 | 0.394 |
| fp16 vs fp16 | 0.0105 | 0.004 | 0.8603 | 0.3226 | 0.1824 |
| 8-bit vs bf16 | 0.1178 | 3.10e-05 | 0.8623 | 0.4084 | 0.3917 |

| pair | length | seqs | mean surprisal (nats/token, fp32) | mean abs dlogp | std log w / sqrt(length) | std log w / sqrt(sum (1-p)^2) |
|---|---|---|---|---|---|---|
| bf16 vs bf16 | 0-255 | 49 | 0.3386 | 0.0097 | 0.028 | 0.0892 |
| bf16 vs bf16 | 256-511 | 70 | 0.2888 | 0.0082 | 0.0269 | 0.0981 |
| bf16 vs bf16 | 512- | 9 | 0.8785 | 0.0166 | 0.0411 | 0.0932 |
| fp16 vs fp16 | 0-255 | 46 | 0.3136 | 0.0011 | 0.0033 | 0.0118 |
| fp16 vs fp16 | 256-511 | 75 | 0.3417 | 0.0011 | 0.0033 | 0.0107 |
| fp16 vs fp16 | 512- | 7 | 0.6028 | 0.0014 | 0.0032 | 0.009 |
| 8-bit vs bf16 | 0-255 | 49 | 0.3381 | 0.0143 | 0.0303 | 0.112 |
| 8-bit vs bf16 | 256-511 | 68 | 0.2967 | 0.0117 | 0.0322 | 0.1135 |
| 8-bit vs bf16 | 512- | 11 | 0.449 | 0.0149 | 0.0439 | 0.1217 |

## Each path's error against PyTorch fp32

- bf16: sampler (MLX, KV cache) 0.0085, MLX, no cache 0.0083, PyTorch bf16 0.0085; error correlations sampler (MLX, KV cache) / MLX, no cache 0.27, sampler (MLX, KV cache) / PyTorch bf16 0.23, MLX, no cache / PyTorch bf16 0.25
- fp16: sampler (MLX, KV cache) 0.001, MLX, no cache 0.001, PyTorch fp16 0.001; error correlations sampler (MLX, KV cache) / MLX, no cache 0.25, sampler (MLX, KV cache) / PyTorch fp16 0.27, MLX, no cache / PyTorch fp16 0.28

## Output layer in fp32 on both sides (mean abs dlogp unless noted)

| precision | comparison | as run | fp32 output layer |
|---|---|---|---|
| bf16 | MLX vs PyTorch, no cache | 0.0095 | 0.0064 |
| bf16 | trainer vs fp32 trainer | 0.0085 | 0.006 |
| bf16 | sampler vs trainer | 0.0097 | 0.0067 |
| bf16 | sampler vs fp32 trainer | 0.0085 | 0.0059 |
| bf16 | answers_w_outside_0.5_2 | - | 0.0391 |
| bf16 | k3_kl | 3.74e-04 | 1.62e-04 |
| fp16 | MLX vs PyTorch, no cache | 0.0011 | 7.66e-04 |
| fp16 | trainer vs fp32 trainer | 0.001 | 7.50e-04 |

## Controls

- right_padded_batch_of_8: tokens 193974, identical 193974, max_abs 0
- left_padded_batch_of_8: tokens 38798, identical 12119, mean_abs 0.0064, max_abs 0.3992
- eager_attention: tokens 38798, identical 102, mean_abs 0.0816, max_abs 15.7365
- sampler_replay: bf16 {'replayed': 8, 'identical': 8}, fp16 {'replayed': 8, 'identical': 8}

## Rollouts

| sampler | GSM8K accuracy | finished | mean length | repeats itself with same seed | first token differing from fp32 rollout (median) | rollouts identical to fp32 |
|---|---|---|---|---|---|---|
| fp32 | 0.5938 | 0.9922 | 308 | True | - | - |
| bf16 | 0.6562 | 1 | 303 | True | 147.5 | 46 |
| fp16 | 0.5938 | 0.9922 | 308 | True | 263.5 | 106 |
| 8-bit | 0.6562 | 1 | 304 | True | 83.5 | 28 |
| 4-bit | 0.4531 | 1 | 292 | True | 1 | 0 |
