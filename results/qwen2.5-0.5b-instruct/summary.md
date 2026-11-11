# Qwen/Qwen2.5-0.5B-Instruct, 128 GSM8K prompts, T=1.0, up to 1024 new tokens

## Sampler vs trainer

| sampler | trainer | tokens | mean abs dlogp (95% CI) | p99 | max | max abs dprob | k3 KL | w outside [0.8,1.25] | w outside [0.5,2] | clipped at C=2 |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp32 | 44029 | 2.39e-06 (2.23e-06 to 2.55e-06) | 2.34e-05 | 8.77e-05 | 2.28e-05 | 1.37e-11 | 0 | 0 | 0 |
| bf16 | bf16 | 42860 | 0.0123 (0.0112 to 0.0134) | 0.1386 | 0.6567 | 0.1565 | 5.50e-04 | 0.0024 | 0 | 0 |
| fp16 | fp16 | 43325 | 0.0015 (0.0013 to 0.0016) | 0.0167 | 0.0562 | 0.0226 | 7.90e-06 | 0 | 0 | 0 |

## Each comparison on its own (mean abs dlogp; these don't add up)

| sampler | sampler vs trainer | sampler vs fp32 trainer | KV cache vs no cache (MLX) | MLX vs PyTorch (no cache) | trainer vs fp32 trainer |
|---|---|---|---|---|---|
| fp32 | 2.39e-06 | 2.39e-06 | 1.97e-06 | 2.36e-06 |  |
| bf16 | 0.0123 | 0.0102 | 0.0117 | 0.0117 | 0.0099 |
| fp16 | 0.0015 | 0.0013 | 0.0014 | 0.0014 | 0.0013 |

## By token probability (fp32 reference), sampler vs trainer

For the fp32 sampler the trainer side is the fp32 reference itself.

| sampler | bin | share of tokens | mean abs dlogp | mean abs dlogp / mean (1-p) | mean signed log w | share of total abs dlogp | share of w outside [0.5,2] |
|---|---|---|---|---|---|---|---|
| fp32 | <0.01 | 0.0151 | 9.90e-06 | 9.93e-06 | -6.75e-07 | 0.0625 | nan |
| fp32 | 0.01-0.1 | 0.0359 | 1.13e-05 | 1.19e-05 | 6.50e-07 | 0.1705 | nan |
| fp32 | 0.1-0.5 | 0.0954 | 8.89e-06 | 1.26e-05 | 6.89e-07 | 0.3548 | nan |
| fp32 | >=0.5 | 0.8536 | 1.15e-06 | 2.09e-05 | 4.65e-08 | 0.4122 | nan |
| bf16 | <0.01 | 0.0186 | 0.0621 | 0.0623 | -0.0024 | 0.0943 | nan |
| bf16 | 0.01-0.1 | 0.0371 | 0.0682 | 0.0717 | -0.0011 | 0.2061 | nan |
| bf16 | 0.1-0.5 | 0.0989 | 0.0509 | 0.0721 | -0.0015 | 0.4102 | nan |
| bf16 | >=0.5 | 0.8454 | 0.0042 | 0.0758 | -4.65e-05 | 0.2895 | nan |
| fp16 | <0.01 | 0.0152 | 0.0084 | 0.0084 | -2.48e-04 | 0.0874 | nan |
| fp16 | 0.01-0.1 | 0.0352 | 0.0083 | 0.0088 | 2.44e-04 | 0.2008 | nan |
| fp16 | 0.1-0.5 | 0.0942 | 0.0063 | 0.009 | -9.39e-05 | 0.4092 | nan |
| fp16 | >=0.5 | 0.8553 | 5.17e-04 | 0.0095 | 5.35e-06 | 0.3026 | nan |

## Sequence-level log weight (sum of token log ratios)

| sampler | std | mean | seqs with w outside [0.5,2] | corr(abs log w, length) |
|---|---|---|---|---|
| fp32 | 1.20e-04 | 4.08e-05 | 0 | 0.6184 |
| bf16 | 0.5713 | -0.0928 | 0.1875 | 0.3495 |
| fp16 | 0.0691 | 1.79e-04 | 0 | 0.3027 |

| sampler | length | seqs | std of log w | std of log w / sqrt(length) | seqs with w outside [0.5,2] |
|---|---|---|---|---|---|
| fp32 | 128-255 | 39 | 6.81e-05 | 4.71e-06 | 0 |
| fp32 | 256-511 | 76 | 1.06e-04 | 5.35e-06 | 0 |
| fp32 | 512-1024 | 13 | 2.18e-04 | 7.46e-06 | 0 |
| bf16 | 128-255 | 39 | 0.5313 | 0.0363 | 0.1538 |
| bf16 | 256-511 | 74 | 0.4462 | 0.0243 | 0.1486 |
| bf16 | 512-1024 | 15 | 1.047 | 0.0413 | 0.4667 |
| fp16 | 128-255 | 39 | 0.0432 | 0.003 | 0 |
| fp16 | 256-511 | 77 | 0.0764 | 0.004 | 0 |
| fp16 | 512-1024 | 12 | 0.0823 | 0.0031 | 0 |

## Same tokens, PyTorch bf16 and fp16 against PyTorch fp32

bf16 keeps 8 significant bits, fp16 keeps 11, so pure rounding predicts a ratio near 2^3 = 8.

| rollouts from | bf16 vs fp32 | fp16 vs fp32 | ratio |
|---|---|---|---|
| bf16 | 0.0099 | 0.0013 | 7.55 |
| fp16 | 0.0097 | 0.0013 | 7.74 |
| fp32 | 0.0098 | 0.0013 | 7.71 |
| all | 0.0098 | 0.0013 | 7.66 |

## Noise model: each token's log ratio as sigma * (1 - p)

| pair | sigma | lag-1 autocorr | share of sum of squares from p < 0.5 | corr(abs log w, sqrt sum (1-p)^2) | corr(abs log w, length) |
|---|---|---|---|---|---|
| bf16 vs bf16 | 0.1004 | -0.0026 | 0.8783 | 0.3245 | 0.3495 |
| fp16 vs fp16 | 0.0126 | -0.0077 | 0.8627 | 0.3136 | 0.3027 |

| pair | length | seqs | mean surprisal (nats/token, fp32) | mean abs dlogp | std log w / sqrt(length) | std log w / sqrt(sum (1-p)^2) |
|---|---|---|---|---|---|---|
| bf16 vs bf16 | 0-255 | 39 | 0.2571 | 0.0097 | 0.0363 | 0.1347 |
| bf16 vs bf16 | 256-511 | 74 | 0.3396 | 0.011 | 0.0243 | 0.0876 |
| bf16 vs bf16 | 512- | 15 | 0.7987 | 0.0174 | 0.0413 | 0.1105 |
| fp16 vs fp16 | 0-255 | 39 | 0.2385 | 0.0011 | 0.003 | 0.0104 |
| fp16 vs fp16 | 256-511 | 77 | 0.3738 | 0.0014 | 0.004 | 0.013 |
| fp16 vs fp16 | 512- | 12 | 0.5711 | 0.0018 | 0.0031 | 0.0088 |

## Each path's error against PyTorch fp32

- bf16: sampler (MLX, KV cache) 0.0102, MLX, no cache 0.0099, PyTorch bf16 0.0099; error correlations sampler (MLX, KV cache) / MLX, no cache 0.28, sampler (MLX, KV cache) / PyTorch bf16 0.18, MLX, no cache / PyTorch bf16 0.23
- fp16: sampler (MLX, KV cache) 0.0013, MLX, no cache 0.0013, PyTorch fp16 0.0013; error correlations sampler (MLX, KV cache) / MLX, no cache 0.30, sampler (MLX, KV cache) / PyTorch fp16 0.27, MLX, no cache / PyTorch fp16 0.28

## Output layer in fp32 on both sides (mean abs dlogp unless noted)

| precision | comparison | as run | fp32 output layer |
|---|---|---|---|
| bf16 | MLX vs PyTorch, no cache | 0.0117 | 0.0092 |
| bf16 | trainer vs fp32 trainer | 0.0099 | 0.0079 |
| bf16 | sampler vs trainer | 0.0123 | 0.0098 |
| bf16 | sampler vs fp32 trainer | 0.0102 | 0.0082 |
| bf16 | answers_w_outside_0.5_2 | - | 0.0781 |
| bf16 | k3_kl | 5.50e-04 | 3.41e-04 |
| fp16 | MLX vs PyTorch, no cache | 0.0014 | 0.0011 |
| fp16 | trainer vs fp32 trainer | 0.0013 | 0.001 |

## Controls

- right_padded_batch_of_8: tokens 130214, identical 130214, max_abs 0
- left_padded_batch_of_8: tokens 42860, identical 12342, mean_abs 0.0074, max_abs 0.3739
- eager_attention: tokens 42860, identical 1851, mean_abs 0.0148, max_abs 0.7157
- sampler_replay: bf16 {'replayed': 8, 'identical': 8}, fp16 {'replayed': 8, 'identical': 8}

## Rollouts

| sampler | GSM8K accuracy | finished | mean length | repeats itself with same seed | first token differing from fp32 rollout (median) | rollouts identical to fp32 |
|---|---|---|---|---|---|---|
| fp32 | 0.3438 | 0.9922 | 344 | True | - | - |
| bf16 | 0.3359 | 0.9922 | 335 | True | 142.5 | 36 |
| fp16 | 0.3438 | 1 | 338 | True | 296 | 107 |
