2.1 The Unicode Standard
(a) \x00
(b) __repr__() should enable object reconstruction
(c) the return of chr(0) can't be displayed in the printf

2.2 Unicode Encodings
(a) UTF-16 and UTF-32 separately take 16 and 32 bytes, which means they consume more storage space. Moreover, they would occur the problem about Endian.
(b) 你好啊；As UTF-8 is only play an effective role in the ASCII code
(c) \xef\xbf\xbf

2.5 Tokenizer Experiments

(a) BPE training. TinyStories: vocab 10000 (9743 merges on top of the 256 byte tokens and the special token); the longest learned token is ' accomplishment' (15 bytes) - an ordinary English word, as expected from a clean children's-story corpus. OpenWebText: vocab 32000 (31743 merges, ~26 minutes to train); the longest learned token is a 64-byte run of repeated '\xc3\x83\xc3\x82' ('ÃÂ' mojibake) - OWT contains enough mis-encoded junk text that the trainer spends merges on it. Training cost grows with corpus size and number of merges, so the 10k-vocab TinyStories run (2.2 GB corpus) is much cheaper than the 32k-vocab OWT run (11.9 GB corpus).

(b) Compression ratios in UTF-8 bytes per token. In-domain numbers are exact (raw file bytes / token count of the full tokenized corpus); cross-domain numbers are measured on a 32 KiB sample:
- TinyStories with the TinyStories tokenizer: 2,227,753,162 B / 541,436,658 tokens = 4.11 B/token
- OpenWebText with the OWT tokenizer: 11,920,511,059 B / 2,750,839,571 tokens = 4.33 B/token
- TinyStories text with the OWT tokenizer: 3.99 B/token (~3% worse than in-domain)
- OpenWebText text with the TinyStories tokenizer: 3.18 B/token (~32% worse than the OWT tokenizer on the same text)
The OWT tokenizer transfers well to clean children's text (its 32k vocabulary covers common English), but the 10k TinyStories tokenizer degrades badly on diverse web text - both because its vocabulary is 3x smaller and because it never saw web-style content (markup, numbers, punctuation patterns) during training. Moral: train the tokenizer on (a superset of) the deployment domain.

(c) Throughput. Single-process encoding runs at ~0.5-0.6 MB/s. With 12 parallel workers we tokenized the 11.9 GB OWT corpus in 1390 s, an aggregate of ~8.6 MB/s. At that aggregate rate, tokenizing the Pile (825 GiB) would take ~29 hours; a single process would need ~17 days. (The memory-critical step of BPE training is counting word frequencies over the corpus, which is why our trainer streams chunk-by-chunk instead of loading everything.)

3.5 Transformer Accounting
(a) P = 2*vocab_size*d_model + num_layers*(4*d_model^2 + 3*d_model*d_ff + 2*d_model) + d_model = 1,640,452,800, about 1.64B trainable parameters. Transformer blocks hold about 90% of them. Loading the model in single precision takes 4 bytes per parameter, about 6.6 GB.
(b) The matrix multiplies are: per block 3 QKV projections (6*L*d_model^2), QK^T and the weighted sum of values (2*L^2*d_model each), output projection (2*L*d_model^2), and the SwiGLU FFN (6*L*d_model*d_ff); plus the LM head (2*L*d_model*vocab_size). With L=1024: FFN 2.02e12 (57.5%), attention projections 1.01e12 (28.6%), attention score/value matmuls 3.22e11 (9.2%), LM head 1.65e11 (4.7%), total about 3.52e12 FLOPs (3.52 TFLOPs) per forward pass.
(c) The FFN dominates (57.5%), followed by the attention linear projections (28.6%); the quadratic attention terms are only 9.2% at context length 1024.
(d) GPT-2 small (12L/768d): FFN 39.8%, attention projections 19.9%, attention matmuls 13.3%, LM head 27.1%; medium (24L/1024d): 50.1% / 24.8% / 12.4% / 12.7%; large (36L/1280d): 54.3% / 27.3% / 10.9% / 7.4%; XL (48L/1600d): 57.5% / 28.6% / 9.2% / 4.7%. As the model grows, the block FLOPs (~num_layers*d_model^2) grow much faster than the LM head (~d_model), so the LM head share shrinks from 27.1% to 4.7% while the FFN and attention projection shares rise; at fixed context length the quadratic attention term slowly declines in share since it only grows linearly with d_model.
(e) With context length 16384 the total forward FLOPs grow to about 1.34e14, roughly 38x the original (the L^2 attention terms grow 256x while the per-token terms grow 16x). The attention score/value matmuls jump from 9.2% to about 62% of the total and become the dominant component, while the FFN share falls to about 24%.

4.2 Learning Rate Tuning
Running the SGD toy example for 10 iterations: with lr=1e1 the loss decays faster than with lr=1 (26.3 -> 3.5 vs 26.3 -> 21.7). With lr=1e2 the loss first stays flat for one step (the update overshoots and flips the sign of the weights) and then drops rapidly to ~0, i.e. it oscillates but still converges on this simple quadratic. With lr=1e3 the loss diverges, growing by roughly two orders of magnitude per step (26.3 -> 2.4e18 within 10 steps).

4.3 AdamW Accounting
(a) Let P = 2*vocab_size*d_model + num_layers*(12*d_model^2 + 2*d_model) + d_model be the number of trainable parameters (using d_ff = 8/3*d_model so 3*d_model*d_ff = 8*d_model^2), and B = batch_size. In float32:
- Parameters: 4P bytes
- Gradients: 4P bytes
- Optimizer state (first and second moments): 8P bytes
- Activations (one saved tensor per listed component): per block 8*B*context_length*d_model (2 RMSNorms, QKV projections, weighted sum of values, output projection) + 2*B*num_heads*context_length^2 (QK^T scores and softmax) + 4*B*context_length*d_ff (W1, SiLU gate, element-wise product, W3); plus B*context_length*d_model (final RMSNorm) and 2*B*context_length*vocab_size (logits and log-probabilities for cross-entropy).
- Peak memory = 16P + 4B*[num_layers*(56/3*context_length*d_model + 2*num_heads*context_length^2) + context_length*d_model + 2*context_length*vocab_size] bytes.

(b) For the GPT-2 XL shape, P = 1,640,452,800, so the static part is 16P = 26.2 GB. The activation term is 4*[48*(56/3*1024*1600 + 2*25*1024^2) + 1024*1600 + 2*1024*50257] = 16.4 GB per batch element. Peak memory = 16.4*batch_size + 26.2 GB. Solving for 80 GB gives batch_size <= (80-26.2)/16.4 = 3.3, so the maximum batch size is 3.

(c) One AdamW step performs only element-wise operations over the P parameters: the decoupled weight-decay update, the two exponential moving average updates for m and v, and the moment-adjusted update (square root, division, multiply-add), i.e. roughly a dozen FLOPs per parameter, about 12P = 2e10 FLOPs in total. This is independent of batch size and negligible compared to the ~6P*batch_size*context_length = 1e16 FLOPs of the forward and backward passes.

(d) Forward+backward FLOPs per step = 3 * 3.52e12 * 1024 = 1.08e16 (backward = 2x forward, batch_size=1024). Over 400K steps this is 4.32e21 FLOPs. At 50% MFU on an H100 the effective throughput is 495e12 * 0.5 = 2.475e14 FLOP/s, so training takes 4.32e21 / 2.475e14 = 1.75e7 seconds, about 4850 hours (roughly 202 days) on a single H100.

4.4 Learning Rate Sweep

Setup: TinyStories baseline (vocab 10000, d_model=512, 4 layers, 16 heads, d_ff=1344, context 256; 22.7M params), batch 128, 10000 iters = 327.68M tokens, cosine schedule (min_lr = peak/10, 300 warmup iters), bf16. We swept the peak learning rate over {1e-3, 3e-3, 1e-2, 3e-2}:

| peak lr  | best valid loss |
|----------|-----------------|
| 1e-3     | 1.3408          |
| 3e-3     | **1.3163**      |
| 1e-2     | 3.1708          |
| 3e-2     | 3.9908          |

3e-3 is the best of the sweep. The two large learning rates show the same instability as the SGD toy example (4.2): the gradient norm spikes by an order of magnitude and the loss stalls on a high plateau (gradient clipping at 1.0 prevents outright divergence). 1e-3 trains stably but undertrains within the fixed token budget. Train/valid loss and grad-norm curves: logs/lr_sweep.png.

4.5 Batch Size

Design: fixed token budget (327.68M tokens) and fixed hyperparameters (peak lr 3e-3), so only the number of optimizer steps and the gradient-noise level change: batch 64 -> 20000 iters, batch 128 -> 10000 iters (baseline), batch 256 -> 5000 iters.

Results:

| batch size | iters | best valid loss | training behavior |
|------------|-------|-----------------|-------------------|
| 64         | 20000 | 2.2490          | unstable: grad-norm spikes up to ~131; best valid reached mid-run, degrades afterwards |
| 128        | 10000 | **1.3163**      | baseline, stable |
| 256        | 5000  | 1.3285          | extremely stable (grad norm ~0.1 throughout) |

At a fixed peak lr of 3e-3, batch 64 is dramatically worse (+0.93): the noisier gradient estimate makes training unstable at this learning rate - late-training gradient spikes knock the weights around and the valid loss climbs back from its mid-run best. The remedy would be a smaller lr, i.e. small batches and large learning rates do not mix. Batch 256 nearly matches the baseline (+0.012) with half as many optimizer steps and perfectly stable gradients; it is slightly worse because the model takes fewer update steps at the same lr - larger batches typically want a larger lr (the well-known linear-scaling heuristic) or more steps. Wall-clock time is similar for all three (~30 min): the 22.7M model is too small to saturate the GPU at any of these batch sizes, so the throughput difference (~1.8-2.0e5 tok/s) is minor. Overall batch 128 is the sweet spot at lr 3e-3, and the experiment shows that batch size and learning rate must be tuned together, not independently.

5. Ablation Experiments

All ablations use the TinyStories baseline above (lr 3e-3, 327.68M tokens); exactly one component is changed per run:

| variant                                | best valid loss | vs baseline |
|----------------------------------------|-----------------|-------------|
| baseline                               | 1.3163          | -           |
| no layer norm (all RMSNorm removed)    | 9.21 (NaN)      | diverged    |
| no positional embedding (RoPE removed) | 1.3898          | +0.074      |
| post-norm instead of pre-norm          | 1.3514          | +0.035      |
| plain SiLU FFN (d_ff=4*d_model)        | 1.3193          | +0.003      |

(a) Removing all RMSNorm is catastrophic: training diverges to NaN partway through and the best valid loss stays at the random-init value ln(10000) ~= 9.21. Normalization is what keeps activations and gradients bounded at this learning rate.
(b) NoPE degrades but does not collapse: causal attention already leaks some positional information (a position can only attend leftward), but the model loses precise position awareness - the largest degradation among the trainable variants.
(c) Post-norm still trains at 4 layers but is slightly worse; the gap is known to grow with depth, which is why modern deep LMs use pre-norm.
(d) SwiGLU vs a parameter-matched plain SiLU FFN (hidden dim 4*d_model, two matrices, ~8*d_model^2 params like SwiGLU) is a wash at this scale and budget: +0.003, within noise.

6. Decoding

Samples from the TinyStories checkpoint (best valid 1.3163), prompt "Once upon a time, there was a little girl named Lily.", 200 new tokens:

- Greedy (T=0): fluent start, but degenerates into repetition ("a big, scary dog" appears twice almost verbatim) and ends abruptly.
- T=0.8, top-p 0.9: a complete, coherent story arc (playing outside -> thunder -> hiding -> happy ending). Best of the three.
- T=1.2, no truncation: grammar and semantics break down ("her room became black", an incoherent preachy "moral of the story").

Temperature divides the logits before softmax: T<1 sharpens the distribution (more conservative), T>1 flattens it (more random), while the ranking of tokens is unchanged; top-p then truncates the low-probability tail. Greedy decoding is deterministic and tends to loop; a moderate temperature with nucleus truncation balances diversity against coherence.

7. Main Experiment: OpenWebText

Tokenizer: BPE trained on owt_train.txt (vocab 32000, 31743 merges, ~26 minutes). The longest learned token (id 25822) is a 64-byte run of repeated 'ÃÂ' mojibake - the tokenizer had to spend vocabulary on junk byte sequences in OWT. The corpus was tokenized into uint16 memmaps: 2,750,839,571 train / 66,923,654 valid tokens.

Training: same architecture with vocab 32000 (45.2M params), 10000 iters = 327.68M tokens, i.e. only ~12% of one epoch. Best valid loss 4.6941, still decreasing at the end of the run - the model is clearly undertrained. This loss is not directly comparable with the TinyStories 1.3163: the vocabulary (32k vs 10k) and the corpus (diverse web text vs children's stories) both differ, and OWT is intrinsically higher-entropy.

Generation (prompt "The history of the internet began", T=0.8, top-p 0.9): the model emits fluent news-style prose full of figures and quotes ("...said Bargel, a federal analyst at the University of Pennsylvania..."), i.e. it has picked up the OWT register, but the content is fabricated and self-contradictory - as expected from a 45M model that has seen only 12% of its training data. A multi-epoch run is the obvious next step.
