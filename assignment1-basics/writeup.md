2.1 The Unicode Standard
(a) \x00
(b) __repr__() should enable object reconstruction
(c) the return of chr(0) can't be displayed in the printf

2.2 Unicode Encodings
(a) UTF-16 and UTF-32 separately take 16 and 32 bytes, which means they consume more storage space. Moreover, they would occur the problem about Endian.
(b) 你好啊；As UTF-8 is only play an effective role in the ASCII code
(c) \xef\xbf\xbf

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
