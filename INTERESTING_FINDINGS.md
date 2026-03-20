# Interesting Findings

## Benchmark: CPU Forward+Backward (synthetic data, no training)

Run via `python benchmark.py --iters 3 --batch 2 --seq-len 128`.

| Model | Params | Raw MB | ~Int8 MB | Fwd ms | Bwd ms | Total ms | tok/s | Init Loss | AvgGrad | MaxGrad | Dead |
|-------|--------|--------|----------|--------|--------|----------|-------|-----------|---------|---------|------|
| gauss | 919K | 3.67 | 0.92 | 34 | 57 | 91 | 7556 | 7.52 | 2.02e+00 | 1.99e+01 | 0 |
| v3 | 329K | 1.31 | 0.33 | 648 | 572 | 1221 | 395 | 7.54 | 1.85e+00 | 1.58e+01 | 0 |
| lgp | 132K | 0.53 | 0.13 | 158 | 218 | 376 | 1618 | 7.59 | 2.04e+00 | 2.51e+01 | 0 |
| graph | 1081K | 4.33 | 1.08 | 215 | 327 | 541 | 1193 | 13.55 | 2.86e+00 | 1.00e+02 | 0 |
| v4 | 102K | 0.41 | 0.10 | 801 | 708 | 1509 | 319 | 21.57 | 5.60e+00 | 5.10e+01 | 0 |
| meta | 4195K | 16.78 | 4.20 | 16 | 30 | 46 | 15716 | 23.15 | 3.79e-01 | 1.25e+01 | 0 |
| v2 | 353K | 1.41 | 0.35 | 627 | 1105 | 1732 | 408 | 23.40 | 2.08e-01 | 1.22e+01 | 0 |
| v1 | 3360K | 13.44 | 3.36 | 263 | 4729 | 4992 | 973 | 23.44 | 3.21e-01 | 1.20e+01 | 5 |
| policy | 1387K | 5.55 | 1.39 | 15 | 25 | 40 | 17026 | 23.63 | 1.15e-01 | 1.21e+01 | 0 |
| wave | 824K | 3.30 | 0.82 | 1471 | 1378 | 2848 | 174 | 23.63 | 4.30e-02 | 1.21e+01 | 0 |

### Observations

**Dense projections are 30-70x faster than Fourier-based models on CPU.** `meta` (46ms) and `policy` (40ms) use dense `nn.Linear` projections and are dramatically faster than Fourier-parameterized models like v3 (1221ms), v4 (1509ms), and wave (2848ms) — despite having more parameters. The Fourier basis matmuls and softmax-over-vocab operations are expensive when vocab=dim.

**Fourier models have better initialization loss.** v3 (7.54), gauss (7.52), and lgp (7.59) start with much lower loss on random data than dense models (meta 23.15, policy 23.63). This suggests the Fourier parameterization provides useful inductive bias at init — the structured basis gives the model a head start. Whether this translates to better trained performance is a separate question (it didn't for v3/v5/v6 due to rank bottleneck).

**gauss is the speed/quality sweet spot among Fourier models.** It achieves the best init loss (7.52) while being 13x faster than v3 and 4x faster than lgp. The FFT-based approach avoids the explicit Fourier basis matmul overhead.

**v1 has 5 dead parameters (zero gradients).** This may indicate unused capacity or initialization issues in the shared attention mechanism.

**v4 is the smallest (102K params) but the second slowest.** Its factored ops and shared Q/K projections save parameters but add computational overhead from the multi-invocation loop.

**graph model has the highest max gradient (100).** This suggests potential training instability — the V×V interaction graph may need gradient clipping or careful learning rate tuning.

**wave is the slowest model by far (2848ms).** The multi-band oscillatory coupling with separate slow/mid/fast Fourier bases compounds the cost of Fourier operations.
