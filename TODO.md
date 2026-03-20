# TODO

## Training runs needed
- [ ] v11b (tpg) — neural TPG with hard routing, multi-timescale memory, adaptive depth
- [ ] v11a (brainwave) — oscillatory primitives
- [ ] v12 (sparse) — sparse register addressing
- [ ] v10 (policy) — state-dependent policy execution
- [ ] v7 (lgp) — differentiable register machine
- [ ] v8 (graph) — word interaction graph

## Infrastructure
- [ ] MLX support for current models — only v0 has an MLX training script
- [ ] Wandb/tensorboard logging
- [ ] Add tpg, brainwave, sparse to run_all.py model list

## Training
- [x] Checkpoint save/resume
- [x] Roundtrip eval optional (ROUNDTRIP_EVAL=1)
- [ ] Learning rate warmup schedule (currently flat after warmup steps)
- [ ] Gumbel temperature annealing for v11b (tpg) — anneal tau from 1.0 → 0.1 during training
