# TODO

## Infrastructure
- [ ] MLX support for current models (v3/v4/v5) — only v0 has an MLX training script
- [ ] Single training script that detects PyTorch CUDA vs MLX and imports accordingly

## Architectures
- [ ] v6 brain wave — implement oscillatory dynamics model
- [ ] True LGP — evolve discrete instruction sequences, not just gradient descent
- [ ] Word interaction graph — sparse learned word-to-word interaction matrix

## Training
- [x] Checkpoint save/resume
- [x] Roundtrip eval optional (ROUNDTRIP_EVAL=1)
- [ ] Learning rate warmup schedule (currently flat after warmup steps)
- [ ] Wandb/tensorboard logging
