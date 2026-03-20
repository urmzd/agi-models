# RegisterGPT

A language model where **each register is a word**.

## The Idea

Standard transformers map tokens into opaque embedding spaces. RegisterGPT keeps computation in vocabulary space the entire time:

```
Input:  one-hot("cat") → R["cat"] = 1.0, all else 0.0
State:  always a distribution over words
Output: register state IS the prediction — R["dog"]=0.3, R["mat"]=0.25
```

No embedding matrix. No output projection. Every intermediate state is readable as "which words are active and how strongly."

## Architecture Iterations

| Version | Cross-position mixing | Within-position | Params | Best val_bpb | Status |
|---------|----------------------|-----------------|--------|-------------|--------|
| [v0](v0_register_lm/) | Shared attention | Fourier ops | 485K | — | Prototype (uses learned embeddings) |
| [v1](v1_shared_attention/) | Shared attention | Fourier ops | 3.2M | **2.83** | Best so far |
| [v2](v2_causal_conv/) | Depthwise causal conv | Fourier ops | 1.3M | — | Abandoned (slow, barely learned) |
| [v3](v3_assoc_memory/) | Associative memory | Fourier ops | 328K–1.7M | — | In progress |
| [v4](v4_param_optimized/) | Assoc memory (shared Q/K) | Factored ops | ~101K | — | Design only |
| [v_gauss](v_gauss/) | FFT-based assoc memory | FFT register ops | — | — | Design only |

Active development is in `model.py` / `train.py` at the root.

## Usage

**GPU** (requires [parameter-golf](https://github.com/openai/parameter-golf) data in `./data/`):
```bash
torchrun --standalone --nproc_per_node=1 train.py
```

All hyperparameters are configurable via environment variables. See [AGENTS.md](AGENTS.md) for the full list.

## Project Structure

```
model.py / train.py           # Current iteration (v3)
v0_register_lm/               # Original prototype with learned embeddings
v1_shared_attention/           # Shared attention (best results)
v2_causal_conv/                # Depthwise conv (abandoned)
v3_assoc_memory/               # Associative memory (in progress)
v4_param_optimized/            # Param-optimized design (untrained)
v_gauss/                       # FFT-based design (untrained)
docs/                          # Research notes and design docs
```

## Context

Built for [OpenAI Parameter Golf](https://github.com/openai/parameter-golf). Inspired by [linear-gp](https://github.com/urmzd/linear-gp) — where complex behavior emerges from sequential execution of cheap operations on a narrow register bank.
