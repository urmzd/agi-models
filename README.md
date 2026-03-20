# AGI Models

Exploring what computation *actually is* — not copying the brain's structure, but finding the simplest mathematical substrate that produces intelligence.

## Philosophy

### 1. The medium is not the message
Everyone in ML copies the brain's **structure**: neurons → hidden units, synapses → weights, cortical layers → transformer layers. This is like trying to fly by building mechanical feathers. We don't copy structure — we look for the **underlying dynamics** that produce intelligent behavior, regardless of substrate.

### 2. Registers are words
Standard transformers map tokens into opaque embedding spaces where intermediate states are uninterpretable. Our models keep computation in **vocabulary space** — every intermediate state is readable as "which words are active and how strongly." Interpretability by construction, not by post-hoc analysis.

```
Input:  one-hot("cat") → R["cat"] = 1.0, all else 0.0
State:  always a distribution over words
Output: register state IS the prediction — no output projection needed
```

### 3. Simple math, composed deeply
Dot products, outer products, dense projections, relu. If the math didn't exist before 1980, we probably don't need it. The power comes from **composition and scale**, not mathematical complexity. This is why attention works — it's just a weighted average. And it's why Fourier projections failed us — too clever, not enough capacity.

### 4. Meta-learning over memorization
The trained weights define **how to learn**. The runtime state (Q-table, associative memory, policy) stores **what was learned** from the current sequence. The model learns during inference — like a Q-table in reinforcement learning that starts empty and fills up through experience.

### 5. Policy over lookup
Instead of memorizing every possible relationship (like attention's Q·K^T over all positions), learn a compact **policy** that decides what to do given the current state. The same mechanism can execute different operations on different inputs — data-dependent branching, not fixed computation.

## What we've learned

**Fourier projections don't work for cross-position mixing.** v3, v5, v6 all used Fourier-parameterized projections (rank-32 bottleneck) for Q/K/V. All produced flat loss. The bottleneck can't capture the complexity of word relationships.

**Dense projections work.** v1 (attention with dense matrices) and v9 (Q-table with dense projections) both learn. The key ingredient is full-rank learned projection matrices.

**Phase transitions are real.** v9 plateaued for 150 steps then loss dropped sharply from 6.30 → 5.60 in 80 steps. Don't kill runs during plateaus — the model may be coordinating internal representations before a breakthrough.

**101K params can learn language structure.** v4 reached val_bpb 3.65 with only 101K parameters (419KB compressed). Not competitive yet, but proves the architectural ideas have merit at extreme compression.

**Hard routing beats soft blending.** Soft mixtures of operations produce average behavior. Hard winner-take-all routing (Gumbel-softmax) produces conditional behavior — "if state looks like X, do op 3" — which is a fundamentally different computational primitive and compresses better (near-one-hot weights eat INT8+zlib for breakfast).

**Language needs multiple temporal scales.** A single decay rate forces the model to choose one timescale. Multiple Q-tables with different decays (fast γ≈0.5, medium γ≈0.95, slow γ≈0.99) capture bigram, paragraph, and document-level patterns simultaneously.

## Architecture Iterations

| Version | Name | Cross-position | Within-position | Params | val_bpb | Status |
|---------|------|---------------|-----------------|--------|---------|--------|
| [v0](v0_register_lm/) | Register LM | Shared attention | Fourier ops | 485K | — | Prototype |
| [v1](v1_shared_attention/) | Shared Attention | Shared attention | Fourier ops | 3.2M | **2.83** | Best bpb |
| [v2](v2_causal_conv/) | Causal Conv | Depthwise conv | Fourier ops | 1.3M | — | Abandoned |
| [v3](v3_assoc_memory/) | Assoc Memory | Assoc memory (Fourier) | Fourier ops | 328K–1.7M | ~3.9 | Fourier bottleneck |
| [v4](v4_param_optimized/) | Param Golf | Assoc memory (shared Q/K) | Factored ops | 101K | 3.65 | Smallest |
| [v5](v5_gauss_fft/) | Gauss FFT | FFT-based assoc memory | FFT ops | 919K | ~4.1 | Flat loss |
| [v6](v6_brain_wave/) | Brain Wave | Oscillatory coupling | Band-specific ops | 824K | ~3.7 | Flat loss |
| [v7](v7_lgp/) | LGP | Causal decay memory | Learned program (op bank) | — | — | Ready |
| [v8](v8_word_graph/) | Word Graph | Word activation similarity | V×V interaction graph | — | — | Ready |
| **[v9](v9_meta_state/)** | **Meta-State** | **Evolving Q-table (dense)** | **Dense MLP** | **4.2M** | **3.32** | **Best non-attn** |
| [v10](v10_policy/) | Policy | Causal decay + policy | State-dependent ops | — | — | Ready |
| [v11a](v11_brainwave/) | BrainWave v2 | EMA + causal decay | 5 oscillatory primitives | — | — | Ready |
| [v11b](v11_tpg/) | Neural TPG | Multi-scale Q-table (3 decays) | Hard Gumbel routing | 6.4M | — | Ready |
| [v12](v12_sparse_register/) | Sparse Register | Causal decay (k-subspace) | MLP in k-subspace | — | — | Ready |

### Evolution of ideas

```
v0-v1:  Can registers = words?              → Yes, with attention (v1 best bpb)
v2:     Can convolutions replace attention?  → No (too slow, no cross-word mixing)
v3-v6:  Can Fourier projections replace      → No (rank bottleneck kills learning)
        dense matrices?
v4:     How small can we go?                 → 101K params, 419KB, val_bpb 3.65
v7:     Can we learn a program?              → Op bank + soft register addressing
v8:     Direct word-to-word graph?           → V×V interaction, multi-hop reasoning
v9:     Q-table with dense projections?      → Yes! Phase transition, val_bpb 3.32
v10:    Policy instead of lookup?            → State-dependent READ→OP→WRITE
v11a:   Oscillatory primitives?              → 5 brain wave ops, sequential composition
v11b:   Neural TPG (hard routing + depth)?   → Gumbel routing, 3 timescales, early exit
v12:    Sparse register addressing?          → Top-k subspace ops, full-rank in subspace
```

## Quick Start

```bash
# One-command setup on RunPod
curl -sSL https://raw.githubusercontent.com/urmzd/agi-models/main/bootstrap.sh | bash

# Or manually
pip install huggingface_hub sentencepiece
python data/download_data.py --variant sp1024

# Train (pick a model)
MODEL_VERSION=meta      torchrun --standalone --nproc_per_node=1 train.py  # v9 Q-table
MODEL_VERSION=tpg       torchrun --standalone --nproc_per_node=1 train.py  # v11b neural TPG
MODEL_VERSION=sparse    torchrun --standalone --nproc_per_node=1 train.py  # v12 sparse register
MODEL_VERSION=policy    torchrun --standalone --nproc_per_node=1 train.py  # v10 policy
MODEL_VERSION=v4        torchrun --standalone --nproc_per_node=1 train.py  # 101K params

# Benchmark all models (no GPU required)
python benchmark.py

# Run all models sequentially and collect results
python run_all.py
```

All hyperparameters configurable via env vars. See [AGENTS.md](AGENTS.md).

## Project Structure

```
train.py                       # Unified training script (all models)
benchmark.py                   # Rapid model benchmarking (synthetic data, no GPU)
run_all.py                     # Train all models sequentially, collect results
results.py                     # Collect manifest.json files into results table
data/download_data.py          # Data download (FineWeb sp1024)
bootstrap.sh                   # One-command RunPod setup
setup.sh                       # Manual setup script
v0_register_lm/               # Prototype (learned embeddings)
v1_shared_attention/           # Shared attention (best bpb)
v2_causal_conv/                # Depthwise conv (abandoned)
v3_assoc_memory/               # Associative memory (Fourier — bottlenecked)
v4_param_optimized/            # Param-optimized (101K params)
v5_gauss_fft/                  # FFT-based (flat loss)
v6_brain_wave/                 # Oscillatory dynamics (flat loss)
v7_lgp/                        # True LGP (op bank + soft addressing)
v8_word_graph/                 # Direct word-to-word graph
v9_meta_state/                 # Q-table meta-state (best non-attention)
v10_policy/                    # State-dependent policy execution
v11_brainwave/                 # Oscillatory primitives (delta/theta/alpha/beta/gamma)
v11_tpg/                       # Neural TPG (hard routing, multi-timescale, adaptive depth)
v12_sparse_register/           # Sparse register addressing (top-k subspace)
docs/                          # Research notes and design docs
```

## Inspirations

- [Linear Genetic Programming](https://github.com/urmzd/linear-gp) — register machines, Q-tables, sequential cheap operations
- [Tangled Program Graphs](https://web.cs.dal.ca/~mheywood/) (Heywood) — hard bidding, multi-timescale memory, input-dependent depth
- Reinforcement learning — Q-tables as meta-learning, policy over lookup
- Hopfield networks (1982) — associative memory via outer products
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — constraints that force architectural innovation
- PonderNet (Banino et al. 2021) — adaptive computation time, halting probability
