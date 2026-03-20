#!/usr/bin/env python3
"""
Rapid model benchmarking — no CUDA, no real data required.

Instantiates every model variant, runs forward+backward with synthetic tokens,
and reports: param count, memory, throughput, loss, gradient health.

Usage:
    python benchmark.py                  # all models, defaults
    python benchmark.py v3 meta policy   # specific models only
    python benchmark.py --seq-len 512 --batch 4 --steps 3  # custom sizes
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model registry: version -> (constructor, kwargs)
# ---------------------------------------------------------------------------

def _build_registry(vocab_size: int, num_steps: int, n_channels: int,
                    n_fourier_basis: int, logit_softcap: float,
                    decay_init: float):
    """Return {version_name: callable() -> model}."""

    common = dict(vocab_size=vocab_size, logit_softcap=logit_softcap)

    registry = {}

    # v1 — shared attention
    def make_v1():
        from v1_shared_attention.model import RegisterGPT
        return RegisterGPT(
            num_heads=8, num_kv_heads=4, num_steps=num_steps,
            n_fourier_basis=n_fourier_basis, n_channels=n_channels,
            rope_base=10000.0, qk_gain_init=1.5, activation="gelu", **common)
    registry["v1"] = make_v1

    # v2 — causal conv
    def make_v2():
        from v2_causal_conv.model import RegisterGPT
        return RegisterGPT(
            num_steps=num_steps, kernel_size=16,
            n_fourier_basis=n_fourier_basis, n_channels=n_channels,
            activation="gelu", **common)
    registry["v2"] = make_v2

    # v3 — associative memory (inline in train.py)
    def make_v3():
        from train import AssocRegisterLM
        return AssocRegisterLM(
            num_steps=num_steps, n_fourier_basis=n_fourier_basis,
            n_channels=n_channels, activation="gelu",
            decay_init=decay_init, **common)
    registry["v3"] = make_v3

    # v4 — param optimized
    def make_v4():
        from v4_param_optimized.model import RegisterGPTv4
        return RegisterGPTv4(
            unique_steps=5, invocations_per_step=2,
            n_fourier_basis=n_fourier_basis, n_channels=n_channels,
            n_heads=4, transform_rank=8, activation="gelu",
            decay_init=decay_init, **common)
    registry["v4"] = make_v4

    # gauss — FFT-based (run_all.py uses N_FOURIER_BASIS=64 for gauss)
    def make_gauss():
        from train import GaussRegisterGPT
        gauss_n_freq = max(n_fourier_basis, 64)  # needs enough freq bins
        return GaussRegisterGPT(
            num_steps=num_steps, n_freq=gauss_n_freq,
            n_channels=n_channels, activation="gelu",
            decay_init=decay_init, **common)
    registry["gauss"] = make_gauss

    # wave — brain wave
    def make_wave():
        from v6_brain_wave.model import BrainWaveGPT
        return BrainWaveGPT(
            num_cycles=num_steps, n_fourier_basis=n_fourier_basis,
            n_channels=n_channels, activation="gelu",
            slow_decay_init=4.0, fast_decay_init=2.0,
            band_split=(4, 4, 8), **common)
    registry["wave"] = make_wave

    # lgp — linear genetic programming
    def make_lgp():
        from v7_lgp.model import LGPGPT
        return LGPGPT(
            num_instructions=num_steps, n_fourier_basis=n_fourier_basis,
            n_channels=64, n_ops=8, decay_init=decay_init, **common)
    registry["lgp"] = make_lgp

    # graph — word graph
    def make_graph():
        from v8_word_graph.model import WordGraphGPT
        return WordGraphGPT(
            num_hops=num_steps, interaction_rank=64,
            activation="gelu", decay_init=decay_init, **common)
    registry["graph"] = make_graph

    # meta — Q-table meta-state
    def make_meta():
        from v9_meta_state.model import MetaStateGPT
        return MetaStateGPT(
            num_steps=num_steps, state_dim=64, inner_dim=128,
            activation="gelu", decay_init=decay_init, **common)
    registry["meta"] = make_meta

    # policy — state-dependent policy
    def make_policy():
        from v10_policy.model import PolicyGPT
        return PolicyGPT(
            num_steps=num_steps, state_dim=64, n_ops=8,
            decay_init=decay_init, **common)
    registry["policy"] = make_policy

    # v11 — brainwave (5 oscillatory primitives)
    def make_brainwave():
        from v11_brainwave.model import BrainWaveGPT
        return BrainWaveGPT(
            num_steps=num_steps, state_dim=64, inner_dim=128,
            gate_dim=64, **common)
    registry["brainwave"] = make_brainwave

    # sparse — sparse register addressing (v12)
    def make_sparse():
        from v12_sparse_register.model import SparseRegisterGPT
        return SparseRegisterGPT(
            num_steps=num_steps, k_active=256, inner_mul=2,
            activation="gelu", decay_init=decay_init, **common)
    registry["sparse"] = make_sparse

    return registry


# ---------------------------------------------------------------------------
# Benchmark one model
# ---------------------------------------------------------------------------

def benchmark_model(name, make_fn, vocab_size, batch, seq_len, n_iters, device):
    """Returns a dict of metrics, or an error string."""
    try:
        model = make_fn()
        use_bf16 = device.type == "cuda"
        if use_bf16:
            model = model.to(device).bfloat16()
            # Keep small params in fp32 (mirrors train.py logic)
            control_patterns = (
                "scale", "bias", "logit", "coeffs", "decay", "diag",
                "mix", "weight", "freq", "op_logits", "op_weights", "op_biases",
            )
            with torch.no_grad():
                for pname, p in model.named_parameters():
                    if (p.ndim < 2 or any(pat in pname for pat in control_patterns)) and p.dtype != torch.float32:
                        p.data = p.data.float()
        else:
            model = model.to(device).float()

        n_params = sum(p.numel() for p in model.parameters())
        raw_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        # Synthetic data
        input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        target_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)

        # Warmup
        model.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = model(input_ids, target_ids)
        loss.backward()
        model.zero_grad(set_to_none=True)

        # Timed forward+backward
        fwd_times, bwd_times, losses = [], [], []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = model(input_ids, target_ids)
            t1 = time.perf_counter()
            loss.backward()
            t2 = time.perf_counter()

            fwd_times.append(t1 - t0)
            bwd_times.append(t2 - t1)
            losses.append(loss.item())
            model.zero_grad(set_to_none=True)

        # Gradient health: run one more pass to inspect grads
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = model(input_ids, target_ids)
        loss.backward()

        grad_norms = []
        n_dead = 0
        for pname, p in model.named_parameters():
            if p.grad is not None:
                gn = p.grad.float().norm().item()
                grad_norms.append(gn)
                if gn == 0.0:
                    n_dead += 1

        avg_grad = sum(grad_norms) / max(len(grad_norms), 1)
        max_grad = max(grad_norms) if grad_norms else 0.0

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "params": n_params,
            "raw_MB": raw_bytes / 1e6,
            "est_int8_MB": n_params / 1e6,
            "fwd_ms": 1000 * sum(fwd_times) / len(fwd_times),
            "bwd_ms": 1000 * sum(bwd_times) / len(bwd_times),
            "total_ms": 1000 * (sum(fwd_times) + sum(bwd_times)) / len(fwd_times),
            "loss": sum(losses) / len(losses),
            "loss_std": (sum((l - sum(losses)/len(losses))**2 for l in losses) / max(len(losses)-1, 1)) ** 0.5,
            "avg_grad": avg_grad,
            "max_grad": max_grad,
            "dead_params": n_dead,
            "tok_per_s": batch * seq_len / (sum(fwd_times) / len(fwd_times)),
        }

    except Exception as e:
        traceback.print_exc()
        return str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rapid model benchmark")
    parser.add_argument("models", nargs="*", help="Model versions to test (default: all)")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8, help="num_steps / num_instructions")
    parser.add_argument("--n-channels", type=int, default=128)
    parser.add_argument("--n-fourier", type=int, default=16)
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations per model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    registry = _build_registry(
        vocab_size=args.vocab_size,
        num_steps=args.steps,
        n_channels=args.n_channels,
        n_fourier_basis=args.n_fourier,
        logit_softcap=30.0,
        decay_init=3.0,
    )

    selected = args.models if args.models else list(registry.keys())
    unknown = [m for m in selected if m not in registry]
    if unknown:
        print(f"Unknown models: {unknown}")
        print(f"Available: {list(registry.keys())}")
        sys.exit(1)

    print(f"Benchmarking {len(selected)} models on {device}")
    print(f"  vocab={args.vocab_size} seq_len={args.seq_len} batch={args.batch} steps={args.steps}")
    print(f"  channels={args.n_channels} fourier={args.n_fourier} iters={args.iters}")
    print()

    results = {}
    for name in selected:
        print(f"  {name:10s} ... ", end="", flush=True)
        t0 = time.perf_counter()
        r = benchmark_model(
            name, registry[name], args.vocab_size,
            args.batch, args.seq_len, args.iters, device,
        )
        elapsed = time.perf_counter() - t0
        if isinstance(r, str):
            print(f"FAILED ({elapsed:.1f}s): {r}")
        else:
            print(f"OK  loss={r['loss']:.3f}  fwd={r['fwd_ms']:.0f}ms  params={r['params']/1e3:.0f}K  ({elapsed:.1f}s)")
        results[name] = r

    # Summary table
    print()
    print("=" * 120)
    hdr = f"{'Model':10s} {'Params':>8s} {'Raw MB':>8s} {'~Int8 MB':>8s} {'Fwd ms':>8s} {'Bwd ms':>8s} {'Total ms':>9s} {'tok/s':>8s} {'Loss':>8s} {'AvgGrad':>9s} {'MaxGrad':>9s} {'Dead':>5s}"
    print(hdr)
    print("-" * 120)

    ok_results = {k: v for k, v in results.items() if isinstance(v, dict)}
    # Sort by loss (lowest first)
    for name in sorted(ok_results, key=lambda k: ok_results[k]["loss"]):
        r = ok_results[name]
        print(
            f"{name:10s} "
            f"{r['params']/1e3:>7.0f}K "
            f"{r['raw_MB']:>8.2f} "
            f"{r['est_int8_MB']:>8.2f} "
            f"{r['fwd_ms']:>8.1f} "
            f"{r['bwd_ms']:>8.1f} "
            f"{r['total_ms']:>9.1f} "
            f"{r['tok_per_s']:>8.0f} "
            f"{r['loss']:>8.4f} "
            f"{r['avg_grad']:>9.2e} "
            f"{r['max_grad']:>9.2e} "
            f"{r['dead_params']:>5d}"
        )

    failed = {k: v for k, v in results.items() if isinstance(v, str)}
    if failed:
        print()
        print("FAILED:")
        for name, err in failed.items():
            print(f"  {name}: {err}")

    print("=" * 120)

    # Insights
    if ok_results:
        best_loss = min(ok_results, key=lambda k: ok_results[k]["loss"])
        fastest = min(ok_results, key=lambda k: ok_results[k]["total_ms"])
        smallest = min(ok_results, key=lambda k: ok_results[k]["params"])
        best_throughput = max(ok_results, key=lambda k: ok_results[k]["tok_per_s"])
        print()
        print(f"  Best loss:       {best_loss:10s}  ({ok_results[best_loss]['loss']:.4f})")
        print(f"  Fastest:         {fastest:10s}  ({ok_results[fastest]['total_ms']:.1f}ms fwd+bwd)")
        print(f"  Smallest:        {smallest:10s}  ({ok_results[smallest]['params']/1e3:.0f}K params)")
        print(f"  Best throughput: {best_throughput:10s}  ({ok_results[best_throughput]['tok_per_s']:.0f} tok/s)")

        # Flag gradient issues
        sick = [k for k, v in ok_results.items() if v["dead_params"] > 0 or v["avg_grad"] > 100 or v["avg_grad"] < 1e-8]
        if sick:
            print()
            print("  Gradient warnings:")
            for k in sick:
                r = ok_results[k]
                issues = []
                if r["dead_params"] > 0:
                    issues.append(f"{r['dead_params']} dead params")
                if r["avg_grad"] > 100:
                    issues.append(f"avg_grad={r['avg_grad']:.1e} (exploding)")
                if r["avg_grad"] < 1e-8:
                    issues.append(f"avg_grad={r['avg_grad']:.1e} (vanishing)")
                print(f"    {k}: {', '.join(issues)}")


if __name__ == "__main__":
    main()
