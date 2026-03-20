#!/usr/bin/env python3
"""Run all model versions sequentially and collect results."""
import os
import subprocess
import sys


MODELS = [
    {"version": "v3",     "env": {}},
    {"version": "v4",     "env": {}},
    {"version": "gauss",  "env": {"N_FOURIER_BASIS": "64"}},
    {"version": "wave",   "env": {}},
    {"version": "lgp",    "env": {"NUM_STEPS": "16", "N_CHANNELS": "64"}},
    {"version": "graph",  "env": {}},
    {"version": "meta",   "env": {}},
    {"version": "policy", "env": {}},
]

# Detect GPU count
try:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    n_gpus = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 1
except FileNotFoundError:
    n_gpus = 1


def main():
    batch = os.environ.get("TRAIN_BATCH_TOKENS", "491520")
    grad_accum = os.environ.get("GRAD_ACCUM_STEPS", "16")
    log_every = os.environ.get("TRAIN_LOG_EVERY", "50")
    iterations = os.environ.get("ITERATIONS", "500")

    results = []
    for m in MODELS:
        version = m["version"]
        run_id = f"{version}_eval"
        print(f"\n{'='*60}")
        print(f"  Running {version} (run_id={run_id})")
        print(f"{'='*60}\n")

        env = {
            **os.environ,
            "MODEL_VERSION": version,
            "TRAIN_BATCH_TOKENS": batch,
            "GRAD_ACCUM_STEPS": grad_accum,
            "TRAIN_LOG_EVERY": log_every,
            "ITERATIONS": iterations,
            "RUN_ID": run_id,
            **m["env"],
        }

        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={n_gpus}",
            "train.py",
        ]

        ret = subprocess.run(cmd, env=env)
        status = "OK" if ret.returncode == 0 else f"FAIL({ret.returncode})"
        results.append((version, status))
        print(f"\n  {version}: {status}\n")

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for version, status in results:
        print(f"  {version:10s} {status}")

    # Print results table
    print(f"\n{'='*60}")
    print("  Results Table")
    print(f"{'='*60}\n")
    subprocess.run([sys.executable, "results.py"])


if __name__ == "__main__":
    main()
