"""
RegisterGPT v3 — Register machine with associative memory.

Replaces attention with a running outer-product associative memory (1970s math).
Each step: query the memory, transform, update the memory.
Content-based dynamic routing without O(T²) attention.

  Input:  one-hot("cat") → R["cat"] = 1.0, everything else 0.0
  State:  always a distribution over words
  Output: register state IS the prediction — no output projection

Architecture:
  1. One-hot encoding over vocabulary (no learned embedding)
  2. N unique steps, each:
     a. Query associative memory   (content-based cross-position retrieval)
     b. Fourier register op        (within-position: combine word activations)
     c. Update associative memory  (store new associations)
  3. Register state → softcap → cross-entropy loss

No attention. No embedding. No output projection.
Cross-position mixing via running associative memory (outer products).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------

def make_fourier_basis(dim: int, n_basis: int) -> Tensor:
    """Fourier basis functions over vocabulary indices."""
    pos = torch.arange(dim, dtype=torch.float32) / dim
    basis = torch.zeros(dim, 2 * n_basis)
    for k in range(n_basis):
        freq = k + 1
        basis[:, 2 * k] = torch.cos(2 * math.pi * freq * pos)
        basis[:, 2 * k + 1] = torch.sin(2 * math.pi * freq * pos)
    return basis


# ---------------------------------------------------------------------------
# Fourier projection (cheap learned linear map via Fourier basis)
# ---------------------------------------------------------------------------

class FourierProjection(nn.Module):
    """Project from vocab-space to a small channel space via Fourier basis.

    Instead of a full (vocab, channels) matrix (~131K params for 1024×128),
    parameterize through Fourier coefficients: (channels, 2*n_basis) (~8K params).
    """

    def __init__(self, n_basis: int, n_channels: int, soft: bool = False):
        super().__init__()
        self.soft = soft
        self.coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * 0.02)

    def forward(self, basis: Tensor) -> Tensor:
        # basis: (V, 2*n_basis), coeffs: (C, 2*n_basis)
        # output: (V, C)
        w = basis @ self.coeffs.T
        if self.soft:
            w = torch.softmax(w, dim=0)
        return w


# ---------------------------------------------------------------------------
# Associative memory (outer product, 1970s Hopfield-style)
# ---------------------------------------------------------------------------

class AssociativeMemoryStep(nn.Module):
    """One step of reading from and writing to a running associative memory.

    Memory M ∈ R^(C×C) accumulates key⊗value outer products.
    Query retrieves content-addressed information.
    All via Fourier-parameterized projections (cheap).

    This is the cross-position mixing mechanism — replaces attention.
    """

    def __init__(self, n_basis: int, n_channels: int, decay_init: float = 0.95):
        super().__init__()
        self.n_channels = n_channels

        # Fourier projections: vocab → channels
        self.query_proj = FourierProjection(n_basis, n_channels, soft=True)
        self.key_proj = FourierProjection(n_basis, n_channels, soft=True)
        self.value_proj = FourierProjection(n_basis, n_channels, soft=True)
        self.output_proj = FourierProjection(n_basis, n_channels, soft=False)

        # Memory decay — how quickly old associations fade
        self.decay = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        """
        x: (B, T, V) — register states
        basis: (V, 2*n_basis) — Fourier basis
        returns: (B, T, V) — retrieved information mapped back to vocab space
        """
        B, T, V = x.shape
        C = self.n_channels
        dtype = x.dtype

        # Project vocab → channels via Fourier
        q_w = self.query_proj(basis).to(dtype)   # (V, C)
        k_w = self.key_proj(basis).to(dtype)     # (V, C)
        v_w = self.value_proj(basis).to(dtype)   # (V, C)
        o_w = self.output_proj(basis).to(dtype)  # (V, C)

        queries = x @ q_w       # (B, T, C)
        keys = x @ k_w          # (B, T, C)
        values = x @ v_w        # (B, T, C)

        decay = torch.sigmoid(self.decay)

        # Scan: maintain running memory, query at each position
        M = torch.zeros(B, C, C, device=x.device, dtype=dtype)
        outputs = []
        for t in range(T):
            # Retrieve: content-based lookup
            retrieved = torch.bmm(queries[:, t:t+1, :], M).squeeze(1)  # (B, C)

            # Update memory: store new association
            k_t = keys[:, t, :]    # (B, C)
            v_t = values[:, t, :]  # (B, C)
            M = decay * M + torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))  # (B, C, C)

            outputs.append(retrieved)

        retrieved = torch.stack(outputs, dim=1)  # (B, T, C)

        # Map back to vocab space
        return retrieved @ o_w.T * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Fourier register operation (within-position transform)
# ---------------------------------------------------------------------------

class FourierRegisterOp(nn.Module):
    """Within-position register transform via Fourier basis."""

    def __init__(self, n_basis: int, n_channels: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.channel_mix = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        read_w = torch.softmax(basis @ self.read_coeffs.T, dim=0)
        values = x @ read_w.to(x.dtype)
        values = values @ self.channel_mix.to(x.dtype) + self.bias.to(x.dtype)
        if self.activation == "relu2":
            values = F.relu(values).square()
        elif self.activation == "swish":
            values = F.silu(values)
        else:
            values = F.gelu(values)
        write_w = (basis @ self.write_coeffs.T).to(x.dtype)
        return values @ write_w.T * self.out_scale.to(x.dtype)


# ---------------------------------------------------------------------------
# Register step
# ---------------------------------------------------------------------------

class RegisterStep(nn.Module):
    """One LGP instruction: memory query + register transform + memory update."""

    def __init__(self, n_basis: int, n_channels: int, activation: str = "gelu",
                 decay_init: float = 0.95):
        super().__init__()
        self.memory = AssociativeMemoryStep(n_basis, n_channels, decay_init)
        self.register_op = FourierRegisterOp(n_basis, n_channels, activation)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.op_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        D = x.size(-1)
        x = x + self.mem_scale.to(x.dtype) * self.memory(
            F.rms_norm(x, (D,)), basis)
        x = x + self.op_scale.to(x.dtype) * self.register_op(
            F.rms_norm(x, (D,)), basis)
        return x


# ---------------------------------------------------------------------------
# RegisterGPT v3
# ---------------------------------------------------------------------------

class RegisterGPT(nn.Module):
    """Register machine with associative memory.

    No embedding. No output projection. No attention.
    Cross-position mixing via running outer-product memory.
    Within-position transforms via Fourier register ops.
    """

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 n_fourier_basis: int = 16, n_channels: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 0.95):
        super().__init__()
        dim = vocab_size
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            RegisterStep(n_fourier_basis, n_channels, activation, decay_init)
            for _ in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fourier_basis",
                             make_fourier_basis(dim, n_fourier_basis))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size

        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = step(x, self.fourier_basis)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
