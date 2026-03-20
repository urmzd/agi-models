"""
v12: Sparse Register Machine — Hard sparse addressing from Linear GP.

Key insight from linear-gp: instructions address SPECIFIC registers, not all of them.
Each step operates on a LEARNED SUBSET of k registers out of V total.
This gives full-rank operations in the active subspace (unlike v9's rank-64
bottleneck) while keeping parameter count manageable.

Architecture:
  1. One-hot → register state (vocab-dimensional)
  2. N steps, each:
     a. ROUTE: learned logits → top-k register selection (fixed per step)
     b. GATHER: extract k active registers via differentiable gather
     c. CROSS-POSITION: causal decay memory in k-dim (full-rank in subspace)
     d. WITHIN-POSITION: MLP in k-dim
     e. SCATTER: write delta back via differentiable scatter
  3. Register state → softcap → cross-entropy loss

Compression advantage:
  - Operations are k×k matrices (k << V), many fewer unique values
  - Routing logits are V floats per step → tiny
  - Inactive register dimensions carry no learned info → compress well

From linear-gp:
  - Hard register addressing (not soft like v7/v10)
  - Fixed addressing per instruction (same for all positions/inputs)
  - Diverse routing: each step's registers are initialized non-overlapping
  - Gradients flow through gather → transforms → scatter (standard autograd)

No embedding. No output projection. No Fourier. No attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalDecayMemory(nn.Module):
    """Cross-position mixing via causal decay in the sparse subspace.

    Same mechanism as v9's QTable, but operating in the k-dim subspace
    of active registers instead of projecting from full V-dim.
    """

    def __init__(self, dim: int, decay_init: float = 3.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(m.weight, std=0.02)

        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, k) → (B, T, k)"""
        B, T, k = x.shape
        dtype = x.dtype

        q = self.q_proj(x.float()).to(dtype)
        k_ = self.k_proj(x.float()).to(dtype)
        v = self.v_proj(x.float()).to(dtype)

        scores = torch.bmm(q, k_.transpose(1, 2))  # (B, T, T)

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        retrieved = torch.bmm(scores, v)
        return self.o_proj(retrieved.float()).to(dtype) * self.out_scale.to(dtype)


class SparseMLP(nn.Module):
    """Within-position transform in the sparse subspace."""

    def __init__(self, dim: int, inner_dim: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        self.down = nn.Linear(dim, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(inner_dim))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        h = self.down(x.float()) + self.bias
        if self.activation == "relu":
            h = F.relu(h)
        elif self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)
        return self.up(h).to(dtype) * self.out_scale.to(dtype)


class SparseRegisterStep(nn.Module):
    """One instruction: sparse route → gather → transform → scatter.

    Each step has a fixed routing pattern (same indices for all positions
    and inputs), like LGP's hard-coded register addresses. The routing
    is initialized to be diverse across steps — each step "owns" a
    primary set of registers, with overlap for information flow.

    Gradient flow: gather and scatter are both differentiable in PyTorch.
    Gradients flow from loss → scatter → transforms → gather → x.
    The route_logits themselves don't get gradients (topk is discrete),
    which is by design: routing is structural, not dynamic.
    """

    def __init__(self, vocab_size: int, k_active: int, inner_mul: int = 2,
                 activation: str = "gelu", decay_init: float = 3.0,
                 step_idx: int = 0, total_steps: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.k_active = k_active

        # Initialize routing so each step addresses a different set.
        # Use a shifted pattern: step i gets registers starting at offset i*V/N,
        # wrapping around. This ensures full coverage across all steps.
        route_init = torch.zeros(vocab_size)
        stride = vocab_size / total_steps
        offset = int(step_idx * stride)
        for j in range(k_active):
            idx = (offset + j) % vocab_size
            route_init[idx] = 1.0 + torch.randn(1).item() * 0.01
        self.route_logits = nn.Parameter(route_init)

        # Cross-position: causal decay memory in k-dim (full-rank!)
        self.memory = CausalDecayMemory(k_active, decay_init)

        # Within-position: MLP in k-dim
        self.mlp = SparseMLP(k_active, k_active * inner_mul, activation)

        self.mem_scale = nn.Parameter(torch.ones(1))
        self.mlp_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, V) → (B, T, V)"""
        B, T, V = x.shape
        dtype = x.dtype
        k = self.k_active

        # ROUTE: select top-k registers (fixed per step, same for all positions)
        _, indices = self.route_logits.topk(k)  # (k,)
        idx = indices.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # (B, T, k)

        # GATHER: extract k registers (differentiable w.r.t. x)
        gathered = torch.gather(x, -1, idx)  # (B, T, k)

        # CROSS-POSITION in sparse subspace
        g_norm = F.rms_norm(gathered, (k,))
        mem_out = self.memory(g_norm)
        gathered = gathered + self.mem_scale.to(dtype) * mem_out

        # WITHIN-POSITION in sparse subspace
        g_norm = F.rms_norm(gathered, (k,))
        mlp_out = self.mlp(g_norm)
        gathered = gathered + self.mlp_scale.to(dtype) * mlp_out

        # SCATTER: write transformed values back (differentiable)
        # delta = what the transforms added (gathered_new - gathered_old)
        original = torch.gather(x, -1, idx)
        delta = gathered - original

        # Non-in-place scatter: creates proper autograd graph
        # Gradients flow: output → delta → transforms → gathered → x
        delta_full = torch.zeros_like(x).scatter(-1, idx, delta)
        return x + delta_full


class SparseRegisterGPT(nn.Module):
    """Language model with sparse register addressing.

    Each step addresses only k out of V registers, operating at full rank
    in the active subspace. Cross-position memory runs at k×k instead of
    V×d, giving higher effective rank per parameter than v9's bottleneck.

    Compare to v9 (state_dim=64):
      v9:  cross-position is rank-64 in V=1024 space (V→64→V projection)
      v12: cross-position is rank-k in k-dim space (full-rank within subset)

    Routing is initialized so steps tile the register space with overlap,
    ensuring every register is touched by at least one step.

    No embedding. No output projection. No Fourier. No attention.
    """

    def __init__(self, vocab_size: int = 1024, num_steps: int = 12,
                 k_active: int = 256, inner_mul: int = 2,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            SparseRegisterStep(vocab_size, k_active, inner_mul,
                               activation, decay_init,
                               step_idx=i, total_steps=num_steps)
            for i in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = step(x)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
