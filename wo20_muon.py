"""Muon optimizer (single file): orthogonalized momentum via
Newton–Schulz for 2D weight matrices, with AdamW-fallback groups.

From Keller Jordan's Muon (modded-nanogpt); the recipe Qwen3.8-Flash-Next
/ LongCat-Flash-Lite report training with. Rules of use:
- Matrices that are updated as WHOLE weights (attention/FFN projections)
  get Newton–Schulz-orthogonalized momentum.
- Embedding-like tensors — byte embeddings, layer norms, the tied
  lm_head, relative-attention bias, and memory-layer lookup tables
  (random access per the quantization-class policy) — stay on AdamW
  inside the same optimizer object, via groups flagged ``adamw=True``.

State save/resume works through the standard ``torch.optim`` dict.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    # quintic iteration coefficients from the modded-nanogpt lineage
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.mT
    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


_MUON_DEFAULTS = {"lr": 0.01, "momentum": 0.95, "nesterov": True,
                  "ns_steps": 5, "weight_decay": 0.0}


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0) -> None:
        settings = {**_MUON_DEFAULTS, "lr": lr, "momentum": momentum,
                    "nesterov": nesterov, "ns_steps": ns_steps,
                    "weight_decay": weight_decay, "adamw": False}
        super().__init__(list(params), settings)

    def add_adamw_group(self, params, lr: float = 1e-4, betas=(0.9, 0.999),
                        weight_decay: float = 0.0) -> None:
        """Embedding-like parameters: standard AdamW math, shared
        scheduler (the cosine scales every group's lr)."""
        self.add_param_group({
            "params": list(params),
            "lr": lr,
            "betas": tuple(betas),
            "weight_decay": weight_decay,
            "adamw": True,
        })

    def add_headwise_group(self, params, heads: int) -> None:
        """Per-head preconditioning (head-wise Muon): each attention
        head's rows of a Q/K weight are orthogonalized as their own
        matrix. Accepts params already inside a Muon group (their
        group is converted in place) or all-new params (a new group is
        added with the constructor's default settings)."""
        params = list(params)
        owned = {id(p) for group in self.param_groups for p in group["params"]}
        hits = {id(p) for p in params if id(p) in owned}
        if hits and len(hits) != len(params):
            raise ValueError("headwise routing takes all-new or all-existing params")
        if hits:
            for group in self.param_groups:
                members = {id(q) for q in group["params"]}
                if hits & members:
                    if hits != members:
                        raise ValueError(
                            "headwise routing converts whole groups; "
                            "split the group first")
                    group["headwise"] = True
                    group["heads"] = heads
            return
        settings = {**_MUON_DEFAULTS, "adamw": False,
                    "params": params, "headwise": True, "heads": heads}
        self.add_param_group(settings)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ARG002
        for group in self.param_groups:
            if group.get("adamw"):
                self._adamw_step(group)
            elif group.get("headwise"):
                self._muon_step_headwise(group)
            else:
                self._muon_step(group)

    def _momentum_direction(self, p, group):
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p.grad)
        buf = state["momentum_buffer"]
        buf.lerp_(p.grad, 1 - group["momentum"])
        return p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf

    def _muon_step(self, group) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = self._momentum_direction(p, group)
            u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
            if group["weight_decay"]:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(u.to(p.dtype), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)

    def _muon_step_headwise(self, group) -> None:
        heads = group["heads"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.size(0) % heads:
                raise ValueError(f"param rows {p.size(0)} not divisible by {heads} heads")
            g = self._momentum_direction(p, group)
            if group["weight_decay"]:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            d = p.size(0) // heads
            for h in range(heads):
                rows = slice(h * d, (h + 1) * d)
                u = zeropower_via_newtonschulz5(g[rows], steps=group["ns_steps"])
                scale = max(1, g[rows].size(-2) / g[rows].size(-1)) ** 0.5
                p[rows].add_(u.to(p.dtype), alpha=-group["lr"] * scale)

    def _adamw_step(self, group) -> None:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "adamw_exp_avg" not in state:
                state["adamw_exp_avg"] = torch.zeros_like(p.grad)
                state["adamw_exp_avg_sq"] = torch.zeros_like(p.grad)
                state["adamw_step"] = 0
            exp_avg, exp_avg_sq = state["adamw_exp_avg"], state["adamw_exp_avg_sq"]
            state["adamw_step"] += 1
            exp_avg.lerp_(p.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            bias_c1 = 1 - beta1 ** state["adamw_step"]
            bias_c2 = 1 - beta2 ** state["adamw_step"]
            denom = (exp_avg_sq / bias_c2).sqrt_().add_(1e-8)
            if group["weight_decay"]:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.addcdiv_(exp_avg / bias_c1, denom, value=-group["lr"])


def _is_embedding_like(name: str, p) -> bool:
    return (
        p.ndim < 2
        or "embed_tokens" in name
        or name == "shared.weight"  # T5 tied byte embedding (transformers 5.x)
        or "lm_head" in name
        or "relative_attention" in name
        or "memory.values" in name
        or "memory.k1" in name
        or "memory.k2" in name
    )


def embed_named(named_params):
    """The 2D embedding-like tensors (tied embedding, prediction heads,
    memory tables) — SinkhornUpdate's surface: rows are token identity,
    columns features. 1D params (layer norms) are excluded; they have
    no row/column structure to balance."""
    return [
        (name, p) for name, p in named_params
        if p.requires_grad and p.ndim >= 2 and _is_embedding_like(name, p)
    ]


def split_parameters(named_params):
    """The standard split: orthogonalizable 2D hidden weights vs
    embedding-like tensors (1D params, embeddings, tied head, relative
    bias, memory lookup tables)."""
    muon, adamw = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        (adamw if _is_embedding_like(name, p) else muon).append(p)
    return muon, adamw


def qk_named(named_params):
    """The Q/K projection weights head-wise Muon applies to (DeepSeek
    V4.1 Flash sec 2.5; GLM-5 and Kimi-K3 validate the same split).
    T5 names both self- and cross-attention projections."""
    import re

    pattern = re.compile(r"(SelfAttention|EncDecAttention)\.(q|k)\.weight$")
    return [(name, p) for name, p in named_params if pattern.search(name)]
