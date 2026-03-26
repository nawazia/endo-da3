"""
LoRA (Low-Rank Adaptation) for the GastroNet/DINOv2 backbone.

Injects trainable rank-r adapters into every attn.qkv and attn.proj
Linear layer in model.backbone.blocks, leaving base weights frozen.

Usage:
    from endo_da3.lora import inject_lora, lora_state_dict

    inject_lora(model, rank=4, lora_alpha=4.0)
    # now model.backbone.blocks.*.attn.qkv are LoRALinear wrappers
    # only lora_A / lora_B parameters require_grad

    # save only LoRA weights (for compact checkpoints):
    torch.save(lora_state_dict(model), "lora.pt")
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a frozen base weight and a
    trainable low-rank side-path:

        y = x W^T + b  +  (x A^T) B^T  ×  (alpha / rank)

    A is initialised with Kaiming-uniform, B with zeros so the adapter
    contributes nothing at the start of Stage 2a (stable warm-up).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear          # base weight — stays frozen
        d_out, d_in = linear.weight.shape
        dev = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=dev))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def extra_repr(self) -> str:
        d_out, d_in = self.linear.weight.shape
        rank = self.lora_A.shape[0]
        return f"d_in={d_in}, d_out={d_out}, rank={rank}, scale={self.scale:.3f}"


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def _set_nested(parent: nn.Module, attr_path: str, new_module: nn.Module):
    """Set a nested attribute given a dotted path, e.g. 'attn.qkv'."""
    parts = attr_path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora(
    model: nn.Module,
    rank: int   = 4,
    lora_alpha: float = 4.0,
    target_keys: tuple[str, ...] = ("attn.qkv",),
) -> int:
    """
    Replace matching Linear layers in model.backbone.blocks with LoRALinear.
    Base weights are frozen; lora_A / lora_B are left trainable.

    Only attn.qkv is targeted by default (not attn.proj) to avoid grid
    artifacts in dense depth output caused by cross-patch mixing in proj.

    Returns the number of adapters injected.
    """
    backbone = model.backbone
    n = 0
    for block_name, block in backbone.blocks.named_children():
        for key in target_keys:
            parts = key.split(".")
            # Navigate to the parent of the target linear
            parent = block
            try:
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue

            if not isinstance(linear, nn.Linear):
                continue

            # Freeze base weight
            linear.weight.requires_grad_(False)
            if linear.bias is not None:
                linear.bias.requires_grad_(False)

            # Replace with LoRALinear
            lora_layer = LoRALinear(linear, rank=rank, alpha=lora_alpha)
            setattr(parent, parts[-1], lora_layer)
            n += 1

    return n


def count_lora_params(model: nn.Module) -> tuple[int, int]:
    """Return (lora_params, total_trainable_params)."""
    lora  = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and ("lora_A" in n or "lora_B" in n))
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return lora, total


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA adapter weights (compact checkpoint)."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}
