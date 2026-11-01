"""Successive low-rank branches; generated B is a runtime autograd tensor."""
from contextvars import ContextVar
import math
import torch
from torch import nn
from torch.nn import functional as F

GENERATED = ContextVar("dclt_generated", default=None)


def init_a(tensor, recipe):
    if recipe == "kaiming_uniform":
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
    elif recipe == "xavier_normal":
        nn.init.xavier_normal_(tensor)
    else:
        raise ValueError(recipe)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=None, dynamic=False, key="", a_init="kaiming_uniform", dora=False, cond_maps=None):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.scale = rank, (rank if alpha in (None, "rank") else alpha) / rank
        self.dynamic, self.key, self.merged = dynamic, key, False
        self.dora = dora
        # Shared CondLoRA maps are owned by the model, not registered once per layer.
        object.__setattr__(self, "cond_maps", cond_maps)
        if cond_maps is None:
            self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
            init_a(self.A, a_init)
            if not dynamic:
                self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        if dora:
            self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))

    def factors(self):
        if self.cond_maps is not None:
            map_a, map_b = self.cond_maps
            # A = Linear_A(W0)^T, B = Linear_B(W0^T), bias-free.
            return map_a(self.base.weight).T, map_b(self.base.weight.T)
        if self.dynamic:
            context = GENERATED.get()
            if context is None or self.key not in context:
                raise RuntimeError("Dynamic adapter used outside a DCLT forward")
            return self.A, context[self.key]
        return self.A, self.B

    def forward(self, x):
        if self.merged:
            return self.base(x)
        A, B = self.factors()
        if self.dora:
            direction = self.base.weight + self.scale * (B @ A)
            # DoRA's magnitude/direction separation; detach normalization per official implementation.
            weight = self.magnitude[:, None] * direction / direction.norm(dim=1, keepdim=True).detach().clamp_min(1e-12)
            return F.linear(x, weight, self.base.bias)
        low = F.linear(x, A)
        if self.dynamic:
            if low.shape[0] != B.shape[0]:
                raise RuntimeError("Generated factor batch mismatch")
            branch = torch.einsum("b...r,bor->b...o", low, B)
        else:
            branch = F.linear(low, B)
        return self.base(x) + self.scale * branch

    @torch.no_grad()
    def merge(self):
        if self.dynamic:
            raise ValueError("Input-conditioned adapters cannot be merged")
        if self.merged:
            return
        A, B = self.factors()
        weight = self.base.weight + self.scale * (B @ A)
        if self.dora:
            weight = self.magnitude[:, None] * weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.base.weight.copy_(weight)
        self.merged = True


class LoRAConv2d(nn.Module):
    def __init__(self, base, rank=4, alpha=None, dynamic=False, key="", a_init="kaiming_uniform"):
        super().__init__()
        if base.groups != 1:
            raise ValueError("Paper's convolutional factorization requires ungrouped convolution")
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.scale = rank, (rank if alpha in (None, "rank") else alpha) / rank
        self.dynamic, self.key, self.merged = dynamic, key, False
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_channels, *base.kernel_size))
        init_a(self.A, a_init)
        if not dynamic:
            self.B = nn.Parameter(base.weight.new_zeros(base.out_channels, rank, 1, 1))

    def factors(self):
        if self.dynamic:
            context = GENERATED.get()
            if context is None or self.key not in context:
                raise RuntimeError("Dynamic convolution used outside a DCLT forward")
            return self.A, context[self.key]
        return self.A, self.B

    def forward(self, x):
        if self.merged:
            return self.base(x)
        A, B = self.factors()
        # _conv_forward preserves padding_mode, stride, padding and dilation, without bias.
        low = self.base._conv_forward(x, A, None)
        branch = (torch.einsum("brhw,bor->bohw", low, B.flatten(2)) if self.dynamic
                  else F.conv2d(low, B))
        return self.base(x) + self.scale * branch

    @torch.no_grad()
    def merge(self):
        if self.dynamic:
            raise ValueError("Input-conditioned adapters cannot be merged")
        if not self.merged:
            delta = torch.einsum("or,rihw->oihw", self.B[:, :, 0, 0], self.A)
            self.base.weight.add_(self.scale * delta)
            self.merged = True


def sliced_linear(weight, bias):
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        module.weight.copy_(weight)
        if bias is not None:
            module.bias.copy_(bias)
    return module.to(device=weight.device, dtype=weight.dtype)


class SplitQKV(nn.Module):
    """timm fused QKV, split in deterministic Q/K/V order without changing its output."""
    def __init__(self, original):
        super().__init__()
        weights = original.weight.chunk(3, dim=0)
        biases = original.bias.chunk(3) if original.bias is not None else (None,) * 3
        self.q, self.k, self.v = [sliced_linear(w, b) for w, b in zip(weights, biases)]

    def forward(self, x):
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)


class VisionSelfAttention(nn.Module):
    """torchvision ViT MultiheadAttention with accessible Q/K/V/O projections."""
    def __init__(self, original):
        super().__init__()
        if not original.batch_first or original.kdim != original.embed_dim or original.vdim != original.embed_dim:
            raise ValueError("Expected batch-first equal-width ViT self-attention")
        self.num_heads, self.dropout = original.num_heads, original.dropout
        weights = original.in_proj_weight.chunk(3, 0)
        biases = original.in_proj_bias.chunk(3) if original.in_proj_bias is not None else (None,) * 3
        self.q, self.k, self.v = [sliced_linear(w, b) for w, b in zip(weights, biases)]
        self.out_proj = original.out_proj

    def forward(self, query, key, value, need_weights=True, attn_mask=None, key_padding_mask=None, average_attn_weights=True, is_causal=False):
        if attn_mask is not None or key_padding_mask is not None or is_causal:
            raise ValueError("This vision adapter expects unmasked self-attention")
        batch, tokens, dim = query.shape
        def reshape(x):
            return x.reshape(batch, -1, self.num_heads, dim // self.num_heads).transpose(1, 2)
        q, k, v = reshape(self.q(query)), reshape(self.k(key)), reshape(self.v(value))
        scores = (q @ k.transpose(-1, -2)) * ((dim // self.num_heads) ** -0.5)
        probabilities = scores.softmax(dim=-1)
        probabilities = F.dropout(probabilities, self.dropout, training=self.training)
        output = (probabilities @ v).transpose(1, 2).reshape(batch, tokens, dim)
        weights = probabilities.mean(1) if average_attn_weights else probabilities
        return self.out_proj(output), weights if need_weights else None

