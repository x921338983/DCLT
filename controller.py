import math
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


class LowRankPrompt(nn.Module):
    def __init__(self, size=224, rank=4):
        super().__init__()
        self.Bp = nn.Parameter(torch.empty(3, size, rank))
        self.Ap = nn.Parameter(torch.zeros(3, rank, size))
        nn.init.normal_(self.Bp, std=0.02)
        self._cached = None

    def tensor(self):
        return self._cached if self._cached is not None else self.Bp @ self.Ap

    def factor_vector(self):
        return torch.cat([torch.cat([self.Bp[c].flatten(), self.Ap[c].flatten()]) for c in range(3)])

    def forward(self, x):
        if x.shape[-2:] != self.Bp.shape[1:2] + self.Ap.shape[2:3]:
            raise ValueError("Prompt and image sizes differ; use the configured preprocessing")
        return x + self.tensor()[None]

    @torch.no_grad()
    def cache(self):
        if self.training:
            raise RuntimeError("Prompt cache is inference-only")
        self._cached = (self.Bp @ self.Ap).detach()

    def train(self, mode=True):
        if mode:
            self._cached = None
        return super().train(mode)

    def _apply(self, fn):
        self._cached = None
        return super()._apply(fn)


class InstanceController(nn.Module):
    def __init__(self, kind, pool_size=8, eps=1e-5, prompt_dim=5376):
        super().__init__()
        self.kind, self.pool_size, self.eps = kind, pool_size, eps
        self.dim = prompt_dim if kind == "original" else 3 * pool_size ** 2
        if kind == "matched_static":
            self.static = nn.Parameter(torch.randn(self.dim))

    def forward(self, normalized_prompted, normalized_image, prompt):
        batch = normalized_prompted.shape[0]
        if self.kind == "matched_static":
            return self.static[None].expand(batch, -1)
        if self.kind == "original":
            if prompt is None:
                raise ValueError("Original controller requires a prompt")
            return prompt.factor_vector()[None].expand(batch, -1)
        image = normalized_image if self.kind == "image_only" else normalized_prompted
        u = F.adaptive_avg_pool2d(image, self.pool_size).flatten(1)
        return F.layer_norm(u, (self.dim,), eps=self.eps)


class ParameterGenerator(nn.Module):
    """Bias-free SLM and its controlled structure ablations."""
    def __init__(self, specs, m=192, q=8, mapping="shared", init="gaussian", decoder_seed=0):
        super().__init__()
        self.specs = specs  # [(key, B shape without batch), ...], fixed insertion order.
        self.mapping, self.m, self.q = mapping, m, q
        sizes = [math.prod(shape) for _, shape in specs]
        self.sizes, self.pB = sizes, sum(sizes)
        if not specs:
            raise ValueError("No target adapters")
        # Independent decoder RNG gives identical U blocks for structure ablations.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(decoder_seed)
            U = torch.empty(self.pB, q)
            if init == "gaussian":
                nn.init.normal_(U, std=1 / math.sqrt(q))
            elif init == "xavier":
                nn.init.xavier_normal_(U, gain=1)
            elif init == "orthogonal_blocks":
                offset = 0
                for size in sizes:
                    if size < q:
                        raise ValueError("Column-orthogonal block requires pB_layer >= q")
                    nn.init.orthogonal_(U[offset:offset + size], gain=1)
                    offset += size
            else:
                raise ValueError(init)
        if mapping == "full":
            self.M = nn.Parameter(torch.zeros(self.pB, m))
        elif mapping == "layer_specific":
            self.V_layers = nn.ParameterList([nn.Parameter(torch.zeros(q, m)) for _ in specs])
            self.register_buffer("U", U)
        else:
            self.V = nn.Parameter(torch.zeros(q, m))
            if mapping == "learnable_decoder":
                self.U = nn.Parameter(U)
            else:
                self.register_buffer("U", U)

    def forward(self, z):
        if self.mapping == "full":
            flat = F.linear(z, self.M)
            parts = flat.split(self.sizes, dim=-1)
        elif self.mapping == "layer_specific":
            U_parts = self.U.split(self.sizes, dim=0)
            parts = [F.linear(F.linear(z, V), U) for V, U in zip(self.V_layers, U_parts)]
        else:
            flat = F.linear(F.linear(z, self.V), self.U)
            parts = flat.split(self.sizes, dim=-1)
        return OrderedDict((key, part.reshape(z.shape[0], *shape)) for (key, shape), part in zip(self.specs, parts))

