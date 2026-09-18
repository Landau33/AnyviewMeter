"""How camera pose enters the backbone: a common interface and four tiers.

THE PROBLEM THIS FILE EXISTS FOR.  Camera pose is a low-frequency, low-dimensional
signal competing with a very high-capacity pretrained visual stream.  VD3D's
ablation is the warning: zero-initialised Plucker features added straight onto
patch tokens produced *almost no* camera controllability, and only a ControlNet-
style cross-attention pathway made the conditioning actually bite.  So "inject
Plucker somewhere in the middle" is not a plan, it is a hypothesis -- and which
injection is strong enough for a progress model is itself a research question.

We therefore implement the tiers behind one interface, so the ablation is a config
flag rather than four forks of the model:

  A  patch_add    zero-init linear on the Plucker map, added to patch tokens.
                  The VD3D-refuted baseline.  Kept deliberately: if the stronger
                  tiers do not beat it, our claim about injection strength is
                  unsupported and we should know that early.

  B  cam_token    A, plus a global <cam> register token per frame carrying a
                  compact pose descriptor.  Mirrors UniScale's split of pose into
                  a camera token and rays into patch tokens: the register token
                  gives the LLM one place to read "which viewpoint is this" that
                  is not competing with image content for the same channel.

  C  cross_attn   B, plus a ControlNet-like side branch: a small pose encoder
                  produces control features that selected LLM layers cross-attend
                  to, through a zero-initialised output gate.  Zero-init means the
                  model starts exactly as the pretrained backbone and has to
                  actively open the gate, so any gain is attributable and training
                  cannot be destabilised at step 0.

  D  qk_pe       B, plus a ray positional encoding on each injected layer's Q and
                 K, after RoPE.  Lives in ``ngi.py``.  The only one of the four
                 that is not a write into the value stream: geometry enters the
                 ADDRESS space, so the attention score gains a term reading the two
                 rays alone.  Added after tier C failed, on the argument that a
                 V-side pathway structurally cannot represent the content<->geometry
                 coupling -- see that file's header.

Every tier starts as an identity function on the backbone at initialisation
(zero-init projections / gates).  That is a deliberate constraint: it makes the
tiers comparable, since none of them perturbs the pretrained features until
learning moves them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

_REGISTRY: Dict[str, type] = {}


def register_injector(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"injector '{name}' already registered")
        _REGISTRY[name] = cls
        cls.injector_name = name
        return cls
    return deco


def build_injector(name: str, **kwargs) -> "PoseInjector":
    if name not in _REGISTRY:
        raise KeyError(f"unknown injector '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_injectors():
    return sorted(_REGISTRY)


def zero_linear(in_dim: int, out_dim: int, bias: bool = True) -> nn.Linear:
    """A linear layer initialised to output exactly zero.

    Used everywhere a new pathway joins the pretrained stream, so the model at
    step 0 is bit-identical to the backbone.
    """
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.zeros_(lin.weight)
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


class PluckerEncoder(nn.Module):
    """Encode a (B, 6, h, w) Plucker map into per-token features (B, h*w, dim).

    Fourier features first.  Raw Plucker components are smooth and low-frequency,
    and an MLP on them tends to learn a nearly-constant map; a positional-style
    frequency expansion is the standard fix and costs almost nothing here.
    """

    def __init__(self, dim: int, n_freqs: int = 6, hidden: Optional[int] = None,
                 use_fourier: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.use_fourier = use_fourier
        in_dim = 6 * (1 + 2 * n_freqs) if use_fourier else 6
        hidden = hidden or max(dim // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, dim)
        self.in_dim = in_dim
        self.dim = dim

    def _fourier(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fourier:
            return x
        feats = [x]
        for i in range(self.n_freqs):
            f = (2.0 ** i) * torch.pi
            feats += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(feats, dim=-1)

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        b, c, h, w = plucker.shape
        if c != 6:
            raise ValueError(f"expected 6 Plucker channels, got {c}")
        x = plucker.permute(0, 2, 3, 1).reshape(b, h * w, 6)
        x = self._fourier(x)
        return self.out(self.net(x))


class PoseInjector(nn.Module, ABC):
    """Base class: how a Plucker map and a pose vector reach the backbone.

    Subclasses implement whichever hooks they need and leave the rest as no-ops:

      inject_patches   modify visual token embeddings in place in the sequence
      extra_tokens     produce additional tokens (e.g. <cam>) to splice in
      control_features produce side-branch features for cross-attention (tier C)
    """

    #: set by @register_injector
    injector_name: str = "base"

    def __init__(self, hidden_dim: int, pose_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim

    @property
    def emits_cam_token(self) -> bool:
        return False

    @property
    def uses_cross_attention(self) -> bool:
        return False

    @property
    def uses_qk_pe(self) -> bool:
        """Tier D: the injector wants a ray PE added to each layer's Q and K.

        Separate from :attr:`uses_cross_attention` because it is a different
        injection SITE, not a different amount of the same thing -- one writes into
        the value stream, the other into the address space -- and the wiring code
        has to know which hook to install.
        """
        return False

    def inject_patches(self, patch_embeds: torch.Tensor, plucker: torch.Tensor,
                       ) -> torch.Tensor:
        """(B, N, D) patch embeds + (B, 6, h, w) map -> (B, N, D).  Default: no-op."""
        return patch_embeds

    def cam_token_embedding(self, pose_vec: torch.Tensor) -> Optional[torch.Tensor]:
        """(B, pose_dim) -> (B, D) embedding for the <cam> register token."""
        return None

    def control_features(self, plucker: torch.Tensor, pose_vec: torch.Tensor,
                         ) -> Optional[torch.Tensor]:
        """(B, M, D) features the backbone may cross-attend to.  Default: none."""
        return None

    @abstractmethod
    def describe(self) -> str:
        """One line for logs and the experiment table."""

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
