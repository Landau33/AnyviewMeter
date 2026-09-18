"""The three pose-injection tiers.  Selected by ``model.pose.injector`` in config.

Read ``base.py`` first for why there are three of these rather than one.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .base import PluckerEncoder, PoseInjector, register_injector, zero_linear


# --------------------------------------------------------------------- tier A
@register_injector("patch_add")
class PatchAddInjector(PoseInjector):
    """Tier A -- zero-init Plucker features added onto patch tokens.

    This is the injection VD3D found insufficient for camera control, and it is
    here as the negative control rather than as a candidate.  If tiers B and C do
    not clear it, the "simple injection is not enough" claim has no evidence and
    the method section should say so.

    The only pathway is ``patch <- patch + W(plucker)`` with ``W`` zero-initialised,
    so at step 0 the backbone is untouched.
    """

    def __init__(self, hidden_dim: int, pose_dim: int, n_freqs: int = 6,
                 encoder_hidden: Optional[int] = None, **_):
        super().__init__(hidden_dim, pose_dim)
        self.encoder = PluckerEncoder(hidden_dim, n_freqs=n_freqs, hidden=encoder_hidden)
        # zero-init the encoder's output so the whole branch starts silent
        nn.init.zeros_(self.encoder.out.weight)
        nn.init.zeros_(self.encoder.out.bias)
        self.last_rel = 0.0

    def inject_patches(self, patch_embeds: torch.Tensor, plucker: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(plucker)                       # (B, h*w, D)
        if feats.shape[1] != patch_embeds.shape[1]:
            raise ValueError(
                f"Plucker grid has {feats.shape[1]} tokens but the backbone produced "
                f"{patch_embeds.shape[1]} patch tokens; the map must be built on the "
                "post-merge token grid (see geometry.plucker.plucker_map)")
        with torch.no_grad():
            # ||added|| / ||host||.  Tier A has no gate, so this scalar is the only way
            # to see whether the branch grew into the backbone or zeroed itself back
            # out: a zero-init encoder that stays at 0 looks exactly like a closed gate.
            self.last_rel = float(feats.norm() / patch_embeds.norm().clamp_min(1e-8))
        return patch_embeds + feats.to(patch_embeds.dtype)

    def injection_ratios(self) -> dict:
        return {"patch": self.last_rel}

    def describe(self) -> str:
        return (f"A/patch_add: zero-init Plucker -> patch tokens "
                f"({self.trainable_parameter_count()/1e6:.2f}M params)")


# --------------------------------------------------------------------- tier B
@register_injector("cam_token")
class CamTokenInjector(PatchAddInjector):
    """Tier B -- tier A plus a global ``<cam>`` register token per frame.

    Rationale (UniScale does essentially this split): rays are a *local* signal and
    belong on patch tokens, but "which viewpoint am I looking from" is a *global*
    fact the LLM wants in one place.  Forcing both through the patch channel makes
    pose compete with image content; a register token gives it its own slot.

    The register embedding is NOT zero-initialised -- a zeroed register token would
    be indistinguishable from padding and carry no gradient signal early on.
    Instead it is small-init, and it enters the sequence at a position the collator
    reserves, so it cannot overwrite any image token.
    """

    def __init__(self, hidden_dim: int, pose_dim: int, n_freqs: int = 6,
                 encoder_hidden: Optional[int] = None, pose_hidden: int = 256,
                 pose_n_freqs: int = 4, **_):
        super().__init__(hidden_dim, pose_dim, n_freqs=n_freqs, encoder_hidden=encoder_hidden)
        self.pose_n_freqs = pose_n_freqs
        in_dim = pose_dim * (1 + 2 * pose_n_freqs)
        self.pose_mlp = nn.Sequential(
            nn.Linear(in_dim, pose_hidden),
            nn.GELU(),
            nn.Linear(pose_hidden, pose_hidden),
            nn.GELU(),
            nn.Linear(pose_hidden, hidden_dim),
        )
        self.pose_norm = nn.LayerNorm(hidden_dim)
        # small (not zero) init: a zero register token is just padding
        nn.init.normal_(self.pose_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.pose_mlp[-1].bias)

    @property
    def emits_cam_token(self) -> bool:
        return True

    def _fourier_pose(self, v: torch.Tensor) -> torch.Tensor:
        feats = [v]
        for i in range(self.pose_n_freqs):
            f = (2.0 ** i) * torch.pi
            feats += [torch.sin(f * v), torch.cos(f * v)]
        return torch.cat(feats, dim=-1)

    def cam_token_embedding(self, pose_vec: torch.Tensor) -> torch.Tensor:
        return self.pose_norm(self.pose_mlp(self._fourier_pose(pose_vec)))

    def describe(self) -> str:
        return (f"B/cam_token: A + global <cam> register token "
                f"({self.trainable_parameter_count()/1e6:.2f}M params)")


# --------------------------------------------------------------------- tier C
class ZeroGatedCrossAttention(nn.Module):
    """One ControlNet-style cross-attention block with a zero-init output gate.

    ``h <- h + tanh(g) * Up(Attn(q=Down(h), kv=control))`` with ``g`` initialised to
    0, so the block is exactly the identity at step 0 and the model must actively
    open the gate.  The scalar gate is also a readable diagnostic: if it stays near
    zero after training, the backbone declined the pose signal -- that is the
    finding, not a bug to hide.

    BOTTLENECK.  Attending at the backbone's full width is what an adapter should
    not do: at hidden=2560 that is ~20M parameters per block, and nine blocks plus
    AdamW state does not fit next to a 4B backbone on a 12 GB card.  Projecting down
    to ``bottleneck`` first cuts it to ~3M per block, which is also the usual shape
    for adapters -- the pose signal is low-dimensional and does not need 2560
    channels to be read.

    GATE MODE.  ``"scalar"`` is everything the docstring above describes: one free
    ``tanh(g)`` per block.  ``"scale"`` replaces it with :class:`~.ngi.ScaleGate`, a
    per-channel gate produced from the clip's ray log-magnitude, because the scalar
    version is the thing we measured failing -- it closed to ~0.01 even after being
    held open for 300 steps, and pinning it by hand answered a different question
    than the one we asked.  An input-conditioned gate cannot be shut by driving one
    number to zero, and if it shuts for only some cameras that is a reading rather
    than a single bit.

    Under ``"scale"`` the zero moves to ``proj`` (the ControlNet arrangement) so the
    block is still exactly the identity at step 0 with no dead gradients: the gate
    starts open, the projection starts at zero, and ``dL/d(proj.W)`` is non-zero
    because the gate is not.  There is then no free switch anywhere in the block --
    closing it means either shutting the conditioned gate (visible in
    ``gate_span``) or zeroing the projection (visible in ``injection_ratio``), and
    both are instrumented.
    """

    def __init__(self, hidden_dim: int, control_dim: int, n_heads: int = 8,
                 dropout: float = 0.0, bottleneck: Optional[int] = None,
                 gate_init: float = 0.0, gate_mode: str = "scalar",
                 gate_hidden: int = 32, gate_bias: float = 2.0):
        super().__init__()
        d = bottleneck or min(hidden_dim, 512)
        self.bottleneck = d
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(control_dim)
        self.down = nn.Linear(hidden_dim, d, bias=False)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          kdim=control_dim, vdim=control_dim,
                                          batch_first=True)
        # EXACTLY ONE of {gate, proj} may be zero-initialised.  Zeroing both makes
        # the block start silent AND permanently dead: with g=0 and W=0,
        #     d(out)/dW    = tanh(g) * (...) = 0
        #     d(out)/dg    = sech^2(g) * W(x) = 0
        # so neither can ever move and the pathway never learns.  ControlNet
        # zero-inits only the output projection for this reason.  Here the gate
        # carries the zero (it is the readable diagnostic) and the projection gets a
        # normal small init, which keeps the block an exact identity at step 0 while
        # leaving d/d(gate) non-zero so it can open.
        self.proj = nn.Linear(d, hidden_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        # ``gate_init`` is the value of tanh(g), not of g, because tanh(g) is the
        # quantity that actually scales the branch and the thing we report.
        #
        # WHY IT MAY NEED TO BE NON-ZERO.  g=0 makes the block an exact identity, which
        # is what makes a tier comparison attributable -- but it also stalls the branch:
        # every parameter except g sits behind the factor tanh(g), so at g=0 the ONLY
        # thing receiving gradient is the scalar itself, and everything downstream then
        # learns at a rate proportional to tanh(g).  The first 400-step run ended with
        # tanh(g) ~ 0.012 and mixed signs across layers, i.e. the gates random-walked
        # rather than opened, and the 9.5M live parameters behind them trained at ~1% of
        # their nominal rate the whole time.  Starting at a small non-zero value trades
        # exact-identity-at-step-0 for a branch that can actually learn.
        if gate_mode not in ("scalar", "scale"):
            raise ValueError(f"gate_mode must be 'scalar' or 'scale', got {gate_mode!r}")
        self.gate_mode = gate_mode
        if gate_mode == "scale":
            from .ngi import ScaleGate          # local: ngi imports tiers
            # the zero moves to the projection, so exactly one factor is still zero
            nn.init.zeros_(self.proj.weight)
            self.scale_gate = ScaleGate(hidden_dim, hidden=gate_hidden,
                                        gate_bias=gate_bias)
            self.gate = None
        else:
            g0 = math.atanh(max(-0.999, min(0.999, float(gate_init))))
            self.gate = nn.Parameter(torch.full((1,), g0))
            self.scale_gate = None
        self.last_rel = 0.0

    def forward(self, hidden: torch.Tensor, control: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                s: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.down(self.norm_q(hidden))
        kv = self.norm_kv(control)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                           need_weights=False)
        if self.scale_gate is not None:
            if s is None:
                raise ValueError(
                    "gate_mode='scale' needs the ray log-magnitude s; the caller must "
                    "pass it rather than let the gate silently fall back to a constant")
            delta = self.scale_gate(s) * self.proj(out)
        else:
            delta = torch.tanh(self.gate) * self.proj(out)
        with torch.no_grad():
            # An OPEN gate only says the injection is permitted.  The block can still
            # make itself irrelevant by driving proj to zero, which from the outside is
            # indistinguishable from a closed gate -- this separates the two.
            self.last_rel = float(delta.norm() / hidden.norm().clamp_min(1e-8))
        return hidden + delta

    @torch.no_grad()
    def gate_value(self) -> float:
        """One number for the log line.  For the conditioned gate it is the mean --
        read ``gate_span`` beside it, since a mean alone cannot tell a gate that
        learned nothing from one that learned to sit half open."""
        if self.scale_gate is not None:
            return float(self.scale_gate.last.get("gate_mean", float("nan")))
        return float(torch.tanh(self.gate).item())

    @torch.no_grad()
    def gate_telemetry(self) -> dict:
        if self.scale_gate is not None:
            return dict(self.scale_gate.last)
        return {"gate_mean": self.gate_value()}


@register_injector("cross_attn")
class CrossAttnInjector(CamTokenInjector):
    """Tier C -- tier B plus a ControlNet-like cross-attention side branch.

    A small transformer encodes the Plucker map (plus the pose descriptor as an
    extra control token) into ``n_control`` control features.  Selected LLM layers
    then cross-attend to them through zero-gated blocks.

    ``layers`` names which backbone layers get a block.  Injecting into every layer
    is wasteful and destabilising; the VD3D-style recipe is a handful of blocks
    spread through the depth, which is what ``layer_fraction`` produces by default.
    """

    def __init__(self, hidden_dim: int, pose_dim: int, n_freqs: int = 6,
                 encoder_hidden: Optional[int] = None, pose_hidden: int = 256,
                 pose_n_freqs: int = 4, control_dim: int = 512, n_control_layers: int = 2,
                 n_heads: int = 8, n_backbone_layers: int = 36,
                 layer_fraction: float = 0.25, layers: Optional[list] = None,
                 bottleneck: Optional[int] = None, gate_init: float = 0.0,
                 gate_mode: str = "scalar", gate_hidden: int = 32,
                 gate_bias: float = 2.0, **_):
        super().__init__(hidden_dim, pose_dim, n_freqs=n_freqs,
                         encoder_hidden=encoder_hidden, pose_hidden=pose_hidden,
                         pose_n_freqs=pose_n_freqs)
        self.control_dim = control_dim
        self.gate_mode = gate_mode
        #: mean ray log-magnitude of the last control batch -- what the conditioned
        #: gate reads.  Cached rather than threaded through the hook signature,
        #: because the forward hooks the model installs cannot carry extra arguments.
        self._last_s: Optional[torch.Tensor] = None

        # side-branch encoder: Plucker tokens + one pose token -> control features
        self.control_plucker = PluckerEncoder(control_dim, n_freqs=n_freqs)
        self.control_pose = nn.Linear(pose_dim * (1 + 2 * pose_n_freqs), control_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=control_dim, nhead=n_heads, dim_feedforward=control_dim * 4,
            batch_first=True, norm_first=True, dropout=0.0)
        self.control_trunk = nn.TransformerEncoder(enc_layer, num_layers=n_control_layers)
        self.control_norm = nn.LayerNorm(control_dim)

        if layers is None:
            step = max(1, int(round(1.0 / max(layer_fraction, 1e-6))))
            layers = list(range(step - 1, n_backbone_layers, step))
        self.layers = list(layers)
        self.blocks = nn.ModuleDict({
            str(i): ZeroGatedCrossAttention(hidden_dim, control_dim, n_heads=n_heads,
                                            bottleneck=bottleneck, gate_init=gate_init,
                                            gate_mode=gate_mode, gate_hidden=gate_hidden,
                                            gate_bias=gate_bias)
            for i in self.layers
        })

    @property
    def uses_cross_attention(self) -> bool:
        return True

    def control_features(self, plucker: torch.Tensor, pose_vec: torch.Tensor) -> torch.Tensor:
        tok = self.control_plucker(plucker)                       # (B, h*w, C)
        pose_tok = self.control_pose(self._fourier_pose(pose_vec))[:, None, :]
        x = torch.cat([pose_tok, tok], dim=1)                     # pose token first
        # ONE s PER BATCH, not per token.  Tier C's blocks act on the LLM's hidden
        # states, which have no per-token ray -- the conditioned gate can only ask
        # "which camera is this", so it gets the clip's mean ray log-magnitude.
        # Tier D, which does have a ray per token, gates per token instead.
        m = plucker[:, 3:6]
        self._last_s = m.norm(dim=1).clamp_min(1e-6).log().mean().reshape(1, 1)
        return self.control_norm(self.control_trunk(x))

    def apply_to_layer(self, layer_idx: int, hidden: torch.Tensor,
                       control: torch.Tensor) -> torch.Tensor:
        key = str(layer_idx)
        if key not in self.blocks:          # nn.ModuleDict has no .get()
            return hidden
        s = None
        if self.gate_mode == "scale":
            if self._last_s is None:
                raise RuntimeError(
                    "apply_to_layer ran before control_features under gate_mode="
                    "'scale'; the conditioned gate has no s to read")
            s = self._last_s.to(hidden.dtype).to(hidden.device)
        return self.blocks[key](hidden, control, s=s)

    def gate_values(self) -> dict:
        """Per-layer gate magnitudes -- the honest read on whether pose got used."""
        return {int(i): b.gate_value() for i, b in self.blocks.items()}

    def injection_ratios(self) -> dict:
        """Per-site ||injected|| / ||host|| from the most recent forward.

        Keyed by str so the tier-A patch site sits in the same dict as the numbered
        cross-attention layers.
        """
        out = {str(i): b.last_rel for i, b in self.blocks.items()}
        out["patch"] = self.last_rel          # inherited from PatchAddInjector
        return out

    def gate_telemetry(self) -> dict:
        out = {}
        for i, b in self.blocks.items():
            for k, v in b.gate_telemetry().items():
                out[f"L{i}/{k}"] = v
        return out

    def describe(self) -> str:
        gm = "" if self.gate_mode == "scalar" else " [scale-conditioned gate]"
        return (f"C/cross_attn: B + zero-gated cross-attn at layers {self.layers}{gm} "
                f"({self.trainable_parameter_count()/1e6:.2f}M params)")
