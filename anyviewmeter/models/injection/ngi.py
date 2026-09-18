"""Tier D -- rays as a second positional coordinate on the attention scores.

WHERE THIS CAME FROM.  Tiers A/B/C all inject pose into the *value stream*: they
change what a token's feature vector says.  SCoPE (arXiv 2606.27345, v1 as
"RayPE") injects into the *address space* instead -- the ray is added to the
pretrained attention's queries and keys, so camera geometry becomes part of how
tokens find each other rather than part of what they contain.

WHY THAT IS WORTH A TIER RATHER THAN A FOOTNOTE.  Expanding the score,

    <q_i, k_j> = <Wq x_i, Wk x_j>          (A) content     -- unchanged
               + <Wq x_i, E_k f_j>         (B) content <- geometry
               + <E_q f_i, Wk x_j>         (C) geometry -> content
               + <E_q f_i, E_k f_j>        (D) pure geometry

and SCoPE's ablation reports that deleting (B)+(C) costs more than any other
component, with the claim that this coupling "cannot be structurally represented
by V-side, cross-attention, AdaLN, or rotation-based approaches".  Our tier C is
exactly a V-side cross-attention, and it is the tier we measured failing: the
gate closed, the causal check was flat, and a branch fed zeros matched it.  So
this is the one structural form we had not tried, and it has an independent
argument for why the forms we did try were the wrong shape.

THE FLIP.  The query is built from ``(d, m_hat, s)`` and the key from the
swapped ``(m_hat, d, s)``.  With ``E_q = E_k = [I_6 | 0]`` term (D) is then

    d_i . m_j + m_i . d_j

which is the Plucker reciprocal product: bilinear (so it fits an inner product
exactly), SE(3)-invariant, and zero precisely when the two lines of sight meet.
That last property is why :class:`RayPE` feeds ``E`` the RAW moment by default --
see its docstring; normalising the moment costs both the zero and the invariance.
That is the second borrowed idea -- an initialisation that is not zero but
*geometrically correct*, so the branch starts at a known-good point instead of at
the origin.  See :class:`RayPE` for why this is the way out of the deadlock
documented in ``tiers.py``.

ONE DEGENERACY YOU MUST KNOW BEFORE READING ANY TIER-D NUMBER.  Two rays that
share a camera centre always meet, so their reciprocal product is IDENTICALLY
ZERO -- for any two patches of the same view, at any timestep.  Our sequences are
single-camera, so term (D) contributes a constant 0 and only (B)+(C) can do any
work.  ``tests/test_ngi.py`` pins this.  Making (D) non-trivial requires packing
several cameras of the same trajectory into one attention sequence; the data
already has 25 cameras per trajectory, so that is a collator change, not a
re-render.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from anyviewmeter.geometry.plucker import NGI_FEATURE_DIM, normalize_moment
from .base import PoseInjector, register_injector
from .tiers import CamTokenInjector


class ScaleGate(nn.Module):
    """Per-channel gate driven by the ray's log-magnitude: ``g = sigmoid(G(s))``.

    THE POINT IS THAT IT IS NOT A FREE SCALAR.  Our tier-C result was that a
    learned scalar gate closes -- ``tanh(g)`` ended at 0.009-0.020 even after being
    held open for 300 steps -- and the only way we could measure the branch at all
    was to pin the gate by hand (``--gate-open``).  Pinning answers a different
    question than the one we asked: it removes the switch instead of making the
    switch informative.

    A gate that is a FUNCTION OF THE INPUT cannot be closed by driving one number
    to zero; shutting it off means learning an MLP that outputs large negatives for
    every ``s`` in the data.  That is strictly harder, and -- more useful -- if it
    closes only for *some* ``s`` that is a readable finding about which camera
    distances the geometry helps at, rather than a single bit saying "the optimiser
    preferred off".

    INIT: ``lin2.weight`` is the zero, and it has to be the OUTPUT one.  The gate
    must start out both constant across tokens and uniform across channels, because
    :class:`RayPE`'s geometric initialisation is a claim about an inner product:
    ``<g * x_i, g * y_j> = sum_c g_c^2 x_ic y_jc`` is the reciprocal product only if
    ``g`` is the same in every channel, and a per-channel gate at random init
    silently turns term (D) into a *channel-weighted* reciprocal product instead
    (measured: 40% spread in the ratio, which is what ``test_ngi.py`` caught the
    first time this class was written with a small random ``lin2``).

    Zeroing the output layer gives ``g = sigmoid(gate_bias)`` exactly, in every
    channel, for every token.  ``lin1`` is small-but-nonzero so its output already
    varies with ``s``; that keeps ``d g / d(lin2.weight)`` non-zero, so the output
    layer moves on the first step and ``lin1`` revives immediately after.  This is
    the ControlNet arrangement and the same house rule as ``tiers.py``: exactly one
    factor in a product may be zero, never both.

    The gate opens rather than closes at init (sigmoid(2) ~ 0.88): identity-at-
    initialisation is carried by :class:`RayPE`'s ``alpha``, and making the gate
    start shut as well would be the two-zeros mistake in a different place.
    """

    def __init__(self, dim: int, hidden: int = 32, gate_bias: float = 2.0):
        super().__init__()
        self.lin1 = nn.Linear(1, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        nn.init.normal_(self.lin1.weight, std=0.5)
        nn.init.normal_(self.lin1.bias, std=0.5)
        nn.init.zeros_(self.lin2.weight)
        nn.init.constant_(self.lin2.bias, gate_bias)
        self.last: Dict[str, float] = {}

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """``(..., 1)`` log-magnitude -> ``(..., dim)`` gate in (0, 1)."""
        g = torch.sigmoid(self.lin2(torch.nn.functional.gelu(self.lin1(s))))
        with torch.no_grad():
            # SPREAD IS THE DIAGNOSTIC.  ``mean`` alone cannot distinguish "the gate
            # learned nothing and sits at its init" from "the gate learned to be
            # uniformly half open"; ``span`` (max - min across tokens) says whether
            # it is doing anything conditional at all.
            self.last = {"gate_mean": float(g.mean()),
                         "gate_min": float(g.min()), "gate_max": float(g.max()),
                         "gate_span": float(g.amax(dim=-1).max() - g.amin(dim=-1).min())}
        return g


class RayPE(nn.Module):
    """Ray-space positional encoding for one attention layer: NGI feature -> (pe_q, pe_k).

    Normalize-Gate-Inject, following SCoPE:

        f_q = (d, m_hat, s)          f_k = (m_hat, d, s)          <- the flip
        pe  = g(s) * RMSNorm(E f)
        q  += alpha * pe_q           k  += alpha * pe_k

    GEOMETRIC INITIALISATION, and why it is not just a nicer zero.  ``tiers.py``
    records a real bind: a new pathway must be the identity at step 0 for the tier
    comparison to be attributable, but zero-initialising the thing that scales it
    starves everything behind it (tier C's gates random-walked at ~0.012 while
    9.5M parameters trained at 1% of their nominal rate).  Our workaround was to
    start the gate at 0.05 and give up exact identity.

    Here the two roles come apart.  ``alpha`` carries the zero -- so the layer is
    bit-exactly the pretrained attention at step 0 -- while ``E_q`` and ``E_k``
    start at ``[I_6 | 0]``, which is not a small random point but the exact
    Plucker reciprocal product.  ``d(out)/d(alpha) = pe != 0``, so alpha lifts off
    immediately; ``E`` is momentarily frozen, but it is frozen AT THE RIGHT ANSWER
    rather than at noise.  That is the difference between "silent and dead" and
    "silent and correct".

    RMSNorm does not break the claim.  At geometric init every token's raw ``E f``
    has squared norm exactly 2 (two unit vectors, zeros elsewhere), so the norm is
    a CONSTANT rescale, not a per-token one -- term (D) stays exactly proportional
    to the reciprocal product.  The same argument is why :class:`ScaleGate` must be
    constant across tokens at init.  ``tests/test_ngi.py`` measures the
    proportionality rather than trusting this paragraph.

    WHICH MOMENT GOES INTO ``E``, and why the default departs from SCoPE.  The
    coplanarity property -- term (D) is zero exactly when the two lines of sight meet
    -- holds for the RAW moment.  It does not survive normalisation: with ``m_hat``
    the term becomes ``d_i.m_j/||m_j|| + d_j.m_i/||m_i||``, which is not zero for
    intersecting rays unless the two magnitudes happen to match.  Measured on one of
    our own cameras: the same-camera term is 3e-8 with the raw moment and 3.9e-1 with
    the unit one.  Normalising also costs SE(3) invariance, since ``||m||`` depends on
    where the world origin sits.

    So ``moment_mode="raw"`` is the default: ``E`` reads ``(d, m, s)`` and keeps the
    geometry exact, while ``s`` still goes to the gate, which is where the scale
    information is actually wanted.  The scale argument for normalising does not apply
    to us -- ``scripts/moment_scale_report.py`` measured every test group inside the
    training range on both datasets, with the camera moving ``s`` by ~25% of what
    perspective does inside one frame.  ``moment_mode="unit"`` is the SCoPE-faithful
    variant, kept because a cross-dataset run would need it and because the difference
    is exactly the kind of thing that should be a flag rather than a silent choice.

    RMSNorm rescales each token by a POSITIVE factor, so under ``"raw"`` term (D) is
    ``recip_ij / (||x_i|| ||x_j||)``: the zero set and the sign of the reciprocal
    product survive, the exact proportionality does not.  That is the claim
    ``test_ngi.py`` checks.

    HEADS.  One ``pe`` is shared by every attention head, scaled by one ``alpha``
    per layer.  Per-head encodings were the alternative; they were dropped because
    (a) grouped-query attention gives q and k different head counts, so "the same
    ray coordinate" would have to mean two different things on the two sides, and
    (b) one alpha per layer is directly comparable to the one ``tanh(g)`` per layer
    that tier C already reports, and a diagnostic we can read beats a little extra
    capacity.
    """

    def __init__(self, head_dim: int, gate_hidden: int = 32, gate_bias: float = 2.0,
                 geometric_init: bool = True, alpha_init: float = 0.0,
                 use_gate: bool = True, moment_mode: str = "raw"):
        super().__init__()
        if moment_mode not in ("raw", "unit"):
            raise ValueError(f"moment_mode must be 'raw' or 'unit', got {moment_mode!r}")
        self.moment_mode = moment_mode
        if head_dim < 6:
            raise ValueError(
                f"head_dim {head_dim} < 6: the geometric initialisation needs six "
                "channels to hold (d, m_hat), so tier D cannot be placed on this "
                "attention without a projection that would destroy the identity")
        self.head_dim = head_dim
        self.E_q = nn.Linear(NGI_FEATURE_DIM, head_dim, bias=False)
        self.E_k = nn.Linear(NGI_FEATURE_DIM, head_dim, bias=False)
        if geometric_init:
            for E in (self.E_q, self.E_k):
                with torch.no_grad():
                    E.weight.zero_()
                    # rows 0..5 read (d, m_hat) straight through; the s channel is
                    # left to the gate, which is where its information belongs
                    E.weight[:6, :6] = torch.eye(6)
        else:
            for E in (self.E_q, self.E_k):
                nn.init.normal_(E.weight, std=0.02)
        self.norm_q = nn.RMSNorm(head_dim)
        self.norm_k = nn.RMSNorm(head_dim)
        self.gate = ScaleGate(head_dim, hidden=gate_hidden, gate_bias=gate_bias) \
            if use_gate else None
        #: zero-init: the layer is bit-exactly the pretrained attention at step 0
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.last_rel = 0.0

    @staticmethod
    def features(plucker: torch.Tensor) -> torch.Tensor:
        """``(B, 6, h, w)`` Plucker map -> ``(B, h*w, 7)`` NGI feature ``(d, m_hat, s)``."""
        f = normalize_moment(plucker)                       # (B, 7, h, w)
        b, c, h, w = f.shape
        return f.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(..., 7)`` NGI feature -> ``(pe_q, pe_k)``, each ``(..., head_dim)``.

        Already scaled by ``alpha``; the caller only has to add them.
        """
        if feat.shape[-1] != NGI_FEATURE_DIM:
            raise ValueError(f"expected {NGI_FEATURE_DIM} NGI channels, got {feat.shape[-1]}")
        d, m_hat, s = feat[..., :3], feat[..., 3:6], feat[..., 6:]
        # "raw" rebuilds m = m_hat * exp(s) for the E path, which is what keeps the
        # coplanarity zero; s still reaches the gate either way.
        m = m_hat * s.exp() if self.moment_mode == "raw" else m_hat
        f_q = torch.cat([d, m, s], dim=-1)
        f_k = torch.cat([m, d, s], dim=-1)                  # the flip
        pe_q = self.norm_q(self.E_q(f_q))
        pe_k = self.norm_k(self.E_k(f_k))
        if self.gate is not None:
            g = self.gate(s)
            pe_q, pe_k = g * pe_q, g * pe_k
        return self.alpha * pe_q, self.alpha * pe_k

    @torch.no_grad()
    def alpha_value(self) -> float:
        return float(self.alpha.item())

    @torch.no_grad()
    def telemetry(self) -> Dict[str, float]:
        out = {"alpha": self.alpha_value(), "rel": self.last_rel}
        if self.gate is not None:
            out.update(self.gate.last)
        return out


def apply_ray_pe(q: torch.Tensor, k: torch.Tensor, pe_q: torch.Tensor,
                 pe_k: torch.Tensor, ray_mask: Optional[torch.Tensor] = None,
                 module: Optional[RayPE] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add a shared-across-heads ray PE to ``(B, H, T, head_dim)`` q/k tensors.

    ``pe_q`` / ``pe_k`` are ``(B, T, head_dim)``; ``ray_mask`` is ``(B, T)`` and is
    what keeps TEXT positions bit-exact.  A text token has no ray, and letting one
    through with a zero feature is not the same thing as leaving it alone --
    ``RMSNorm(0)`` is only zero up to its epsilon, and the gate bias would still
    admit a constant.  Masking says what we mean.

    MUST BE CALLED AFTER RoPE, not before.  RoPE rotates q and k by their positions,
    and ``<R_i q_i, R_j k_j> = <q_i, R_{j-i} k_j>`` -- a PE added before the rotation
    would come out multiplied by a relative-position rotation, which is exactly the
    thing the reciprocal product must not depend on.  Adding after keeps RoPE
    bit-exact on the content term and leaves term (D) purely geometric.
    """
    if ray_mask is not None:
        m = ray_mask.to(pe_q.dtype)[..., None]
        pe_q, pe_k = pe_q * m, pe_k * m
    pq = pe_q[:, None].to(q.dtype)
    pk = pe_k[:, None].to(k.dtype)
    if module is not None:
        with torch.no_grad():
            # Same role as tier C's injection_ratio: alpha only says the injection is
            # PERMITTED.  This says how large it actually is next to the content term,
            # which is the number that separates "open" from "open and irrelevant".
            module.last_rel = float(pq.norm() / q.norm().clamp_min(1e-8))
    return q + pq, k + pk


@register_injector("qk_pe")
class QKRayPEInjector(CamTokenInjector):
    """Tier D -- tier B, plus a ray positional encoding on every injected layer's Q/K.

    Inherits tier B rather than standing alone so the ablation ladder stays a
    ladder: D differs from C only in WHERE the same geometry enters, and both sit
    on top of the same patch-add and ``<cam>`` register token.  ``patch_add=False``
    turns the tier-A pathway off for a clean "Q/K only" arm.
    """

    def __init__(self, hidden_dim: int, pose_dim: int, head_dim: int = 128,
                 n_freqs: int = 6, encoder_hidden: Optional[int] = None,
                 pose_hidden: int = 256, pose_n_freqs: int = 4,
                 n_backbone_layers: int = 36, layer_fraction: float = 0.25,
                 layers: Optional[List[int]] = None, gate_hidden: int = 32,
                 gate_bias: float = 2.0, geometric_init: bool = True,
                 alpha_init: float = 0.0, use_gate: bool = True,
                 patch_add: bool = True, moment_mode: str = "raw", **_):
        super().__init__(hidden_dim, pose_dim, n_freqs=n_freqs,
                         encoder_hidden=encoder_hidden, pose_hidden=pose_hidden,
                         pose_n_freqs=pose_n_freqs)
        self.head_dim = head_dim
        self.patch_add = patch_add
        if layers is None:
            step = max(1, int(round(1.0 / max(layer_fraction, 1e-6))))
            layers = list(range(step - 1, n_backbone_layers, step))
        self.layers = list(layers)
        self.pe = nn.ModuleDict({
            str(i): RayPE(head_dim, gate_hidden=gate_hidden, gate_bias=gate_bias,
                          geometric_init=geometric_init, alpha_init=alpha_init,
                          use_gate=use_gate, moment_mode=moment_mode)
            for i in self.layers})
        self.moment_mode = moment_mode

    @property
    def uses_qk_pe(self) -> bool:
        return True

    def inject_patches(self, patch_embeds: torch.Tensor, plucker: torch.Tensor):
        if not self.patch_add:
            return patch_embeds
        return super().inject_patches(patch_embeds, plucker)

    def ray_features(self, plucker: torch.Tensor) -> torch.Tensor:
        """``(B, 6, h, w)`` -> ``(B, h*w, 7)``.  See :meth:`RayPE.features`."""
        return RayPE.features(plucker)

    def qk_pe(self, layer_idx: int, feat: torch.Tensor):
        """``(pe_q, pe_k)`` for one layer, or ``None`` if that layer is not injected."""
        key = str(layer_idx)
        if key not in self.pe:
            return None
        return self.pe[key](feat)

    def alpha_values(self) -> Dict[int, float]:
        """Per-layer alpha -- tier D's answer to tier C's ``tanh(g)``."""
        return {int(i): p.alpha_value() for i, p in self.pe.items()}

    def injection_ratios(self) -> Dict[str, float]:
        out = {str(i): p.last_rel for i, p in self.pe.items()}
        if self.patch_add:
            out["patch"] = self.last_rel
        return out

    def gate_telemetry(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, p in self.pe.items():
            for k, v in p.telemetry().items():
                out[f"L{i}/{k}"] = v
        return out

    def describe(self) -> str:
        bits = f"Q/K ray PE ({self.moment_mode} moment) at layers {self.layers}"
        if self.patch_add:
            bits = "B + " + bits
        return (f"D/qk_pe: {bits} "
                f"({self.trainable_parameter_count()/1e6:.2f}M params)")
