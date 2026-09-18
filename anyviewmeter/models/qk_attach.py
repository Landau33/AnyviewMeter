"""Attaching tier D to a real backbone: the ray PE has to land AFTER RoPE.

WHY THIS NEEDS ITS OWN FILE.  Tiers A and C attach with plain forward hooks -- they
read or rewrite a module's output and the backbone's own code never changes.  Tier
D cannot: the ray PE must be added to the queries and keys *after* the rotary
embedding has been applied and *before* the attention scores are formed, and that
moment does not exist as a module boundary.  A hook on ``q_proj`` fires too early
(RoPE would then rotate the PE, making term (D) depend on relative token position,
which is exactly the thing a geometric term must not do) and a hook on the
attention block fires too late.

So the layer's ``forward`` is replaced with a copy that is otherwise line-for-line
the original, with two additions.  That is a real cost -- it pins us to the
attention implementation of the installed transformers version -- and it is paid
deliberately rather than worked around, because every cheaper attachment point
changes what the method IS.  :func:`attach_ray_pe` checks the module surface it
depends on and raises if anything is missing, rather than silently attaching to
something whose forward does something else.
"""
from __future__ import annotations

import types
from typing import List, Optional

import torch
import torch.nn as nn

from anyviewmeter.models.injection.ngi import apply_ray_pe


class RayPEState:
    """What the patched forwards read.  One per model.

    Set ``features`` to ``(B, T_seq, 7)`` and ``mask`` to ``(B, T_seq)`` before the
    forward pass; set ``features = None`` to make every patched layer fall straight
    through to the original behaviour, which is what the pose-blind control and the
    ``zero`` arm of the causal check need.
    """

    def __init__(self):
        self.features: Optional[torch.Tensor] = None
        self.mask: Optional[torch.Tensor] = None

    def clear(self):
        self.features = None
        self.mask = None


def ray_mask_and_features(seq_features: torch.Tensor, visual_positions: torch.Tensor,
                          seq_len: int) -> tuple:
    """Scatter per-visual-token ray features into a full-sequence tensor.

    ``seq_features`` is ``(N_visual, 7)`` in image order, ``visual_positions`` is the
    ``(N_visual,)`` index of each of those tokens in the LLM sequence.  Text
    positions come back masked, not zero-filled: a zero NGI feature is still a
    feature (``RMSNorm(0)`` is only zero up to epsilon, and a gate bias would still
    admit a constant), whereas the mask makes them bit-exact.
    """
    n = int(visual_positions.numel())
    if seq_features.shape[0] != n:
        raise ValueError(
            f"{seq_features.shape[0]} ray features for {n} visual positions.  These "
            "must correspond one-to-one and in the same order, or the geometry is "
            "silently attached to the wrong pixels.")
    feats = seq_features.new_zeros(1, seq_len, seq_features.shape[-1])
    mask = seq_features.new_zeros(1, seq_len)
    feats[0, visual_positions] = seq_features
    mask[0, visual_positions] = 1.0
    return feats, mask


def _patched_forward(orig_cls_forward, layer_idx: int, injector, state: RayPEState):
    """Build the replacement ``forward`` for one attention module."""

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb, eager_attention_forward)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if state.features is None:
            return orig_cls_forward(self, hidden_states, position_embeddings,
                                    attention_mask, past_key_values=past_key_values,
                                    cache_position=cache_position, **kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- the whole reason this file exists: after RoPE, before the scores ----
        # The injector is held in fp32 while the trunk runs bf16, so the FEATURES must
        # not be cast down to q's dtype -- that pushes bf16 into an fp32 Linear and
        # raises.  Compute the PE in the injector's dtype and cast the result instead,
        # which also keeps the reciprocal product out of bf16's 8-bit mantissa.
        module = injector.pe[str(layer_idx)]
        pdt = next(module.parameters()).dtype
        pe = injector.qk_pe(layer_idx, state.features.to(pdt))
        if pe is not None:
            pe = (pe[0].to(q.dtype), pe[1].to(q.dtype))
            q, k = apply_ray_pe(q, k, pe[0], pe[1], ray_mask=state.mask, module=module)
        # -------------------------------------------------------------------------

        if past_key_values is not None:
            k, v = past_key_values.update(
                k, v, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})

        attn_fn = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        out, weights = attn_fn(self, q, k, v, attention_mask,
                               dropout=0.0 if not self.training else self.attention_dropout,
                               scaling=self.scaling, **kwargs)
        return self.o_proj(out.reshape(*input_shape, -1).contiguous()), weights

    return forward


_REQUIRED = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
             "head_dim", "scaling", "config")


def attach_ray_pe(layers: List[nn.Module], injector, state: RayPEState) -> int:
    """Replace ``self_attn.forward`` on each injected layer.  Returns the count.

    Raises rather than skipping when a layer does not have the surface the patched
    forward assumes.  A tier-D run that silently attached to nothing would look
    exactly like a tier-D run whose geometry did not help, and those are the two
    outcomes this whole line of work exists to tell apart.
    """
    n = 0
    for idx in injector.layers:
        if idx >= len(layers):
            continue
        attn = getattr(layers[idx], "self_attn", None)
        if attn is None:
            raise RuntimeError(f"layer {idx} has no .self_attn to attach tier D to")
        missing = [a for a in _REQUIRED if not hasattr(attn, a)]
        if missing:
            raise RuntimeError(
                f"layer {idx}'s attention ({type(attn).__name__}) is missing "
                f"{missing}; the tier-D forward was written against "
                "Qwen3VLTextAttention and must be updated rather than attached "
                "to a module whose forward does something else")
        if getattr(attn, "_ray_pe_attached", False):
            continue
        attn._ray_pe_orig = type(attn).forward
        attn.forward = types.MethodType(
            _patched_forward(type(attn).forward, idx, injector, state), attn)
        attn._ray_pe_attached = True
        n += 1
    return n


def detach_ray_pe(layers: List[nn.Module]) -> int:
    n = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "_ray_pe_attached", False):
            del attn.forward
            attn._ray_pe_attached = False
            n += 1
    return n
