"""Attach AnyviewMeter's pose injection to a real Robometer-4B checkpoint.

Robometer-4B is Qwen3-VL (patch 16, 2x2 merge, hidden 2560, 36 layers), so the
8x8 token grid AnyviewMeter builds for a 256px frame lines up with its patch tokens
without any reshaping.

DESIGN CHOICE: we reuse Robometer's OWN collator and forward pass and add nothing to
the token sequence.  Injection happens through two forward hooks:

  ``model.visual``                 output += encoded Plucker   (tier A pathway)
  ``model.language_model.layers``  zero-gated cross-attention  (tier C pathway)

The reason is comparability.  If we rebuilt the prompt to carry ``<cam>`` tokens we
would change the sequence, the position ids and the frame-extraction offsets all at
once, and any difference against baseline Robometer would no longer be attributable
to pose conditioning alone.  With hooks, the baseline and the conditioned model see
byte-identical inputs and differ only in the injected features.

Consequence: tier B's ``<cam>`` register token is NOT available here -- it needs a
token added to the tokenizer and a resized embedding table.  Tier C does not need
it: its control branch already carries the global pose descriptor as a control
token, so the pose still reaches the model, just through cross-attention instead of
through the sequence.  Tier B against Robometer is follow-up work.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from anyviewmeter.geometry.camera import CameraParams
from anyviewmeter.geometry.plucker import POSE_VECTOR_DIM, plucker_map, pose_vector
from anyviewmeter.models.injection import build_injector
from anyviewmeter.models.qk_attach import (RayPEState, attach_ray_pe, detach_ray_pe,
                                           ray_mask_and_features)


def load_robometer(model_id: str = "robometer/Robometer-4B",
                   device: Optional[torch.device] = None, dtype=torch.bfloat16):
    """Load the RBM checkpoint plus its processor/tokenizer/collator."""
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp, tok, proc, model = load_model_from_hf(model_id, device=device)
    model = model.to(dtype).eval()
    collator = setup_batch_collator(proc, tok, exp, is_eval=True)

    lc = getattr(exp, "loss", None)
    is_discrete = (getattr(lc, "progress_loss_type", "l2").lower() == "discrete") if lc else False
    n_bins = (getattr(lc, "progress_discrete_bins", None)
              or getattr(exp.model, "progress_discrete_bins", 10))
    return dict(exp=exp, tokenizer=tok, processor=proc, model=model,
                collator=collator, is_discrete=is_discrete, n_bins=n_bins,
                hidden=model.config.text_config.hidden_size
                if hasattr(model.config, "text_config") else 2560)


class PoseConditionedRobometer(nn.Module):
    """Robometer-4B with a pose-injection adapter bolted on through hooks.

    The backbone is frozen by default.  That is not only a memory decision: with the
    backbone frozen, any difference against baseline Robometer is attributable to the
    adapter rather than to general finetuning, which is the comparison we want.
    """

    def __init__(self, bundle: dict, injector_name: str = "cross_attn",
                 token_grid: int = 8, n_freqs: int = 6, pose_n_freqs: int = 4,
                 control_dim: int = 256, n_control_layers: int = 2, n_heads: int = 8,
                 layer_fraction: float = 0.25, layers: Optional[List[int]] = None,
                 bottleneck: Optional[int] = 256, patch_inject: bool = True,
                 pose_dropout: float = 0.1, freeze_backbone: bool = True,
                 gate_init: float = 0.0):
        super().__init__()
        self.rbm = bundle["model"]
        self.collator = bundle["collator"]
        self.tokenizer = bundle["tokenizer"]
        self.is_discrete = bundle["is_discrete"]
        self.n_bins = bundle["n_bins"]
        self.token_grid = token_grid
        self.pose_dropout = pose_dropout
        #: CAPACITY CONTROL.  True feeds the branch a constant (zeros) always, training
        #: AND eval, so it keeps every parameter and every bit of its injection but is
        #: told nothing about the camera.  An arm that matches its pose counterpart with
        #: this on has shown the gain was the side branch, not the geometry -- which is
        #: exactly what happened to the tier-C gate-open arm.  Distinct from
        #: pose_dropout, which only fires during training and only sometimes.
        self.pose_null = False
        # Patch injection sits at the VISION TOWER OUTPUT, so gradients have to
        # travel back through all 36 decoder layers to reach it.  Cross-attention at
        # layer L only needs the graph above L.  On a 12 GB card next to a 4B bf16
        # backbone the difference decides whether training fits at all, so the two
        # pathways are separable.
        self.patch_inject = patch_inject
        self.view_embed = None        # set by the trainer for the --view-embed arm
        # Block-causal attention for interleaved multi-view input: set per forward to the
        # number of views, None otherwise.  See _build_block_mask.
        self.block_views = None
        self.block_mask_no_own_prog = False
        self._block_mask = None
        self._lm_hook = None
        self._view_slots = None       # (F,) long, one view index per frame
        self.last_view_rel = 0.0

        hidden = bundle["hidden"]
        n_layers = len(self._layers())
        self.injector = build_injector(
            injector_name, hidden_dim=hidden, pose_dim=POSE_VECTOR_DIM,
            n_freqs=n_freqs, pose_n_freqs=pose_n_freqs, control_dim=control_dim,
            n_control_layers=n_control_layers, n_heads=n_heads,
            n_backbone_layers=n_layers, layer_fraction=layer_fraction, layers=layers,
            bottleneck=bottleneck, gate_init=gate_init)

        if freeze_backbone:
            for p in self.rbm.parameters():
                p.requires_grad = False
        self.injector.to(dtype=torch.float32)
        self._freeze_unreachable()
        self._grad_ckpt = False

        self._hooks: List = []
        self.ray_state = RayPEState()
        self._plucker: Optional[torch.Tensor] = None
        self._control: Optional[torch.Tensor] = None
        self._patch_hits = 0

    def _freeze_unreachable(self) -> dict:
        """Turn off grad for injector submodules that this path never calls.

        ``CrossAttnInjector`` inherits from the tier-A and tier-B injectors, so it
        carries their modules whether or not they run:

          ``encoder``               tier A's Plucker->patch projection.  Only reached
                                    from ``_inject_visual``, i.e. when patch_inject.
          ``pose_mlp``/``pose_norm``  tier B's <cam> register embedding.  NEVER reached
                                    here -- this path adds no tokens to the sequence.

        Measured on the tier-C config with 5 layers, that is 5.79M of 15.37M parameters
        (38%) sitting in the optimiser collecting AdamW state, receiving no gradient,
        and inflating the adapter size we report.  Freezing them changes no math; it
        just stops us paying for and quoting parameters that cannot move.
        """
        off = {}
        if not self.patch_inject and hasattr(self.injector, "encoder"):
            off["encoder"] = self.injector.encoder
        for n in ("pose_mlp", "pose_norm"):
            if hasattr(self.injector, n):
                off[n] = getattr(self.injector, n)
        frozen = {}
        for n, mod in off.items():
            c = 0
            for p in mod.parameters():
                p.requires_grad = False
                c += p.numel()
            frozen[n] = c
        return frozen

    # ------------------------------------------------------------------ hooks
    def _layers(self):
        return self.rbm.model.language_model.layers

    def _visual(self):
        return self.rbm.model.visual

    def attach(self) -> int:
        """Install the visual and layer hooks.  Returns how many were installed."""
        self.detach()

        def visual_hook(_m, _args, output):
            """Add Plucker features to whatever the vision tower returns.

            The return type is NOT stable across transformers patch releases: 4.57.1
            hands back a bare tensor, 4.57.0 hands back a
            ``BaseModelOutputWithDeepstackFeatures`` dataclass.  Handling only the
            shapes we happened to see locally is how this silently breaks on another
            machine, so all three forms are handled and anything else raises rather
            than being passed through unmodified (a pass-through would disable pose
            conditioning without any error).
            """
            if (self._plucker is None or not self.patch_inject) \
                    and self.view_embed is None:
                return output

            if torch.is_tensor(output):
                return self._inject_visual(output)

            if isinstance(output, tuple):
                return (self._inject_visual(output[0]),) + output[1:]

            # transformers ModelOutput: the merged patch tokens live in
            # last_hidden_state.  deepstack_features are separate feature maps the
            # LLM merges later and are deliberately left alone -- injecting the same
            # Plucker signal twice would double-count it.
            lhs = getattr(output, "last_hidden_state", None)
            if lhs is not None:
                output.last_hidden_state = self._inject_visual(lhs)
                return output

            raise TypeError(
                f"vision tower returned {type(output).__name__}, which this hook does "
                "not know how to inject into; extend visual_hook rather than letting "
                "pose conditioning silently become a no-op")

        self._hooks.append(self._visual().register_forward_hook(visual_hook))

        if getattr(self.injector, "uses_qk_pe", False):
            # Tier D is NOT a hook: the ray PE has to land between RoPE and the
            # attention scores, which is not a module boundary, so attach_ray_pe
            # replaces each layer's self_attn.forward.  It raises if the attention
            # surface is not the one that forward was written against, because a
            # tier-D run attached to nothing looks exactly like a tier-D run whose
            # geometry did not help.
            n_d = attach_ray_pe(self._layers(), self.injector, self.ray_state)
            if n_d != len(self.injector.layers):
                raise RuntimeError(f"tier D attached to {n_d} of "
                                   f"{len(self.injector.layers)} requested layers")

        if self.injector.uses_cross_attention:
            layers = self._layers()

            def make(idx):
                def hook(_m, _a, output):
                    if self._control is None:
                        return output
                    if isinstance(output, tuple):
                        h = self.injector.apply_to_layer(idx, output[0].float(),
                                                         self._control)
                        return (h.to(output[0].dtype),) + output[1:]
                    return self.injector.apply_to_layer(
                        idx, output.float(), self._control).to(output.dtype)
                return hook

            for i in self.injector.layers:
                if i < len(layers):
                    self._hooks.append(layers[i].register_forward_hook(make(i)))
        return len(self._hooks)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        detach_ray_pe(self._layers())
        self.ray_state.clear()

    def enable_gradient_checkpointing(self, enable: bool = True):
        """Trade compute for memory on the frozen backbone.

        Injection at the vision tower means gradients must reach back through the
        whole 36-layer stack, and storing those activations does not fit next to a
        4B bf16 backbone on a 12 GB card.  Checkpointing recomputes them instead.

        The backbone is frozen, so this only works because the injected features
        DO require grad -- the recomputed graph is rooted at the adapter, not at the
        backbone weights.
        """
        self._grad_ckpt = enable
        target = getattr(self.rbm, "model", self.rbm)
        for obj in (self.rbm, target):
            fn = getattr(obj, "gradient_checkpointing_enable" if enable
                         else "gradient_checkpointing_disable", None)
            if callable(fn):
                try:
                    fn(gradient_checkpointing_kwargs={"use_reentrant": False}) if enable else fn()
                    return True
                except TypeError:
                    fn()
                    return True
        return False

    def _inject_visual(self, emb: torch.Tensor) -> torch.Tensor:
        """Add encoded Plucker features to the merged visual embeddings.

        Qwen3-VL's visual tower returns a flat ``(total_tokens, hidden)`` stack of
        merged patch tokens for every image in the batch, in image order.  Our
        Plucker map is ``(T, 6, g, g)`` = ``T * g * g`` tokens in the same order, so
        the two line up one-to-one -- but only if the frames really produced
        ``g * g`` tokens each, which depends on the processor's resize.  A mismatch
        is a silent misalignment of geometry to pixels, so it raises.
        """
        pl = self._plucker
        if pl is None or not self.patch_inject:
            # view-embedding-only arm: no geometry at all, just a learned tag saying
            # which camera this frame came from.  It exists to separate "knowing the
            # view index" from "knowing the geometry" -- with two views interleaved
            # into one sequence, a ray map silently supplies the former as well as the
            # latter, so tier A's gain over the no-camera baseline is not attributable
            # until this arm is run.
            return self._add_view_embed(emb)
        g = self.token_grid
        want = pl.shape[0] * g * g
        flat = emb.reshape(-1, emb.shape[-1])
        if flat.shape[0] != want:
            raise ValueError(
                f"visual tower produced {flat.shape[0]} tokens but the Plucker map "
                f"expects {want} ({pl.shape[0]} frames x {g}x{g}).  The processor's "
                "resize does not match token_grid; pass the true grid or disable "
                "patch injection.")
        feats = self.injector.encoder(pl.to(torch.float32))     # (T, g*g, H)
        feats = feats.reshape(-1, feats.shape[-1]).to(emb.dtype)
        self._patch_hits += 1
        with torch.no_grad():
            # Telemetry belongs HERE and not in injector.inject_patches: this method
            # calls the encoder directly and never goes through it, so an instrument
            # placed there reports a flat 0% while the injection is in fact happening.
            self.injector.last_rel = float(feats.norm() / flat.norm().clamp_min(1e-8))
        return self._add_view_embed((flat + feats).reshape(emb.shape))

    def install_block_mask_hook(self):
        """Let a prepared 4D mask reach the language model without touching M-RoPE.

        The mask is swapped in at the LANGUAGE MODEL's input, not passed to the VL model
        at the top: Qwen3VLModel computes the 3-D M-RoPE position ids from whatever mask it
        receives (it takes the diagonal of a 4D one), and every position there must stay
        exactly what it was.  Inside the text model create_causal_mask returns an
        already-4D mask as-is, and SDPA turns is_causal off whenever a mask is given, so
        this reaches every layer -- including tier D's patched attention, which forwards
        attention_mask unchanged to the same SDPA call.
        """
        if self._lm_hook is not None:
            return
        lm = self.rbm.model.language_model

        def pre(_m, args, kwargs):
            if self._block_mask is not None:
                kwargs["attention_mask"] = self._block_mask
            return args, kwargs
        self._lm_hook = lm.register_forward_pre_hook(pre, with_kwargs=True)

    def _build_block_mask(self, inp, n_views: int) -> torch.Tensor:
        """Causal across instants, bidirectional within one: (1, 1, L, L) bool, True=attend.

        Frames arrive interleaved as v0_t, v1_t, ..., each laid out by the collator as
        ``<|vision_start|> <|image_pad|>*g*g <|vision_end|> <|prog_token|>``.  Under the
        plain causal mask the first camera of an instant cannot see the others' frames of
        that same instant while the last one sees them all, so the views are not
        symmetric.  Block t spans from the first view's <|vision_start|> to the last
        view's <|prog_token|>; inside it every token attends to every other.  The prompt
        and the "Picture N:" labels between blocks stay causal.
        """
        ids = inp["input_ids"]
        if ids.shape[0] != 1:
            raise ValueError("block mask is built for batch 1, the only shape used here")
        tok = self.rbm.processor.tokenizer
        vs = (ids[0] == tok.convert_tokens_to_ids("<|vision_start|>")).nonzero(as_tuple=True)[0]
        pg = (ids[0] == tok.convert_tokens_to_ids("<|prog_token|>")).nonzero(as_tuple=True)[0]
        if len(vs) != len(pg) or len(vs) % n_views or not bool((pg > vs).all()):
            raise ValueError(f"{len(vs)} images / {len(pg)} prog tokens do not form "
                             f"{n_views}-view instants; the layout is not what this assumes")
        L = ids.shape[1]
        m = torch.ones(L, L, dtype=torch.bool, device=ids.device).tril_()
        for t in range(len(vs) // n_views):
            s, e = int(vs[t * n_views]), int(pg[t * n_views + n_views - 1]) + 1
            m[s:e, s:e] = True
        am = inp.get("attention_mask")
        if am is not None:                          # keep padding out, should any appear
            m = m & am[0].bool()[None, :]
        return m[None, None]

    def _build_block_mask_no_own_prog(self, inp, n_views: int) -> torch.Tensor:
        """Block-causal variant: frames cannot attend to their own progress token.

        Frame queries include vision-start through vision-end.  Cross-view
        visibility and all progress-token queries retain the original block mask.
        """
        mask = self._build_block_mask(inp, n_views)
        ids = inp["input_ids"][0]
        tok = self.rbm.processor.tokenizer
        vs = (ids == tok.convert_tokens_to_ids("<|vision_start|>")).nonzero(as_tuple=True)[0]
        pg = (ids == tok.convert_tokens_to_ids("<|prog_token|>")).nonzero(as_tuple=True)[0]
        for frame_start, progress_pos in zip(vs.tolist(), pg.tolist()):
            mask[:, :, frame_start:progress_pos, progress_pos] = False
        return mask

    def _add_view_embed(self, emb: torch.Tensor) -> torch.Tensor:
        """Add a per-view learned vector to every patch token of each frame.

        ``_view_slots`` is one view index per FRAME, in the order the frames were
        stacked; the vision tower returns frames' tokens contiguously in that same
        order, so the tag lands on the frame it names.  A mismatch would tag frames
        with the wrong camera and still train, so the token count is checked.
        """
        if self.view_embed is None or self._view_slots is None:
            return emb
        g = self.token_grid
        slots = self._view_slots
        flat = emb.reshape(-1, emb.shape[-1])
        want = len(slots) * g * g
        if flat.shape[0] != want:
            raise ValueError(
                f"visual tower produced {flat.shape[0]} tokens but the view tags "
                f"expect {want} ({len(slots)} frames x {g}x{g})")
        tag = self.view_embed(slots.to(flat.device))                  # (F, H)
        tag = tag.repeat_interleave(g * g, dim=0).to(flat.dtype)      # (F*g*g, H)
        with torch.no_grad():
            self.last_view_rel = float(tag.norm() / flat.norm().clamp_min(1e-8))
        return (flat + tag).reshape(emb.shape)

    # ------------------------------------------------------------------- pose
    def set_pose(self, cam: CameraParams, training: bool = False):
        """Compute and stash the Plucker map / control features for one clip."""
        g = self.token_grid
        dev = next(self.injector.parameters()).device
        pl = plucker_map(cam, g, g).to(dev, torch.float32)
        pv = pose_vector(cam, None).to(dev, torch.float32)

        if self.pose_null or (training and self.pose_dropout > 0
                              and torch.rand(()) < self.pose_dropout):
            pl = torch.zeros_like(pl)
            pv = torch.zeros_like(pv)

        self._plucker = pl
        if self.injector.uses_cross_attention:
            # (T, 1 + g*g, C) -> (1, T*(1+g*g), C).
            # The LLM sees ONE sequence with all T frames interleaved, so the control
            # side must be one sequence too: any position should be able to attend to
            # any frame's pose.  Leaving it batched per frame would silently mismatch
            # against a batch-1 hidden state.
            c = self.injector.control_features(pl, pv)
            self._control = c.reshape(1, -1, c.shape[-1])
        else:
            self._control = None

    def clear_pose(self):
        """Run the backbone with no conditioning -- this IS the baseline path."""
        self._plucker = None
        self._control = None
        # features=None makes every patched tier-D forward fall straight through to
        # the original attention, which is what the pose-blind control needs.
        self.ray_state.clear()

    # ---------------------------------------------------------------- forward
    def build_inputs(self, frames, prompt: str, device):
        """Robometer's own collator; the conditioned and baseline models share it."""
        from robometer.data.dataset_types import ProgressSample, Trajectory
        import numpy as np

        f = np.asarray(frames)
        traj = Trajectory(frames=f, frames_shape=tuple(f.shape), task=prompt, id="0",
                          metadata={"subsequence_length": len(f)}, video_embeddings=None)
        batch = self.collator([ProgressSample(trajectory=traj, sample_type="progress")])
        inp = batch["progress_inputs"]
        md = next(self.rbm.parameters()).dtype
        for k, v in list(inp.items()):
            if hasattr(v, "to"):
                v = v.to(device)
                if torch.is_floating_point(v):
                    v = v.to(md)
                inp[k] = v
        return inp

    IMAGE_TOKEN_ID = 151655        # Qwen3-VL's image placeholder, from the config

    def _set_ray_state(self, inp):
        """Scatter this clip's ray features onto the visual positions of the sequence.

        THE ONLY PLACE TIER D CAN GO WRONG QUIETLY.  ``_inject_visual`` gets the
        vision tower's own output and lines up positionally; here the features have
        to be placed into the LLM sequence by hand, and a wrong position attaches
        one patch's ray to another patch's content without any shape ever
        disagreeing.  So the placeholder count is checked against T*g*g rather than
        trusted: if the processor's resize or the frame count ever changes, this
        raises instead of training on scrambled geometry.
        """
        pl = self._plucker
        if pl is None:
            self.ray_state.clear()
            return
        ids = inp.get("input_ids")
        if ids is None:
            raise RuntimeError("tier D needs input_ids to locate the visual tokens")
        ids = ids.reshape(-1)
        pos = (ids == self.IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        g = self.token_grid
        want = pl.shape[0] * g * g
        if pos.numel() != want:
            raise ValueError(
                f"{pos.numel()} image placeholders in the sequence but the Plucker "
                f"map has {want} tokens ({pl.shape[0]} frames x {g}x{g}).  Tier D "
                "would attach geometry to the wrong positions.")
        feat = self.injector.pe[str(self.injector.layers[0])].features(
            pl.to(torch.float32))                       # (T, g*g, 7)
        feat = feat.reshape(-1, feat.shape[-1])         # (T*g*g, 7) in image order
        f, m = ray_mask_and_features(feat, pos, int(ids.shape[0]))
        self.ray_state.features, self.ray_state.mask = f, m

    def forward(self, inp, return_logits: bool = False):
        """-> per-frame progress in [0,1], shape (T,).

        Robometer-4B is a DISCRETE progress model (10 bins), so the head emits
        ``(T, n_bins)`` logits.  Scalar progress is the expectation over bin centres,
        matching Robometer's own eval-time conversion -- using an argmax instead
        would quantise the curve to 10 levels and inflate the tie count that
        Kendall tau treats as zero.
        """
        if getattr(self.injector, "uses_qk_pe", False):
            self._set_ray_state(inp)
        if self.block_views:
            if self.block_mask_no_own_prog:
                self._block_mask = self._build_block_mask_no_own_prog(inp, self.block_views)
            else:
                self._block_mask = self._build_block_mask(inp, self.block_views)
        try:
            out, _ = self.rbm(**inp, sample_type="progress")
        finally:
            self._block_mask = None
        # Robometer computes success for every frame on the progress path too (its
        # rewind_transformer does it right beside the progress logits).  Stash rather
        # than return it, so the many call sites that want progress alone keep working.
        sl = getattr(out, "success_logits", None)
        if isinstance(sl, dict):
            sl = sl.get("A")
        self.last_success_logits = None if sl is None else sl.reshape(-1).float()
        prog = out.progress_logits
        if isinstance(prog, dict):
            prog = prog["A"]
        if prog.dim() >= 2 and prog.shape[0] == 1:
            prog = prog[0]
        prog = prog.float()

        if not self.is_discrete:
            return (prog, prog) if return_logits else prog

        logits = prog                                        # (T, n_bins)
        centres = (torch.arange(self.n_bins, device=logits.device, dtype=logits.dtype)
                   + 0.5) / self.n_bins
        scalar = (logits.softmax(-1) * centres).sum(-1)
        return (scalar, logits) if return_logits else scalar

    # ----------------------------------------------------------------- params
    def adapter_parameters(self):
        return [p for p in self.injector.parameters() if p.requires_grad]

    def describe(self) -> str:
        live = sum(p.numel() for p in self.adapter_parameters())
        dead = sum(p.numel() for p in self.injector.parameters() if not p.requires_grad)
        gates = ((self.injector.gate_values() if hasattr(self.injector, "gate_values") else {}) if self.injector.uses_cross_attention else {})
        g = " ".join(f"L{k}={v:+.3f}" for k, v in sorted(gates.items()))
        return (f"PoseConditionedRobometer[{self.injector.describe()}] "
                f"trainable={live/1e6:.2f}M (+{dead/1e6:.2f}M frozen-unreachable), "
                f"patch_inject={self.patch_inject}, backbone frozen="
                f"{not any(p.requires_grad for p in self.rbm.parameters())}"
                + (f", gates[{g}]" if g else ""))
