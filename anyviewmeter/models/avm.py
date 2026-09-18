"""AnyviewMeter: a progress model whose backbone is conditioned on camera pose.

Structure mirrors ``robometer/models/rbm.py`` -- a VLM backbone plus progress /
success / preference heads -- with one addition: a :class:`PoseInjector` that
routes Plucker ray maps and a global pose descriptor into the backbone.

WHERE POSE ENTERS.  Three separate points, used by tiers A/B/C respectively:

  1. patch tokens.  After the vision tower has produced visual embeddings and
     before they are scattered into the LLM sequence, ``inject_patches`` adds the
     encoded Plucker map.  This is the hook that requires the map to be built on
     the post-merge token grid.
  2. the ``<cam>`` register token.  The collator reserves one position per frame;
     its embedding is overwritten with the pose descriptor.  Overwriting rather
     than adding matters -- the reserved token's pretrained embedding carries no
     useful information and adding to it would just inject noise.
  3. LLM layers.  Zero-gated cross-attention blocks read from a control branch,
     applied through forward hooks so the backbone code is untouched and the
     wrapper keeps working across transformers versions.

The whole thing is designed so that ``pose.enabled=False`` walks the identical
code path with the injector removed, which is the control every pose result is
measured against.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from anyviewmeter.geometry.camera import CameraParams
from anyviewmeter.geometry.plucker import POSE_VECTOR_DIM, plucker_map, pose_vector
from anyviewmeter.models.heads import PoseConsistencyHead, PredictionHeadsMixin
from anyviewmeter.models.injection import build_injector


class AVMOutput:
    """Loose container mirroring robometer's ModelOutput."""

    def __init__(self):
        self.progress_logits = None
        self.success_logits = None
        self.pref_logits = None
        self.pose_pred = None
        self.frame_features = None
        self.diagnostics: Dict[str, float] = {}


def token_grid_for(image_size: int, patch_size: int, merge_size: int) -> int:
    """Side length of the visual-token grid a square image collapses to.

    Qwen3-VL: patch 16 with a 2x2 merge means one token per 32x32 pixels, so a
    256px frame becomes an 8x8 grid.  The Plucker map must match this exactly or
    ``inject_patches`` raises.
    """
    eff = patch_size * merge_size
    if image_size % eff:
        raise ValueError(f"image_size {image_size} is not divisible by "
                         f"patch_size*merge_size = {eff}")
    return image_size // eff


class AnyviewMeter(PredictionHeadsMixin, nn.Module):
    """Pose-conditioned progress model.

    ``backbone`` is any VLM exposing ``get_input_embeddings()`` and a decoder-layer
    list; it is passed in rather than constructed so tests can run on a tiny stub
    and training can load a real Qwen3-VL without this file knowing the difference.
    """

    def __init__(self, backbone: Optional[nn.Module], hidden_dim: int,
                 model_config, processor=None, tokenizer=None,
                 n_backbone_layers: int = 36, token_grid: int = 8):
        super().__init__(hidden_dim=hidden_dim, model_config=model_config,
                         dropout=getattr(model_config, "dropout", 0.1))
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.model_config = model_config
        self.processor = processor
        self.tokenizer = tokenizer
        self.token_grid = token_grid

        pose_cfg = model_config.pose
        self.pose_cfg = pose_cfg
        self.pose_enabled = bool(pose_cfg.enabled)

        if self.pose_enabled:
            self.injector = build_injector(
                pose_cfg.injector,
                hidden_dim=hidden_dim, pose_dim=POSE_VECTOR_DIM,
                n_freqs=pose_cfg.n_freqs, pose_n_freqs=pose_cfg.pose_n_freqs,
                control_dim=pose_cfg.control_dim,
                n_control_layers=pose_cfg.n_control_layers,
                n_heads=pose_cfg.n_heads, n_backbone_layers=n_backbone_layers,
                layer_fraction=pose_cfg.layer_fraction, layers=pose_cfg.layers)
            self.pose_probe = PoseConsistencyHead(hidden_dim, POSE_VECTOR_DIM)
        else:
            self.injector = None
            self.pose_probe = None

        self._hooks: List = []
        self._control: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ pose IO
    def build_pose_inputs(self, cameras: List[CameraParams],
                          workspace_centre: Optional[torch.Tensor] = None):
        """-> (plucker (B,6,g,g), pose_vec (B,POSE_VECTOR_DIM)) on the model device."""
        g = self.pose_cfg.plucker_grid or self.token_grid
        pl, pv = [], []
        for cam in cameras:
            pl.append(plucker_map(cam, g, g, normalize=self.pose_cfg.normalize_directions))
            pv.append(pose_vector(cam, workspace_centre))
        plucker = torch.cat([x.reshape(-1, 6, g, g) for x in pl], dim=0)
        pose_vec = torch.cat([x.reshape(-1, POSE_VECTOR_DIM) for x in pv], dim=0)
        dev = next(self.parameters()).device
        return plucker.to(dev), pose_vec.to(dev)

    def maybe_drop_pose(self, plucker: torch.Tensor, pose_vec: torch.Tensor):
        """Randomly blank the pose signal during training.

        Keeps the model usable without pose and stops it routing everything through
        the pose channel -- the analogue of classifier-free guidance dropout.
        """
        p = float(self.pose_cfg.pose_dropout)
        if not self.training or p <= 0:
            return plucker, pose_vec
        keep = (torch.rand(plucker.shape[0], device=plucker.device) >= p).float()
        return plucker * keep[:, None, None, None], pose_vec * keep[:, None]

    # ------------------------------------------- injection point 1: patch tokens
    def inject_into_patches(self, patch_embeds: torch.Tensor,
                            plucker: torch.Tensor) -> torch.Tensor:
        if not self.pose_enabled:
            return patch_embeds
        return self.injector.inject_patches(patch_embeds, plucker)

    # -------------------------------------------- injection point 2: <cam> token
    def cam_token_embeddings(self, pose_vec: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.pose_enabled:
            return None
        return self.injector.cam_token_embedding(pose_vec)

    def scatter_cam_tokens(self, inputs_embeds: torch.Tensor, input_ids: torch.Tensor,
                           cam_token_id: int, cam_embeds: torch.Tensor) -> torch.Tensor:
        """Overwrite the reserved ``<cam>`` positions with the pose embeddings.

        Overwrite, not add: the reserved token's pretrained embedding is arbitrary
        and adding to it would inject noise into an otherwise clean channel.
        """
        mask = input_ids == cam_token_id
        n = int(mask.sum())
        if n == 0:
            return inputs_embeds
        if n != cam_embeds.shape[0]:
            raise ValueError(f"{n} <cam> positions in the batch but {cam_embeds.shape[0]} "
                             "pose embeddings; the collator and the camera list disagree")
        out = inputs_embeds.clone()
        out[mask] = cam_embeds.to(out.dtype)
        return out

    # ------------------------------------- injection point 3: LLM cross-attention
    def _decoder_layers(self) -> Optional[nn.ModuleList]:
        """Locate the decoder-layer list across the backbone layouts we support."""
        for path in ("language_model.layers", "model.language_model.layers",
                     "model.layers", "layers"):
            obj = self.backbone
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, (nn.ModuleList, list)):
                return obj
        return None

    def attach_cross_attention(self) -> int:
        """Install forward hooks that apply the tier-C blocks.  Returns hook count.

        Hooks keep the backbone's own code untouched, which is what lets the same
        wrapper work across transformers versions and across backbones.
        """
        self.detach_cross_attention()
        if not (self.pose_enabled and self.injector.uses_cross_attention):
            return 0
        layers = self._decoder_layers()
        if layers is None:
            raise RuntimeError("could not locate decoder layers on the backbone; "
                               "pass n_backbone_layers/layers explicitly or extend "
                               "_decoder_layers()")

        def make_hook(idx):
            def hook(_module, _args, output):
                if self._control is None:
                    return output
                if isinstance(output, tuple):
                    h = self.injector.apply_to_layer(idx, output[0], self._control)
                    return (h,) + output[1:]
                return self.injector.apply_to_layer(idx, output, self._control)
            return hook

        for idx in self.injector.layers:
            if idx < len(layers):
                self._hooks.append(layers[idx].register_forward_hook(make_hook(idx)))
        return len(self._hooks)

    def detach_cross_attention(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def set_control(self, plucker: Optional[torch.Tensor],
                    pose_vec: Optional[torch.Tensor]):
        """Compute and stash the control features the hooks will read."""
        if not (self.pose_enabled and self.injector.uses_cross_attention) or plucker is None:
            self._control = None
            return
        self._control = self.injector.control_features(plucker, pose_vec)

    # ------------------------------------------------------------------- heads
    def apply_heads(self, frame_feats: torch.Tensor) -> AVMOutput:
        """frame_feats: (B, T, D) one vector per frame -> per-frame predictions."""
        out = AVMOutput()
        out.frame_features = frame_feats
        progress = self.progress_head(frame_feats)
        if not self.use_discrete_progress:
            progress = progress.squeeze(-1)
        out.progress_logits = progress
        out.success_logits = self.success_head(frame_feats).squeeze(-1)
        if self.pose_probe is not None:
            out.pose_pred = self.pose_probe(frame_feats)
        if self.pose_enabled and self.pose_cfg.log_gate_values and \
                self.injector.uses_cross_attention:
            out.diagnostics.update({f"gate/layer_{k}": v
                                    for k, v in self.injector.gate_values().items()})
        return out

    # ------------------------------------------------------------------- utils
    def describe(self) -> str:
        if not self.pose_enabled:
            return "AnyviewMeter[pose disabled] -- pose-blind control"
        return f"AnyviewMeter[{self.injector.describe()}]"

    def pose_parameters(self):
        """Parameters belonging to the pose pathway (for a separate learning rate)."""
        mods = [m for m in (self.injector, self.pose_probe) if m is not None]
        for m in mods:
            yield from m.parameters()

    def head_parameters(self):
        for m in (self.progress_head, self.success_head, self.preference_head):
            yield from m.parameters()

    def freeze_backbone(self, freeze: bool = True):
        if self.backbone is None:
            return
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
