"""Collator: turn MultiViewSamples into backbone inputs plus pose tensors.

The chat template per view is

    <task text> [ <|cam_token|> <frame image> <|progress_token|> ] x T

so every frame carries two reserved positions:

  ``<|cam_token|>``       overwritten with the pose descriptor (tiers B and C)
  ``<|progress_token|>``  read out by the progress head

Putting ``<cam>`` *before* its frame is deliberate: with causal attention the frame's
own tokens can attend to the pose that describes them, which they could not do if
the pose token came afterwards.

Both tokens must be added to the tokenizer before use -- :func:`add_special_tokens`
does that and returns the ids, and the model's embedding table has to be resized to
match.  The tests exercise the id bookkeeping without a real tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from anyviewmeter.data.dataset_types import MultiViewSample, Trajectory

CAM_TOKEN = "<|cam_token|>"
PROGRESS_TOKEN = "<|progress_token|>"
SPLIT_TOKEN = "<|split_token|>"
SPECIAL_TOKENS = [CAM_TOKEN, PROGRESS_TOKEN, SPLIT_TOKEN]


def add_special_tokens(tokenizer, model=None) -> Dict[str, int]:
    """Register our reserved tokens and (optionally) resize the embedding table."""
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added and model is not None and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    return {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}


@dataclass
class CollatedBatch:
    """What the trainer hands to the model."""

    frames: torch.Tensor                 # (B*V, T, H, W, 3) uint8
    plucker: torch.Tensor                # (B*V*T, 6, g, g)
    pose_vec: torch.Tensor               # (B*V*T, pose_dim)
    target_progress: torch.Tensor        # (B*V, T)
    success_label: torch.Tensor          # (B*V, T)
    prompts: List[str]
    view_group: torch.Tensor             # (B*V,) which sample each view came from
    meta: List[dict]

    def to(self, device):
        for f in ("frames", "plucker", "pose_vec", "target_progress",
                  "success_label", "view_group"):
            setattr(self, f, getattr(self, f).to(device))
        return self

    @property
    def num_views(self) -> int:
        return self.frames.shape[0]

    def groups(self) -> Dict[int, List[int]]:
        """sample index -> row indices, i.e. which rows are views of one trajectory."""
        out: Dict[int, List[int]] = {}
        for i, g in enumerate(self.view_group.tolist()):
            out.setdefault(int(g), []).append(i)
        return out


class AVMCollator:
    """Builds :class:`CollatedBatch` from a list of samples.

    ``processor``/``tokenizer`` are optional; without them the collator still
    produces frames, Plucker maps and targets, which is enough for the geometry
    and loss tests and for a backbone-free smoke run.
    """

    def __init__(self, token_grid: int = 8, pose_dim: int = 16,
                 processor=None, tokenizer=None,
                 normalize_directions: bool = True,
                 workspace_centres: Optional[Dict[str, Sequence[float]]] = None):
        self.token_grid = token_grid
        self.pose_dim = pose_dim
        self.processor = processor
        self.tokenizer = tokenizer
        self.normalize_directions = normalize_directions
        self.workspace_centres = workspace_centres or {}

    # ------------------------------------------------------------------ pieces
    def _pose_for(self, traj: Trajectory):
        from anyviewmeter.geometry.plucker import plucker_map, pose_vector

        if traj.camera is None:
            raise ValueError(f"trajectory {traj.id} has no camera; AnyviewMeter "
                             "cannot build Plucker maps without one")
        n = traj.num_frames
        cam = traj.camera.as_params(num_frames=n)
        g = self.token_grid
        wc = traj.camera.workspace_centre
        if wc is None and traj.metadata:
            wc = self.workspace_centres.get(traj.metadata.get("task"))
        wc_t = torch.as_tensor(wc, dtype=torch.float32) if wc is not None else None
        pl = plucker_map(cam, g, g, normalize=self.normalize_directions)   # (T,6,g,g)
        pv = pose_vector(cam, wc_t)                                        # (T,pose_dim)
        return pl, pv

    def build_prompt(self, traj: Trajectory) -> str:
        """Text with one <cam>/<progress> pair per frame."""
        per_frame = f"{CAM_TOKEN}{PROGRESS_TOKEN}"
        return f"{traj.task or ''}{SPLIT_TOKEN}" + per_frame * traj.num_frames

    # ------------------------------------------------------------------- call
    def __call__(self, samples: List[MultiViewSample]) -> CollatedBatch:
        frames, plucker, poses, targets, success = [], [], [], [], []
        prompts, groups, meta = [], [], []

        for gi, sample in enumerate(samples):
            views = sample.views if isinstance(sample, MultiViewSample) else [sample.trajectory]
            for traj in views:
                f = np.asarray(traj.frames)
                pl, pv = self._pose_for(traj)
                frames.append(torch.as_tensor(f))
                plucker.append(pl)
                poses.append(pv)
                targets.append(torch.as_tensor(traj.target_progress, dtype=torch.float32))
                success.append(torch.as_tensor(traj.success_label, dtype=torch.float32))
                prompts.append(self.build_prompt(traj))
                groups.append(gi)
                meta.append(dict(traj.metadata or {}, id=traj.id))

        return CollatedBatch(
            frames=torch.stack(frames),
            plucker=torch.cat(plucker, dim=0),
            pose_vec=torch.cat(poses, dim=0),
            target_progress=torch.stack(targets),
            success_label=torch.stack(success),
            prompts=prompts,
            view_group=torch.as_tensor(groups, dtype=torch.long),
            meta=meta,
        )
