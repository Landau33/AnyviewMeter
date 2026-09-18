"""Dataset over the multi-camera clips rendered in phase A.

Each clip on disk is an ``.npz`` holding ``frames`` (T,H,W,3 uint8) and a JSON
``meta`` that already carries ``cam_params`` with ``intrinsic_cv`` / ``extrinsic_cv``
-- i.e. the Plucker inputs are present in the data we have, no re-render needed.

Two design points that matter for what the results will mean:

  * indexing is by ``(task, trajectory)``, not by clip, so a batch can hold several
    views of the same motion.  The viewpoint-consistency loss is only defined
    between views whose underlying trajectory is identical.

  * ``holdout_cameras`` removes poses from training entirely.  Evaluating on poses
    that appeared in training measures interpolation; the claim we want to make is
    about transfer to unseen viewpoints, and only a held-out pose can support it.

Progress targets are linear in frame index, matching Robometer's convention for
successful demos.  For the object-frozen ``failure`` clips the target is held at
the start value, because no task progress occurs -- that is precisely the P_obj
signal phase A measured, and it must be in the training targets, not only in eval.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from anyviewmeter.data.dataset_types import CameraView, MultiViewSample, Trajectory


def load_clip(path: str):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    return z["frames"], meta


class MultiCamClipDataset(Dataset):
    """Yields :class:`MultiViewSample` -- several camera views of one trajectory."""

    def __init__(self, clips_root: str, tasks: Sequence[str],
                 kinds: Sequence[str] = ("success", "failure"),
                 cameras: Optional[Sequence[str]] = None,
                 holdout_cameras: Optional[Sequence[str]] = None,
                 holdout_tasks: Optional[Sequence[str]] = None,
                 trajectories: Optional[Sequence[str]] = None,
                 max_trajectories: Optional[int] = None,
                 views_per_sample: int = 2,
                 split: str = "train",
                 workspace_centres: Optional[Dict[str, List[float]]] = None,
                 seed: int = 20260728):
        self.root = clips_root
        self.kinds = list(kinds)
        self.views_per_sample = views_per_sample
        self.split = split
        self.workspace_centres = workspace_centres or {}
        self.rng = np.random.default_rng(seed)

        held_cams = set(holdout_cameras or [])
        held_tasks = set(holdout_tasks or [])
        want_cams = set(cameras) if cameras else None

        # index[(task, kind, traj)] = {cam: path}
        self.index: Dict[tuple, Dict[str, str]] = {}
        for task in tasks:
            if split == "train" and task in held_tasks:
                continue
            if split == "holdout_task" and task not in held_tasks:
                continue
            for kind in self.kinds:
                for p in sorted(glob.glob(os.path.join(clips_root, task, kind, "*.npz"))):
                    stem = os.path.basename(p)[:-4]
                    if "__" not in stem:
                        continue
                    traj, cam = stem.split("__", 1)
                    if want_cams and cam not in want_cams:
                        continue
                    if split == "train" and cam in held_cams:
                        continue
                    if split == "holdout_cam" and cam not in held_cams:
                        continue
                    self.index.setdefault((task, kind, traj), {})[cam] = p

        # a sample needs at least the requested number of views
        self.keys = [k for k, v in sorted(self.index.items()) if len(v) >= 1]
        if trajectories is not None:
            keep = set(trajectories)
            self.keys = [k for k in self.keys if k[2] in keep]
        if max_trajectories is not None:
            trajs = sorted({k[2] for k in self.keys}, key=_traj_num)[:max_trajectories]
            keep = set(trajs)
            self.keys = [k for k in self.keys if k[2] in keep]

        if not self.keys:
            raise RuntimeError(
                f"no clips found under {clips_root} for tasks={list(tasks)} "
                f"kinds={self.kinds} split={split}.  Render them first with "
                "multicam_ws/scripts/render_probe.py")

    def __len__(self) -> int:
        return len(self.keys)

    # ------------------------------------------------------------------ target
    @staticmethod
    def _progress_target(kind: str, n: int) -> List[float]:
        """Linear ramp for a successful demo; flat for the object-frozen failure.

        The failure clips replay the full arm motion with the manipulated object
        pinned at its start pose, so nothing task-relevant happens.  A model that
        reports rising progress there is reading the arm, not the task -- the exact
        confusion P_obj measures.
        """
        if n <= 1:
            return [0.0] * n
        if kind == "success":
            return list(np.linspace(0.0, 1.0, n))
        return [0.0] * n

    def _camera(self, meta: dict, task: str, cam_name: str) -> CameraView:
        cp = meta["cam_params"]
        res = int(meta.get("res", 256))
        return CameraView(intrinsic=cp["intrinsic_cv"], extrinsic=cp["extrinsic_cv"],
                          height=res, width=res, name=cam_name,
                          workspace_centre=self.workspace_centres.get(task,
                                                                     cp.get("target")))

    def _trajectory(self, path: str, task: str, kind: str, traj: str,
                    cam_name: str) -> Trajectory:
        frames, meta = load_clip(path)
        n = len(frames)
        return Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            camera=self._camera(meta, task, cam_name),
            id=f"{task}|{traj}|{kind}|{cam_name}",
            task=meta.get("prompt", ""),
            data_source=task,
            target_progress=self._progress_target(kind, n),
            success_label=[1.0 if kind == "success" else 0.0] * n,
            partial_success=1.0 if kind == "success" else 0.0,
            metadata=dict(task=task, kind=kind, traj=traj, cam=cam_name,
                          precision=meta.get("precision"),
                          manip_frac=meta.get("manip_frac"),
                          arm_frac=meta.get("arm_frac")),
        )

    def __getitem__(self, i: int) -> MultiViewSample:
        task, kind, traj = self.keys[i]
        by_cam = self.index[(task, kind, traj)]
        cams = sorted(by_cam)
        k = min(self.views_per_sample, len(cams))
        if self.split == "train" and k < len(cams):
            chosen = list(self.rng.choice(cams, size=k, replace=False))
        else:
            chosen = cams[:k]
        views = [self._trajectory(by_cam[c], task, kind, traj, c) for c in chosen]
        return MultiViewSample(views=views)

    # ------------------------------------------------------------------ splits
    def describe(self) -> str:
        cams = sorted({c for v in self.index.values() for c in v})
        tasks = sorted({k[0] for k in self.keys})
        trajs = sorted({k[2] for k in self.keys})
        return (f"{self.__class__.__name__}[{self.split}]: {len(self.keys)} samples, "
                f"{len(tasks)} tasks, {len(trajs)} trajectories, {len(cams)} cameras "
                f"({', '.join(cams[:6])}{'...' if len(cams) > 6 else ''})")


def _traj_num(t: str) -> int:
    try:
        return int(t.split("_")[-1])
    except ValueError:
        return 0


def build_splits(cfg, workspace_centres: Optional[Dict[str, List[float]]] = None):
    """Train / val / held-out-camera datasets from a :class:`DataConfig`."""
    common = dict(clips_root=cfg.clips_root, tasks=cfg.tasks, kinds=cfg.kinds,
                  cameras=cfg.cameras, holdout_cameras=cfg.holdout_cameras,
                  holdout_tasks=cfg.holdout_tasks,
                  workspace_centres=workspace_centres, seed=cfg.seed)

    all_trajs = sorted({os.path.basename(p)[:-4].split("__")[0]
                        for t in cfg.tasks
                        for k in cfg.kinds
                        for p in glob.glob(os.path.join(cfg.clips_root, t, k, "*.npz"))},
                       key=_traj_num)
    n_val = min(cfg.val_trajectories, max(0, len(all_trajs) - 1))
    val_trajs = all_trajs[-n_val:] if n_val else []
    train_trajs = all_trajs[:len(all_trajs) - n_val]
    if cfg.train_trajectories:
        train_trajs = train_trajs[:cfg.train_trajectories]

    splits = {
        "train": MultiCamClipDataset(split="train", trajectories=train_trajs, **common),
        "val": MultiCamClipDataset(split="val", trajectories=val_trajs, **common),
    }
    # Held-out poses / tasks are the only honest measure of viewpoint transfer:
    # scoring on cameras that appeared in training measures interpolation.
    if cfg.holdout_cameras:
        splits["holdout_cam"] = MultiCamClipDataset(split="holdout_cam",
                                                    trajectories=val_trajs, **common)
    if cfg.holdout_tasks:
        splits["holdout_task"] = MultiCamClipDataset(split="holdout_task",
                                                     trajectories=val_trajs, **common)
    return splits
