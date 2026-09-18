"""Trajectory / sample dataclasses.

Deliberately the same shape as ``robometer/data/dataset_types.py`` plus one field:
``camera``.  Keeping the rest identical means a Robometer checkpoint and an
AnyviewMeter checkpoint can be scored by the same evaluation code, which is what
makes the ablation "does pose conditioning help" answerable at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict


class CameraView(BaseModel):
    """Camera geometry for one clip.

    Per-frame extrinsics are supported (``(T, 3, 4)``) for the moving-camera case;
    a single ``(3, 4)`` is broadcast over the clip.  Our current renders are static
    per clip, but the model path must not assume that -- a wrist camera is the
    obvious next dataset and it moves every frame.
    """

    intrinsic: Union[np.ndarray, torch.Tensor, List]          # (3,3) or (T,3,3)
    extrinsic: Union[np.ndarray, torch.Tensor, List]          # (3,4) or (T,3,4)
    height: int
    width: int

    name: Optional[str] = None
    workspace_centre: Optional[List[float]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def as_params(self, num_frames: Optional[int] = None):
        """-> :class:`anyviewmeter.geometry.camera.CameraParams` with a leading T dim."""
        from anyviewmeter.geometry.camera import CameraParams

        K = torch.as_tensor(np.asarray(self.intrinsic), dtype=torch.float32)
        E = torch.as_tensor(np.asarray(self.extrinsic), dtype=torch.float32)
        if K.ndim == 2:
            K = K[None]
        if E.ndim == 2:
            E = E[None]
        if num_frames is not None:
            if K.shape[0] == 1:
                K = K.expand(num_frames, 3, 3)
            if E.shape[0] == 1:
                E = E.expand(num_frames, 3, 4)
            if K.shape[0] != num_frames or E.shape[0] != num_frames:
                raise ValueError(f"camera has {K.shape[0]}/{E.shape[0]} entries "
                                 f"but the clip has {num_frames} frames")
        return CameraParams(K, E, self.height, self.width)


class Trajectory(BaseModel):
    """One clip: frames, task text, targets, and the camera that produced it."""

    frames: Union[List[str], np.ndarray, None] = None
    frames_shape: Optional[Tuple] = None

    camera: Optional[CameraView] = None

    id: Optional[str] = None
    task: Optional[str] = None
    data_source: Optional[str] = None

    target_progress: Optional[Union[List[float], torch.Tensor]] = None
    success_label: Optional[List[float]] = None
    partial_success: Optional[Union[float, torch.Tensor]] = None

    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def num_frames(self) -> int:
        if self.frames is None:
            return 0
        return len(self.frames)


class ProgressSample(BaseModel):
    trajectory: Trajectory
    sample_type: str = "progress"
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PreferenceSample(BaseModel):
    chosen_trajectory: Trajectory
    rejected_trajectory: Trajectory
    sample_type: str = "preference"
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MultiViewSample(BaseModel):
    """The SAME trajectory seen from several cameras.

    This is the sample type the viewpoint-consistency loss needs, and it is why
    the dataset indexes by (task, trajectory) rather than by clip: a consistency
    penalty is only meaningful between views whose underlying motion is identical.
    """

    views: List[Trajectory]
    sample_type: str = "multiview"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def num_views(self) -> int:
        return len(self.views)


SampleType = Union[ProgressSample, PreferenceSample, MultiViewSample]
