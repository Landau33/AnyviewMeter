"""Dataclass configs, mirroring ``robometer/configs/experiment_configs.py``.

Only the pose-conditioning section is genuinely new; the rest keeps Robometer's
shape so a Robometer checkpoint and an AnyviewMeter checkpoint stay comparable
term for term.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ------------------------------------------------------------------ pose block


@dataclass
class PoseConfig:
    """How camera pose is represented and how it enters the backbone."""

    enabled: bool = field(default=True, metadata={
        "help": "Master switch.  enabled=False reproduces a pose-blind Robometer-"
                "style baseline through this same code path, which is the control "
                "every pose result is measured against."})

    injector: str = field(default="cross_attn", metadata={
        "help": "Injection tier: 'patch_add' (A), 'cam_token' (B), 'cross_attn' (C). "
                "See models/injection/base.py for why there are three."})

    # --- Plucker map
    plucker_grid: Optional[int] = field(default=None, metadata={
        "help": "Token grid the Plucker map is built on.  None = derive from the "
                "image size and the vision tower's patch/merge sizes, which is what "
                "keeps the map aligned with the patch tokens."})
    normalize_directions: bool = field(default=True, metadata={
        "help": "Unit-length ray directions.  Off makes the signal scale with focal "
                "length, so the same camera at another resolution would look different."})
    n_freqs: int = field(default=6, metadata={
        "help": "Fourier bands on the Plucker map.  Raw Plucker is low-frequency and "
                "an MLP on it tends to learn a near-constant map."})
    pose_n_freqs: int = field(default=4, metadata={"help": "Fourier bands on the pose vector."})

    # --- <cam> register token (tiers B, C)
    cam_token: str = field(default="<|cam_token|>", metadata={
        "help": "Reserved token the collator inserts once per frame; its embedding is "
                "replaced by the pose descriptor."})
    workspace_centre: Optional[List[float]] = field(default=None, metadata={
        "help": "World anchor for the framing entries of the pose descriptor.  Phase A "
                "found viewpoint quality is mostly a statement about the camera "
                "RELATIVE to the workspace, so leaving this None discards real signal."})

    # --- cross-attention branch (tier C)
    control_dim: int = 512
    n_control_layers: int = 2
    n_heads: int = 8
    layer_fraction: float = field(default=0.25, metadata={
        "help": "Fraction of backbone layers receiving a cross-attention block. "
                "Every layer is wasteful and destabilising."})
    layers: Optional[List[int]] = field(default=None, metadata={
        "help": "Explicit layer indices; overrides layer_fraction."})

    # --- regularisation / diagnostics
    pose_dropout: float = field(default=0.1, metadata={
        "help": "Probability of zeroing the pose signal for a sample during training. "
                "Forces the model to stay usable without pose and prevents it from "
                "routing everything through the pose channel."})
    log_gate_values: bool = field(default=True, metadata={
        "help": "Log tier-C gate magnitudes.  A gate stuck near zero means the backbone "
                "declined the pose signal -- report it, do not hide it."})


# ----------------------------------------------------------------- model block


@dataclass
class ModelConfig:
    base_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    hidden_size: Optional[int] = None            # inferred from the backbone config
    dropout: float = 0.1

    use_multi_image: bool = True
    use_per_frame_progress_token: bool = True
    frame_pooling: str = "mean"                  # mean | boundary | attention
    frame_pooling_attn_temperature: float = 1.0

    progress_loss_type: str = "l2"               # l2 | discrete
    progress_discrete_bins: int = 10

    pose: PoseConfig = field(default_factory=PoseConfig)

    def __post_init__(self):
        if isinstance(self.pose, dict):
            self.pose = PoseConfig(**self.pose)
        if self.use_per_frame_progress_token and not self.use_multi_image:
            raise ValueError("use_per_frame_progress_token=True requires use_multi_image=True")


# ------------------------------------------------------------------ data block


@dataclass
class DataConfig:
    clips_root: str = "/home/yuang/ws_jepa/multicam_ws/outputs/clips"
    tasks: List[str] = field(default_factory=lambda: ["StackCube", "PegInsertionSide", "PickCube"])
    cameras: Optional[List[str]] = field(default=None, metadata={
        "help": "None = every camera present, including the sampled sweep poses. "
                "The sweep poses are the whole point: they are what makes pose a "
                "continuous input rather than a 5-way categorical."})
    kinds: List[str] = field(default_factory=lambda: ["success", "failure"])

    train_trajectories: Optional[int] = None
    val_trajectories: int = 6
    holdout_cameras: Optional[List[str]] = field(default=None, metadata={
        "help": "Cameras held out of training entirely.  This is the only honest test "
                "of viewpoint generalisation: scoring on poses the model trained on "
                "measures interpolation, not transfer."})
    holdout_tasks: Optional[List[str]] = None

    num_frames: int = 32
    image_size: int = 256
    batch_size: int = 1
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 20260728


# --------------------------------------------------------------- train / loss


@dataclass
class LossConfig:
    progress_weight: float = 1.0
    success_weight: float = 0.5
    preference_weight: float = 0.0
    progress_loss_type: str = "l2"
    progress_discrete_bins: int = 10

    pose_consistency_weight: float = field(default=0.0, metadata={
        "help": "Penalty on progress disagreement between two views of the SAME "
                "trajectory.  This is the direct training signal for viewpoint "
                "robustness -- but see the E4 gate: driving S_view down while H_time "
                "collapses is a degenerate solution, not a result."})


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/anyviewmeter"
    run_name: str = "avm"
    epochs: int = 1
    max_steps: Optional[int] = None
    lr: float = 1e-4
    pose_lr: Optional[float] = field(default=5e-4, metadata={
        "help": "Separate (higher) LR for the injection adapter.  It starts at zero "
                "and has to move much further than the pretrained weights."})
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True

    freeze_backbone: bool = field(default=True, metadata={
        "help": "Train only the injection adapter and the heads.  The honest default "
                "for a first result: a frozen backbone makes any change attributable "
                "to the pose pathway rather than to general finetuning."})
    freeze_vision_tower: bool = True

    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    seed: int = 20260728


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        for name, cls in (("model", ModelConfig), ("data", DataConfig),
                          ("loss", LossConfig), ("training", TrainingConfig)):
            v = getattr(self, name)
            if isinstance(v, dict):
                setattr(self, name, cls(**v))
        # keep the two places that mention the progress loss in sync
        self.model.progress_loss_type = self.loss.progress_loss_type
        self.model.progress_discrete_bins = self.loss.progress_discrete_bins
