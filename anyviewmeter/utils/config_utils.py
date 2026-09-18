"""YAML -> dataclass config loading, with CLI dotted overrides."""
from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional

import yaml

from anyviewmeter.configs.experiment_configs import (DataConfig, ExperimentConfig,
                                                     LossConfig, ModelConfig,
                                                     PoseConfig, TrainingConfig)


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None,
                overrides: Optional[List[str]] = None) -> ExperimentConfig:
    """Load a YAML config and apply ``a.b.c=value`` overrides."""
    raw: Dict[str, Any] = {}
    if path:
        if not os.path.exists(path):
            here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "configs", path)
            path = here if os.path.exists(here) else path
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed

    model = raw.get("model", {}) or {}
    pose = model.pop("pose", {}) or {}
    # unknown keys are a config typo, and silently ignoring them is how an
    # ablation ends up not actually ablating anything
    _reject_unknown(pose, PoseConfig, "model.pose")
    _reject_unknown(model, ModelConfig, "model")
    cfg = ExperimentConfig(
        model=ModelConfig(pose=PoseConfig(**pose), **model),
        data=DataConfig(**_checked(raw.get("data", {}) or {}, DataConfig, "data")),
        loss=LossConfig(**_checked(raw.get("loss", {}) or {}, LossConfig, "loss")),
        training=TrainingConfig(**_checked(raw.get("training", {}) or {}, TrainingConfig,
                                           "training")),
    )
    return cfg


def _field_names(cls) -> set:
    return set(getattr(cls, "__dataclass_fields__", {}))


def _reject_unknown(d: dict, cls, where: str):
    bad = set(d) - _field_names(cls)
    if bad:
        raise ValueError(f"unknown key(s) in '{where}': {sorted(bad)}; "
                         f"valid: {sorted(_field_names(cls))}")


def _checked(d: dict, cls, where: str) -> dict:
    _reject_unknown(d, cls, where)
    return d


def config_to_dict(cfg: ExperimentConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def save_config(cfg: ExperimentConfig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
