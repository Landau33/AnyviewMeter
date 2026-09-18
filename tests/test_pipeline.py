"""Collator, losses and the E4 gate.

The gate tests are the important ones: a gate that only ever prints PASS is worse
than no gate, so it has to be shown rejecting the degenerate case it exists for.

Run:  python AnyviewMeter/tests/test_pipeline.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anyviewmeter.data.collators.avm_collator import AVMCollator  # noqa: E402
from anyviewmeter.data.dataset_types import CameraView, MultiViewSample, Trajectory  # noqa: E402
from anyviewmeter.evals.diagnostics import (gate_verdict, h_time, p_obj,  # noqa: E402
                                            progress_tau, s_view)
from anyviewmeter.geometry.camera import look_at_extrinsic, pinhole_intrinsic  # noqa: E402
from anyviewmeter.trainers.avm_trainer import (pose_consistency_loss,  # noqa: E402
                                               progress_loss)

T, RES, G = 8, 256, 8


def _traj(cam_name, eye, kind="success"):
    K = pinhole_intrinsic(np.deg2rad(45.0), RES, RES)
    E = look_at_extrinsic(eye, [0.0, 0.0, 0.0])
    frames = np.zeros((T, RES, RES, 3), dtype=np.uint8)
    prog = list(np.linspace(0, 1, T)) if kind == "success" else [0.0] * T
    return Trajectory(
        frames=frames, frames_shape=frames.shape,
        camera=CameraView(intrinsic=K.numpy(), extrinsic=E.numpy(), height=RES,
                          width=RES, name=cam_name, workspace_centre=[0.0, 0.0, 0.0]),
        id=f"t|{cam_name}|{kind}", task="stack the red cube on the green cube",
        target_progress=prog, success_label=[1.0 if kind == "success" else 0.0] * T,
        metadata={"task": "StackCube", "cam": cam_name, "kind": kind})


def _sample():
    return MultiViewSample(views=[_traj("frontal_good", [0.4, 0, 0.4]),
                                  _traj("side", [0.0, 0.55, 0.05])])


# ------------------------------------------------------------------- collator
def test_collator_shapes():
    b = AVMCollator(token_grid=G)([_sample(), _sample()])
    assert b.frames.shape == (4, T, RES, RES, 3)
    assert b.plucker.shape == (4 * T, 6, G, G)
    assert b.pose_vec.shape == (4 * T, 16)
    assert b.target_progress.shape == (4, T)
    assert b.num_views == 4


def test_collator_groups_views_of_one_trajectory():
    b = AVMCollator(token_grid=G)([_sample(), _sample()])
    g = b.groups()
    assert len(g) == 2 and all(len(v) == 2 for v in g.values()), g


def test_prompt_has_one_token_pair_per_frame():
    b = AVMCollator(token_grid=G)([_sample()])
    p = b.prompts[0]
    assert p.count("<|cam_token|>") == T
    assert p.count("<|progress_token|>") == T
    # <cam> must precede its frame's progress token so causal attention can use it
    assert p.index("<|cam_token|>") < p.index("<|progress_token|>")


def test_collator_rejects_camera_free_trajectory():
    t = _traj("frontal_good", [0.4, 0, 0.4])
    t.camera = None
    try:
        AVMCollator(token_grid=G)([MultiViewSample(views=[t])])
    except ValueError as e:
        assert "camera" in str(e).lower()
        return
    raise AssertionError("missing camera did not raise")


def test_different_cameras_give_different_plucker_rows():
    b = AVMCollator(token_grid=G)([_sample()])
    a, c = b.plucker[:T], b.plucker[T:2 * T]
    assert (a - c).abs().max() > 0.5, "two views produced near-identical Plucker maps"


# --------------------------------------------------------------------- losses
def test_pose_consistency_is_zero_for_identical_views():
    p = torch.rand(2, T)
    both = torch.stack([p[0], p[0]])
    assert float(pose_consistency_loss(both, {0: [0, 1]})) < 1e-8


def test_pose_consistency_positive_for_disagreeing_views():
    p = torch.stack([torch.zeros(T), torch.ones(T)])
    assert float(pose_consistency_loss(p, {0: [0, 1]})) > 0.1


def test_pose_consistency_ignores_singleton_groups():
    p = torch.rand(1, T)
    assert float(pose_consistency_loss(p, {0: [0]})) == 0.0


def test_progress_loss_discrete_and_continuous():
    tgt = torch.linspace(0, 1, T)[None]
    assert float(progress_loss(tgt.clone(), tgt)) < 1e-8
    logits = torch.randn(1, T, 10)
    assert float(progress_loss(logits, tgt, discrete=True, n_bins=10)) > 0


# ----------------------------------------------------------------------- gate
def _diag(sview, stuck, shuffled, pobj):
    trajs = [f"traj_{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    best = {t: 0.8 + 0.01 * rng.standard_normal() for t in trajs}
    worst = {t: 0.8 - sview + 0.01 * rng.standard_normal() for t in trajs}
    succ = {t: 0.8 + 0.01 * rng.standard_normal() for t in trajs}
    fail = {t: 0.8 - pobj + 0.01 * rng.standard_normal() for t in trajs}
    return {
        "S_view": s_view({"best": best, "worst": worst}),
        "H_time": h_time({"stuck": [stuck] * 10, "reversed": [-0.4] * 10,
                          "shuffled": [shuffled] * 10}),
        "P_obj": p_obj(succ, fail),
    }


def test_gate_passes_a_healthy_improvement():
    ref = _diag(sview=0.50, stuck=0.40, shuffled=0.10, pobj=0.50)
    cur = _diag(sview=0.20, stuck=0.15, shuffled=-0.20, pobj=0.55)
    assert gate_verdict(cur, ref)["verdict"] == "PASS"


def test_gate_disqualifies_position_prior_decay():
    """S_view improves but the frozen-clip drift worsens -> must be refused."""
    ref = _diag(sview=0.50, stuck=0.20, shuffled=-0.10, pobj=0.50)
    cur = _diag(sview=0.10, stuck=0.60, shuffled=+0.30, pobj=0.10)
    v = gate_verdict(cur, ref)
    assert v["verdict"].startswith("DISQUALIFIED"), v
    assert v["H_time_regressed"] and v["P_obj_regressed"]


def test_gate_disqualifies_when_only_pobj_collapses():
    ref = _diag(sview=0.50, stuck=0.20, shuffled=-0.10, pobj=0.50)
    cur = _diag(sview=0.10, stuck=0.20, shuffled=-0.10, pobj=0.05)
    assert gate_verdict(cur, ref)["verdict"].startswith("DISQUALIFIED")


def test_gate_reports_no_improvement():
    ref = _diag(sview=0.30, stuck=0.20, shuffled=-0.10, pobj=0.50)
    assert gate_verdict(ref, ref)["verdict"] == "NO S_view IMPROVEMENT"


def test_baseline_mode_does_not_judge():
    assert gate_verdict(_diag(0.3, 0.2, -0.1, 0.5))["verdict"] == "BASELINE"


def test_abstention_is_not_scored_as_honesty():
    """An all-abstention frozen clip must be flagged, not credited."""
    ht = h_time({"stuck": [0.0] * 10}, {"stuck": [True] * 10})
    assert ht["stuck"]["declined"] and ht["stuck"]["n"] == 0
    v = gate_verdict({"S_view": None, "H_time": ht, "P_obj": None})
    assert any("declined" in n for n in v["notes"]), v["notes"]


def test_pobj_undefined_when_all_pairs_invalid():
    succ = {f"t{i}": 0.5 for i in range(5)}
    fail = {f"t{i}": 0.5 for i in range(5)}
    valid = {f"t{i}": False for i in range(5)}
    assert p_obj(succ, fail, valid)["undefined"]


def test_progress_tau_matches_expectation():
    assert progress_tau(list(np.linspace(0, 1, 10))) == 1.0
    assert progress_tau(list(np.linspace(1, 0, 10))) == -1.0
    assert progress_tau([0.5] * 10) == 0.0


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            torch.manual_seed(0)
            fn()
            print(f"[ok ] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERR ] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
