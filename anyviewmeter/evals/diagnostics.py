"""The ternary diagnostic (S_view / H_time / P_obj), carried over from phase A.

This is not a new metric set -- it is the standing acceptance gate the phase-A G1
document binds every later phase to, reimplemented here so AnyviewMeter reports it
by default rather than as an afterthought.

  S_view  delta-tau(best camera - worst camera).  Should go DOWN after training.
  H_time  |tau| on frozen clips should be small; tau on reversed/shuffled negative.
  P_obj   tau(success) - tau(object-frozen failure).  Should hold or improve.

THE HARD RULE, and the reason this file exists at all: a model that ignores pixels
and counts frames is trivially viewpoint-invariant.  Phase A measured that
Robometer -- the checkpoint we finetune from -- already carries a position prior
(|tau| up to +0.65 on 32 byte-identical frames, and POSITIVE tau on temporally
shuffled clips).  So an S_view improvement here is meaningless on its own: it must
come with H_time and P_obj not degrading, or the model has simply decayed further
into that prior.  :func:`gate_verdict` refuses such a checkpoint.

Two accounting rules inherited from phase A, both of which changed numbers there:
  * an abstention (constant output) is a DECLINED answer, never tau = 0;
  * P_obj is undefined wherever the success and object-frozen clips are identical,
    which happens whenever the manipulated object is never rendered in that view.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

SEED = 20260728


# ------------------------------------------------------------------- metrics
def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Tau-a; ties contribute 0.  Matches the phase-A definition exactly."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    n = len(xa)
    if n < 2:
        return 0.0
    dx = np.sign(xa[:, None] - xa[None, :])
    dy = np.sign(ya[:, None] - ya[None, :])
    return float(np.triu(dx * dy, 1).sum() / (0.5 * n * (n - 1)))


def progress_tau(progress: Sequence[float]) -> float:
    p = np.asarray(progress, dtype=np.float64)
    return kendall_tau(np.arange(len(p)), p)


def is_abstention(progress: Sequence[float]) -> bool:
    return len(np.unique(np.asarray(progress))) <= 1


def boot_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
            seed: int = SEED) -> Dict[str, Optional[float]]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    if len(v) == 1:
        return dict(mean=float(v[0]), lo=float(v[0]), hi=float(v[0]), n=1)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 100 * alpha / 2)),
                hi=float(np.percentile(bs, 100 * (1 - alpha / 2))), n=int(len(v)))


def boot_ci_paired(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                   seed: int = SEED) -> Dict[str, Optional[float]]:
    aa = np.asarray(a, dtype=np.float64)
    ba = np.asarray(b, dtype=np.float64)
    if len(aa) != len(ba):
        raise ValueError("paired bootstrap needs equal-length inputs")
    if len(aa) < 2:
        return dict(mean=float((aa - ba).mean()) if len(aa) else None,
                    lo=None, hi=None, n=len(aa))
    d = aa - ba
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n=int(len(d)))


# ------------------------------------------------------------ the three terms
def s_view(per_cam: Dict[str, Dict[str, float]]) -> Optional[dict]:
    """S_view from ``{camera: {traj: tau}}``.

    Best/worst are chosen on cell means, then the difference is bootstrapped on
    per-trajectory PAIRS -- trajectory difficulty is a shared nuisance factor, so
    an unpaired CI would be needlessly wide.
    """
    means = {c: float(np.mean(list(v.values()))) for c, v in per_cam.items() if v}
    if len(means) < 2:
        return None
    best = max(means, key=lambda k: means[k])
    worst = min(means, key=lambda k: means[k])
    shared = sorted(set(per_cam[best]) & set(per_cam[worst]))
    if len(shared) < 2:
        return None
    d = boot_ci_paired([per_cam[best][t] for t in shared],
                       [per_cam[worst][t] for t in shared])
    d.update(best_cam=best, worst_cam=worst, cam_means=means, n_pairs=len(shared))
    return d


def h_time(taus: Dict[str, List[float]],
           abstentions: Optional[Dict[str, List[bool]]] = None) -> Dict[str, dict]:
    """Temporal honesty per perturbation kind.

    An all-abstention cell is reported as such and NOT scored: declining to answer
    on a frozen clip is not a demonstration of temporal honesty, whatever number
    the abstention happens to produce.
    """
    out = {}
    for kind, vals in taus.items():
        ab = (abstentions or {}).get(kind, [False] * len(vals))
        kept = [t for t, a in zip(vals, ab) if not a]
        ci = boot_ci(kept)
        ci["abstention_rate"] = float(np.mean(ab)) if len(ab) else 0.0
        ci["declined"] = bool(ci["abstention_rate"] >= 0.5)
        out[kind] = ci
    return out


def p_obj(success: Dict[str, float], failure: Dict[str, float],
          valid: Optional[Dict[str, bool]] = None) -> Optional[dict]:
    """P_obj, filtered PER TRAJECTORY.

    ``valid[traj]=False`` marks trajectories where the success and object-frozen
    clips are byte-identical (the object is never rendered from that view), so the
    difference is an exact zero that reflects the renderer, not the model.
    Averaging those in drags P_obj toward zero.
    """
    shared = sorted(set(success) & set(failure))
    if valid is not None:
        shared = [t for t in shared if valid.get(t, True)]
    if len(shared) < 2:
        return dict(mean=None, lo=None, hi=None, n=0, undefined=True)
    d = boot_ci_paired([success[t] for t in shared], [failure[t] for t in shared])
    d["undefined"] = False
    return d


# --------------------------------------------------------------------- gate
def gate_verdict(cur: dict, ref: Optional[dict] = None, tol: float = 0.05) -> dict:
    """Apply the hard rule.  ``ref=None`` records a baseline instead of judging.

    ``cur``/``ref`` are ``{"S_view": ..., "H_time": ..., "P_obj": ...}`` as produced
    above.  Returns a verdict plus the notes that explain it.
    """
    sv, ht, po = cur.get("S_view"), cur.get("H_time") or {}, cur.get("P_obj")
    notes = []

    stuck = ht.get("stuck")
    if stuck and stuck.get("declined"):
        notes.append(f"H_time/stuck is {stuck['abstention_rate']*100:.0f}% abstention -- "
                     "the model declined rather than demonstrated temporal honesty")
    if po and po.get("undefined"):
        notes.append("P_obj undefined: no view where the object-frozen failure is visible")
    if sv and sv.get("n_pairs") is not None and sv.get("cam_means") and \
            sv["n_pairs"] < len(sv["cam_means"]):
        notes.append(f"S_view uses only {sv['n_pairs']} paired trajectories")

    if ref is None:
        return dict(verdict="BASELINE", notes=notes)

    r_sv, r_ht, r_po = ref.get("S_view"), ref.get("H_time") or {}, ref.get("P_obj")
    sv_better = bool(sv and r_sv and sv["mean"] is not None and r_sv["mean"] is not None
                     and sv["mean"] < r_sv["mean"] - 1e-9)
    ht_worse = _h_time_regressed(ht, r_ht, tol)
    po_worse = bool(po and r_po and po.get("mean") is not None
                    and r_po.get("mean") is not None and po["mean"] < r_po["mean"] - tol)

    if sv_better and (ht_worse or po_worse):
        verdict = "DISQUALIFIED - decayed into a position prior"
    elif sv_better:
        verdict = "PASS"
    else:
        verdict = "NO S_view IMPROVEMENT"
    return dict(verdict=verdict, S_view_improved=sv_better,
                H_time_regressed=ht_worse, P_obj_regressed=po_worse, notes=notes)


def _h_time_regressed(cur: dict, ref: dict, tol: float) -> bool:
    c, r = cur.get("stuck"), ref.get("stuck")
    if c and r and c.get("mean") is not None and r.get("mean") is not None:
        if abs(c["mean"]) > abs(r["mean"]) + tol:      # frozen clip drifted more
            return True
    for k in ("reversed", "shuffled"):
        c, r = cur.get(k), ref.get(k)
        if c and r and c.get("mean") is not None and r.get("mean") is not None:
            if c["mean"] > r["mean"] + tol:            # less negative = worse
                return True
    return False


def format_report(name: str, diag: dict, verdict: Optional[dict] = None) -> str:
    L = [f"{name}", "-" * max(40, len(name))]
    sv = diag.get("S_view")
    if sv and sv.get("mean") is not None:
        L.append(f"  S_view  {sv['mean']:+.3f} [{sv['lo']:+.3f},{sv['hi']:+.3f}]  "
                 f"(best={sv['best_cam']} worst={sv['worst_cam']}, n={sv['n_pairs']})")
    po = diag.get("P_obj")
    if po and po.get("mean") is not None:
        L.append(f"  P_obj   {po['mean']:+.3f} [{po['lo']:+.3f},{po['hi']:+.3f}] (n={po['n']})")
    elif po:
        L.append("  P_obj   undefined")
    for kind, c in (diag.get("H_time") or {}).items():
        if c.get("declined"):
            L.append(f"  H_time/{kind:<11} ABSTAINED {c['abstention_rate']*100:.0f}%")
        elif c.get("mean") is not None:
            L.append(f"  H_time/{kind:<11} {c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]")
    if verdict:
        L.append(f"  VERDICT {verdict['verdict']}")
        for n in verdict.get("notes", []):
            L.append(f"    ! {n}")
    return "\n".join(L)
