"""Two-camera comparison as designed: single-view R1/R0 run on EACH camera separately and
fused afterwards, versus P1, which sees both cameras in one sequence.

Inputs are per-clip curves from dump_test_curves.py:
    pred|traj__kind|cam   label|traj__kind|cam   (label only for labelled trajectories)
A single-view dump covers every camera including `canonical`; a joint dump is keyed by the
battery camera, its prediction already conditioned on canonical.

Per test pair (trajectory, battery camera) the single-view systems are
    battery    the model on the battery camera alone
    canonical  the model on the fixed camera alone
    mean       the average of the two curves -- a fusion that needs no labels
    best-MAE   ORACLE: per clip, whichever of the two has lower MAE against the label
    best-tau   ORACLE: per clip, whichever of the two has higher Kendall tau
The oracles read the test labels to choose, so they are an upper bound on any rule that
picks one camera; beating them is the strong form of "two cameras jointly beat two cameras
separately".  They exist only for labelled clips, so their consistency columns are blank.
"""
import argparse, json, re, numpy as np
from collections import defaultdict

G = ["az_seen", "az_edge", "az_far", "ood_fov"]

def kendall(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    return float(np.triu(np.sign(y[:, None] - y[None, :]) * np.sign(p[:, None] - p[None, :]), 1).sum()
                 / (0.5 * n * (n - 1)))

# Camera names are not uniform: the out-of-distribution-FOV cameras are "oodfov00".."oodfov03"
# while their group is "ood_fov", so slicing the name silently dropped that whole group.
# The dataset index is the source of truth when given; otherwise strip the trailing digits,
# apply the known alias, and REFUSE an unknown group rather than quietly skipping it.
ALIAS = {"oodfov": "ood_fov"}
CAM_GROUP = {}

def group_of(cam):
    if cam in CAM_GROUP:
        return CAM_GROUP[cam]
    if cam == "canonical":
        return None
    base = re.sub(r"\d+$", "", cam)
    g = ALIAS.get(base, base)
    if g not in G:
        raise ValueError(f"camera {cam!r} maps to unknown group {g!r}; "
                         "pass --index <dataset>/index.json")
    return g

def load(path):
    z = np.load(path)
    pred, lab = {}, {}
    for k in z.files:
        kind, key, cam = k.split("|", 2)
        (pred if kind == "pred" else lab)[(key, cam)] = z[k].astype(np.float64)
    return pred, lab

def systems_separate(name, pred):
    """name -> {(key, cam): curve or None} over battery cameras; None = not defined."""
    out = {f"{name} battery": {}, f"{name} canonical": {}, f"{name} mean": {}}
    for (key, cam), p in pred.items():
        if group_of(cam) not in G:
            continue
        pc = pred.get((key, "canonical"))
        if pc is None or len(pc) != len(p):
            raise ValueError(f"{name}: no matching canonical curve for {key}")
        out[f"{name} battery"][(key, cam)] = p
        out[f"{name} canonical"][(key, cam)] = pc
        out[f"{name} mean"][(key, cam)] = 0.5 * (p + pc)
    return out

def add_oracles(name, sysd, labels):
    b, c = sysd[f"{name} battery"], sysd[f"{name} canonical"]
    om, ot = {}, {}
    for kc, y in labels.items():
        if kc not in b:
            continue
        pb, pc = b[kc], c[kc]
        om[kc] = pb if np.abs(pb - y).mean() <= np.abs(pc - y).mean() else pc
        ot[kc] = pb if kendall(y, pb) >= kendall(y, pc) else pc
    sysd[f"{name} best-MAE*"] = om
    sysd[f"{name} best-tau*"] = ot

def metrics(curves, labels, group, oracle):
    ks = [kc for kc in curves if group_of(kc[1]) == group]
    lab = [kc for kc in ks if kc in labels]
    if not lab:
        return None
    ae = np.concatenate([np.abs(curves[kc] - labels[kc]) for kc in lab])
    tau = np.mean([kendall(labels[kc], curves[kc]) for kc in lab])
    allp = np.concatenate([curves[kc] for kc in ks])
    ally = np.concatenate([labels[kc] for kc in lab])
    res = dict(mae=float(ae.mean()), tau=float(tau), n=len(lab),
               spread=float(allp.std() / ally.std()))
    if not oracle:
        by = defaultdict(list)
        for kc in ks:
            by[kc[0]].append(curves[kc])
        sv, ep = [], []
        for key, cs in by.items():
            if len(cs) < 2:
                continue
            M = np.stack(cs)
            sv.append(M.std(0).mean())
            d = np.abs(M[:, None] - M[None]); iu = np.triu_indices(len(cs), 1)
            ep.append(d[iu].mean())
        res.update(s_view=float(np.mean(sv)), e_pair=float(np.mean(ep)),
                   share=float(np.mean(sv) / allp.std()))
    return res

def paired(a, b, labels, group, reps=2000, seed=0):
    """b - a per labelled clip, bootstrap over TRAJECTORIES (clips of one trajectory are
    not independent)."""
    ks = [kc for kc in labels if group_of(kc[1]) == group and kc in a and kc in b]
    if not ks:
        return None
    by = defaultdict(list)
    for kc in ks:
        y = labels[kc]
        by[kc[0]].append((kendall(y, b[kc]) - kendall(y, a[kc]),
                          np.abs(b[kc] - y).mean() - np.abs(a[kc] - y).mean()))
    trajs = list(by)
    per = np.array([np.mean(by[t], axis=0) for t in trajs])          # (n_traj, 2)
    rng = np.random.default_rng(seed)
    boot = per[rng.integers(0, len(trajs), (reps, len(trajs)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    win = np.mean([d[0] > 0 for t in trajs for d in by[t]])
    return per.mean(0), lo, hi, win, len(trajs)

ap = argparse.ArgumentParser()
ap.add_argument("--sep", action="append", default=[], help="NAME=single_view_dump.npz")
ap.add_argument("--joint", action="append", default=[], help="NAME=joint_dump.npz")
ap.add_argument("--vs", action="append", default=[], help="JOINT_NAME:SYSTEM_NAME paired test")
ap.add_argument("--index", default="", help="dataset index.json: camera -> group, overrides names")
a = ap.parse_args()
if a.index:
    for e in json.load(open(a.index))["index"]:
        CAM_GROUP[e["cam"]] = None if e["group"] == "canonical" else e["group"]

systems, labels = {}, {}
for spec in a.sep:
    name, path = spec.split("=", 1)
    pred, lab = load(path)
    for kc, y in lab.items():
        if group_of(kc[1]) in G:
            labels.setdefault(kc, y)
    s = systems_separate(name, pred)
    add_oracles(name, s, labels)
    systems.update(s)
for spec in a.joint:
    name, path = spec.split("=", 1)
    pred, lab = load(path)
    systems[name] = {kc: p for kc, p in pred.items() if group_of(kc[1]) in G}
    for kc, y in lab.items():
        if kc in labels and not np.allclose(labels[kc], y, atol=1e-5):
            raise ValueError(f"label mismatch for {kc} between dumps")

for g in G:
    if all(metrics(cv, labels, g, True) is None for cv in systems.values()):
        continue
    print(f"\n### {g}")
    print(f"{'system':26s}{'MAE':>8s}{'tau':>8s}{'spread':>8s}{'σ_view占比':>11s}{'e_pair':>9s}{'n':>5s}")
    for nm, cv in systems.items():
        orc = nm.endswith("*")
        m = metrics(cv, labels, g, orc)
        if m is None:
            continue
        cons = f"{100*m['share']:10.1f}%{m['e_pair']:9.4f}" if not orc else f"{'—':>11s}{'—':>9s}"
        print(f"{nm:26s}{m['mae']:8.4f}{m['tau']:+8.3f}{m['spread']:8.2f}{cons}{m['n']:5d}")
if a.vs:
    print("\n=== 成对比较（逐 clip，按轨迹 bootstrap 95% CI）  Δ = 前者 − 后者 ===")
    for spec in a.vs:
        jn, sn = spec.split(":", 1)
        for g in G:
            r = paired(systems[sn], systems[jn], labels, g)
            if r is None:
                continue
            (dt, dm), lo, hi, win, nt = r
            print(f"  {jn} vs {sn:22s} {g:8s} Δτ {dt:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]  "
                  f"ΔMAE {dm:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]  τ胜率 {win:.0%}  ({nt} 轨迹)")
