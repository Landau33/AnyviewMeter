"""Order sensitivity of the causal two-view setup, on checkpoints trained with it.

Frames are interleaved v0_t0, v1_t0, v0_t1, ... under Qwen3-VL's CAUSAL mask, so at
every instant the first camera cannot see the second camera's frame of that instant
while the second can see the first.  Evaluation always put the battery camera first
and the fixed canonical camera second, and fused by averaging the two per-view progress
tokens -- i.e. it averaged a token that saw both views with one that saw only its own.

Each test pair is run in BOTH orders and every readout is scored:
    A = [battery, canonical]    B = [canonical, battery]
    first-prog = the view that is blind to the other at that instant
    last-prog  = the view that sees both
and two disagreements are measured that need no labels:
    order     |mean_A - mean_B|             -- does swapping the cameras move the output
    same-cam  |battery blind - battery seeing| -- how much one camera's own token depends
                                                on whether it saw the other one
"""
import sys, os, json, argparse, time, numpy as np, torch
for p in ("~/avm_train/robometer", "~/avm/AnyviewMeter", "~/avm/AnyviewMeter/scripts"):
    sys.path.insert(0, os.path.expanduser(p))
import train_viewpoint as T

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--arm", required=True)
ap.add_argument("--inject", default="patch"); ap.add_argument("--data", required=True)
ap.add_argument("--n-traj", type=int, default=40); ap.add_argument("--out", required=True)
ap.add_argument("--block", action="store_true", help="run with the block-causal mask")
ap.add_argument("--frames", type=int, default=0,
                help="0 = full clip (as the training script evaluates).  K = K evenly spaced "
                     "frames, the SAME indices for both views.  The comparisons this script "
                     "exists for -- order A vs B, first vs last token -- hold on any common "
                     "frame set; absolute MAE/tau then differ from the full-clip results.json.")
a = ap.parse_args()
dev = torch.device("cuda")
ns = argparse.Namespace(arm=a.arm, inject=a.inject, xattn_layers="27,31,35", control_dim=256,
    bottleneck=256, gate_init=0.05, gate_open=None, pose_dropout=0.0, pose_null=False,
    pose_source="true", workspace_centre=[0, 0, 0], encoder_init="zero", views=2,
    view_embed=False, rayqk_only=False, block_mask=a.block, mv_readout="mean", lora_r=32, lora_layers=36, lora_dropout=0.05,
    lora_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
m, up = T.build_model(a.arm, T.load_robometer(device=dev), dev, ns)
T.add_lora(m.rbm, ns)
T.load_state(m, torch.load(a.ckpt, map_location=dev), up)
m.rbm.eval()
if up: m.injector.eval()
print(f"loaded {a.ckpt}  use_pose={up}", flush=True)

@torch.no_grad()
def raw(frames_list, metas):
    n, Tn = len(frames_list), len(frames_list[0])
    m._view_slots = None
    inp = m.build_inputs(T.interleave(frames_list), metas[0]["prompt"], dev)
    if up: m.set_pose(T.cam_params_multi(metas, Tn, dev), training=False)
    else: m.clear_pose()
    m.block_views = n if a.block else None
    try:
        _, lg = m(inp, return_logits=True)
    finally:
        m.block_views = None
    p = lg.float().softmax(-1).reshape(Tn, n, -1)
    return (p * T.bin_centres(dev)).sum(-1).cpu().numpy()          # (T, n)

store = T.ClipStore(a.data, "test", "mask")
G = ["az_seen", "az_edge", "az_far", "ood_fov"]
R = ["A_mean", "B_mean", "A_first", "A_last", "B_first", "B_last"]
acc = {g: {r: dict(ae=[], tau=[]) for r in R} for g in G}
dis = {g: dict(order=[], batt=[], canon=[]) for g in G}
t0, n_pair = time.time(), 0
for key in store.keys[:a.n_traj]:
    cams = store.cams_in(key)
    if "canonical" not in cams: continue
    fc, mc = store.load(key, "canonical")
    idx = None
    if a.frames > 0:
        idx = np.linspace(0, len(fc) - 1, a.frames).round().astype(int).tolist()
        fc, mc = store.load(key, "canonical", idx)
    for cam in cams:
        g = store.cam_group[cam]
        if cam == "canonical" or g not in G: continue
        fb, mb = store.load(key, cam, idx)
        A = raw([fb, fc], [mb, mc])        # col0 battery (blind), col1 canonical (sees)
        B = raw([fc, fb], [mc, mb])        # col0 canonical (blind), col1 battery (sees)
        pr = dict(A_mean=A.mean(1), B_mean=B.mean(1), A_first=A[:, 0], A_last=A[:, 1],
                  B_first=B[:, 0], B_last=B[:, 1])
        dis[g]["order"].append(float(np.abs(A.mean(1) - B.mean(1)).mean()))
        dis[g]["batt"].append(float(np.abs(A[:, 0] - B[:, 1]).mean()))
        dis[g]["canon"].append(float(np.abs(A[:, 1] - B[:, 0]).mean()))
        if mb["progress"] is not None:
            y = np.asarray(mb["progress"], dtype=np.float64)
            for r, v in pr.items():
                acc[g][r]["ae"].append(np.abs(v - y)); acc[g][r]["tau"].append(T.kendall_tau(y, v))
        n_pair += 1
        if n_pair % 40 == 0:
            print(f"  {n_pair} pairs  {time.time()-t0:.0f}s", flush=True)

res = {g: {r: dict(mae=float(np.mean(np.concatenate(acc[g][r]["ae"]))),
                   tau=float(np.mean(acc[g][r]["tau"]))) for r in R} for g in G}
for g in G:
    res[g]["disagree"] = {k: float(np.mean(v)) for k, v in dis[g].items()}
json.dump(dict(ckpt=a.ckpt, n_traj=a.n_traj, n_pairs=n_pair, frames=a.frames, block=a.block,
               res=res), open(a.out, "w"), indent=1)
print(f"\n=== {os.path.basename(os.path.dirname(a.ckpt))}   {n_pair} pairs ===")
print(f"{'readout':10s}" + "".join(f"{g:>17s}" for g in G))
for r in R:
    print(f"{r:10s}" + "".join(f"{res[g][r]['mae']:.4f}/{res[g][r]['tau']:+.3f}".rjust(17) for g in G))
for k, lbl in [("order", "|A-B| 换位"), ("batt", "battery 盲vs见"), ("canon", "canon 盲vs见")]:
    print(f"{lbl:12s}" + "".join(f"{res[g]['disagree'][k]:17.4f}" for g in G))
