"""R1 (RGB LoRA) vs P2 (LoRA + Plucker + view consistency) on unseen camera poses.

THE QUESTION.  Given the same trajectories, the same RGB, the same LoRA and the same
step budget, does adding camera conditioning make the progress model more accurate and
more self-consistent on viewpoints it never trained on?

WHAT MAKES THE COMPARISON HONEST.

  * R1 and P2 share everything except the pose pathway.  Same clips, same sampled
    timesteps (same seed), same LoRA config, same steps, same LR schedule.  P2's only
    extra input is the camera.
  * Labels come from simulator state, not frame index, so neither arm can win by
    counting frames.  ``--report-frame-counter`` prints what a pure frame counter
    scores on the same test set; any result at or below that line is not a result.
  * Model selection is on VALIDATION MAE over the held-out camera group(s) named by
    --val-group, never on training loss.
  * Test cameras are split into groups and reported separately.  A canonical-only
    number would hide exactly the effect being measured.

MEMORY.  The view-consistency term needs the same states seen from two cameras.  Both
forwards will not fit next to a 4B backbone, so camera B runs under ``no_grad`` and
contributes only its (T, 10) probabilities; gradient flows through camera A alone.  The
roles swap every step, so the term is symmetric in expectation rather than per step --
that asymmetry is deliberate and is why ``--no-swap`` exists to test it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from anyviewmeter.geometry.camera import CameraParams  # noqa: E402
from anyviewmeter.models.robometer_backbone import (PoseConditionedRobometer,  # noqa: E402
                                                    load_robometer)

N_BINS = 10


# ------------------------------------------------------------------------ data
class ClipStore:
    """index.json -> clips grouped by physical trajectory.

    The grouping is the point: the view-consistency term and every consistency metric
    need two renders of the SAME states, which only exist inside one trajectory group.
    """

    def __init__(self, root: str, split: str, fail_policy: str = "mask"):
        self.root = root
        self.fail_policy = fail_policy
        idx = json.load(open(f"{root}/index.json"))
        self.battery = idx["battery"]
        self.groups: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.cam_group: Dict[str, str] = {}
        self.kind: Dict[str, str] = {}
        for r in idx["index"]:
            if r["split"] != split:
                continue
            key = f"{r['traj']}__{r['kind']}"
            self.groups[key][r["cam"]] = f"{root}/{r['path']}"
            self.cam_group[r["cam"]] = r["group"]
            self.kind[key] = r["kind"]
        self.keys = sorted(self.groups)
        if not self.keys:
            raise RuntimeError(f"no clips for split={split} under {root}")

    def labeled(self, key: str) -> bool:
        """Whether this trajectory carries progress supervision.

        CORRECTED 2026-09-04, having read their source rather than inferred it.
        ``_compute_absolute_first_frame_progress`` never consults ``quality_label``, so
        Robometer DOES give a failed episode the same progress ramp as a success; what
        it does differently is ``compute_success_labels``, which returns all-zeros for
        ``failure/suboptimal``, plus a ``quality_mask`` forcing every frame of such a
        trajectory into the success loss.  The failure lives in the SUCCESS head, not in
        the progress target.  The earlier claim here -- that they never assign a curve
        to a failure -- was wrong.

          ``mask``      failures carry NO progress target.  They still train and score
                        on view CONSISTENCY, which needs no ground truth.  This is what
                        every result before 2026-09-04 used.
          ``zero_last`` all-zero progress.
          ``robometer`` the official shape: failures keep a progress target and the
                        failure is carried by the success head (needs --success-head).

        Successes and recoveries keep the state-derived label either way -- that is the
        part that stops a frame counter from scoring, and it was not in scope to change.
        """
        return (not self.kind[key].startswith("failure")
                or self.fail_policy in ("zero_last", "robometer"))

    def cams_in(self, key: str, group: Optional[str] = None) -> List[str]:
        cams = sorted(self.groups[key])
        if group:
            cams = [c for c in cams if self.cam_group[c] == group]
        return cams

    def load(self, key: str, cam: str, steps=None):
        # inlined rather than imported from scripts/probe_lib.py: that module lives in
        # the render-side repo, which is not checked out on the training box
        z = np.load(self.groups[key][cam], allow_pickle=False)
        frames = z["frames"]
        meta = json.loads(str(z["meta"]))
        if "progress" not in meta:
            # phase-A clips carry no label.  Fall back to Robometer's own convention,
            # ``absolute_first_frame``: progress = (i - start) / (N - start - 1) with
            # start = 0, i.e. the linear ramp it was trained against.  Evaluated on the
            # FULL clip before subsetting, so a sampled frame keeps its absolute progress.
            meta["progress"] = list(np.linspace(0.0, 1.0, len(frames)))
        if steps is not None:
            frames = frames[steps]
            meta = dict(meta, progress=[meta["progress"][i] for i in steps])
        meta["is_failure"] = self.kind[key].startswith("failure")
        if meta["is_failure"] and self.fail_policy != "robometer":
            n = len(frames)
            meta = dict(meta, progress=([0.0] * n if self.fail_policy == "zero_last"
                                        else None))
        # Under 'robometer' the progress target is left exactly as it is: their ramp is
        # computed without consulting quality_label, and the failure is expressed through
        # the success head instead.  Note ours is a STATE-BASED curve rather than their
        # linear ramp, so a failure here still ends low -- keeping it is strictly more
        # information than the official recipe has, not less.
        return frames, meta


def sample_steps(rng: random.Random, n_src: int, t: int) -> List[int]:
    """t timesteps in increasing order, deliberately NOT evenly spaced.

    Even spacing would make frame index an affine function of true time within a clip,
    which reintroduces exactly the shortcut the state-based labels exist to remove.
    """
    return sorted(rng.sample(range(n_src), t))


def per_frame(x, n_frames: int, shape) -> torch.Tensor:
    """One matrix for the whole clip (static camera) or one per frame (moving camera, e.g. a
    wrist camera); either way -> (n_frames, *shape).  A per-frame list must already be
    subsampled to the clip's frames -- a length mismatch pairs rays with the wrong image."""
    t = torch.tensor(x, dtype=torch.float32)
    if t.dim() == len(shape):
        return t[None].expand(n_frames, *shape)
    if t.shape[0] != n_frames:
        raise ValueError(f"per-frame camera parameters have {t.shape[0]} entries for {n_frames} frames")
    return t


def cam_params_of(meta, n_frames: int, device) -> CameraParams:
    cp = meta["cam_params"]
    K = per_frame(cp["intrinsic_cv"], n_frames, (3, 3))
    E = per_frame(cp["extrinsic_cv"], n_frames, (3, 4))
    return CameraParams(K.to(device), E.to(device), int(meta["res"]), int(meta["res"]))


def cam_params_multi(metas, n_frames: int, device) -> CameraParams:
    """One CameraParams over N views' frames INTERLEAVED as view0_t0, view1_t0, view0_t1...

    The order has to match how the frames are stacked, because ``_inject_visual`` maps
    Plucker token k of slot j onto patch k of image j positionally -- a mismatch is a
    silent pairing of one camera's rays with another camera's pixels, which trains and
    evaluates without ever raising.

    THIS IS ALSO THE ONLY SETTING WHERE THE PLUCKER GEOMETRY IS NON-DEGENERATE.  Every
    ray of a single camera passes through one optical centre, so the reciprocal product
    d_i.m_j + d_j.m_i is identically zero for any two rays of that clip and the geometry
    carries nothing beyond a per-clip tag.  Rays from two centres are skew and the
    product is their signed distance -- the quantity triangulation is made of.
    """
    Ks = [per_frame(m["cam_params"]["intrinsic_cv"], n_frames, (3, 3)) for m in metas]
    Es = [per_frame(m["cam_params"]["extrinsic_cv"], n_frames, (3, 4)) for m in metas]
    K, E = [], []
    for t in range(n_frames):
        for v in range(len(metas)):
            K.append(Ks[v][t])
            E.append(Es[v][t])
    res = int(metas[0]["res"])
    return CameraParams(torch.stack(K).to(device), torch.stack(E).to(device), res, res)


def interleave(frames_list):
    """N arrays of (T,H,W,3) -> (N*T,H,W,3) as view0_t0, view1_t0, view0_t1, ...

    Interleaved rather than concatenated so the two views of one instant sit next to
    each other in the sequence.  Robometer inserts a progress token after every frame,
    so this also puts the two per-view predictions for one timestep adjacent, which is
    what makes the fusion below a local operation.
    """
    import numpy as np
    a = [np.asarray(f) for f in frames_list]
    t = a[0].shape[0]
    if any(x.shape[0] != t for x in a):
        raise ValueError(f"views disagree on frame count: {[x.shape[0] for x in a]}")
    return np.stack(a, axis=1).reshape(-1, *a[0].shape[1:])


def forward_probs_multi(model, frames_list, metas, device, use_pose: bool,
                        training: bool, grad: bool = True):
    """-> (T, N_BINS) for ONE timestep sequence seen from N cameras at once.

    Fusion is the mean of the per-view probabilities, deliberately parameter-free: a
    learned fuser would add capacity on top of the multi-view input and the result
    could no longer be attributed to seeing two views.  The views are not independent
    here anyway -- they are in one attention sequence and have already exchanged
    information before the head sees them.
    """
    n = len(frames_list)
    frames = interleave(frames_list)
    # interleave() stacks as view0_t0, view1_t0, view0_t1, ... so the view index of
    # frame f is simply f % n.  Set before build_inputs so the visual hook sees it.
    model._view_slots = torch.arange(len(frames), device=device) % n
    inp = model.build_inputs(frames, metas[0]["prompt"], device)
    if use_pose:
        model.set_pose(cam_params_multi(metas, len(frames_list[0]), device),
                       training=training)
    else:
        model.clear_pose()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    if getattr(model, "block_mask_enabled", False):
        model.block_views = n
    try:
        with ctx:
            _, logits = model(inp, return_logits=True)
    finally:
        model.block_views = None
    p = logits.float().softmax(-1)
    if p.shape[0] != n * len(frames_list[0]):
        raise ValueError(f"head returned {p.shape[0]} predictions for "
                         f"{n * len(frames_list[0])} frames")
    p = p.reshape(-1, n, p.shape[-1])
    # 'last' reads only the final view's progress token of each instant -- under the
    # causal mask the one that has seen every view of that instant, under the block
    # mask one token per instant rather than an average of several.
    return p[:, -1] if getattr(model, "mv_readout", "mean") == "last" else p.mean(1)


# ---------------------------------------------------------------------- losses
def soft_bin_targets(y: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """Continuous y in [0,1] -> (T, n_bins) mass split over the two nearest bin centres.

    A hard argmax target would throw away everything the continuous label knows: a
    y of 0.549 and one of 0.551 would land in different bins and look maximally
    different, while y=0.51 and y=0.59 would look identical.
    """
    centres_pos = y.clamp(0, 1) * n_bins - 0.5
    lo = torch.floor(centres_pos)
    frac = (centres_pos - lo).clamp(0, 1)
    lo = lo.long()
    q = y.new_zeros(y.shape[0], n_bins)
    lo_c = lo.clamp(0, n_bins - 1)
    hi_c = (lo + 1).clamp(0, n_bins - 1)
    q.scatter_add_(1, lo_c[:, None], (1 - frac)[:, None])
    q.scatter_add_(1, hi_c[:, None], frac[:, None])
    return q / q.sum(-1, keepdim=True).clamp_min(1e-8)


def bin_centres(device, n_bins: int = N_BINS) -> torch.Tensor:
    return (torch.arange(n_bins, device=device, dtype=torch.float32) + 0.5) / n_bins


def bin_weights(store, device, alpha: float = 0.5, n_bins: int = N_BINS) -> torch.Tensor:
    """Per-bin loss weights ~ freq^-alpha, normalised so the mean weight is 1.

    The label distribution on this task is badly skewed: 63% of all frames land in bins
    3-4 (progress 0.3-0.5, the approach-and-grasp plateau) and under 7% sit above 0.8.
    Unweighted, the cheapest way to cut the loss is to predict the middle of that mass
    and stop looking -- which is exactly what the previous run did, ending at a near
    constant 0.47 with prediction std 0.02 against a label std 0.20.

    alpha=0.5 (square-root inverse frequency) rather than full inverse frequency: at
    alpha=1 the rarest bin gets ~100x the weight of the densest one here, and a handful
    of frames then dominate every gradient.
    """
    counts = np.zeros(n_bins)
    for key in store.keys:
        if not store.labeled(key):
            continue
        _, m = store.load(key, store.cams_in(key)[0])
        y = np.asarray(m["progress"], dtype=float)
        np.add.at(counts, np.clip((y * n_bins).astype(int), 0, n_bins - 1), 1)
    freq = counts / max(counts.sum(), 1)
    w = np.where(freq > 0, np.maximum(freq, 1e-6) ** (-alpha), 0.0)
    w = w / max((freq * w).sum(), 1e-8)          # E[w] = 1 under the data distribution
    return torch.tensor(w, dtype=torch.float32, device=device)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon between two (T, K) categorical rows, averaged over T.

    JS rather than KL because the two views are symmetric claims about the same state:
    neither is the reference, and JS stays bounded when one view is confidently wrong.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.log() - b.log())).sum(-1)  # noqa: E731
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()


def lambda_view_at(step: int, warm: int, ramp: int, peak: float) -> float:
    """0 while the model still cannot predict progress, then linear to `peak`.

    Turning consistency on from step 0 has an obvious degenerate optimum -- agree on
    the same wrong answer -- and it is cheapest to reach before the progress terms have
    any grip.  Delay-then-ramp is the standard guard.
    """
    if step < warm:
        return 0.0
    if step >= warm + ramp:
        return peak
    return peak * (step - warm) / max(ramp, 1)


# ----------------------------------------------------------------------- model
def build_model(arm: str, bundle, device, args):
    """R0/R1 carry no pose pathway at all; P1/P2 add one.

    WHERE the pose enters is the single biggest lever measured so far, bigger than the
    arm, the gate schedule or the step budget.  ``--inject xattn`` (tier C) hands the
    decoder a control sequence with no correspondence between a token and its own ray,
    so the alignment has to be learned through attention inside a frozen trunk -- and it
    never is: the injection sits at 3-5% of ||hidden|| and blanking the camera changes
    nothing.  ``--inject patch`` (tier A) puts feature k on patch k, before the LLM
    rather than in its last third, and the camera starts being read.
    """
    use_pose = arm in ("P1", "P2")
    layers = [int(x) for x in args.xattn_layers.split(",") if x.strip()]
    if args.inject == "patch":
        layers = []                      # no cross-attention blocks at all
    gate_init = args.gate_init if args.gate_open is None else args.gate_open
    # Tier D is the only tier whose geometry term is a function of a PAIR of rays --
    # <pe_q_i, pe_k_j> is the Plucker reciprocal product, which is what relates two
    # cameras.  Measured on this data: 3e-08 within one camera (i.e. identically
    # zero, every ray through one centre intersects every other) and 0.24-0.45
    # across two, against a ray scale of 0.60.  So it is only worth running with
    # --views 2, and build_model says so rather than letting it train on nothing.
    name = "qk_pe" if args.inject == "rayqk" else "cross_attn"
    if args.inject == "rayqk":
        if args.views < 2:
            raise ValueError("--inject rayqk with --views 1: every ray of a single "
                             "camera passes through one optical centre, so the tier-D "
                             "geometric term is identically zero and the run would "
                             "measure nothing.  Use --views 2.")
        layers = layers or [27, 31, 35]
    model = PoseConditionedRobometer(
        bundle, injector_name=name, control_dim=args.control_dim,
        bottleneck=args.bottleneck, layers=layers,
        # rayqk normally rides ON TOP of tier A, so its Q/K term is a 0.6% increment
        # over a 380% patch injection and the two cannot be told apart.  --rayqk-only
        # turns tier A off so the arm measures what the reciprocal product is worth by
        # itself.
        patch_inject=(args.inject in ("patch", "both")
                      or (args.inject == "rayqk" and not args.rayqk_only)),
        pose_dropout=args.pose_dropout if use_pose else 0.0,
        freeze_backbone=True, gate_init=gate_init)
    model.injector.to(device)
    model.pose_null = args.pose_null
    model.block_mask_no_own_prog = bool(getattr(args, "block_mask_no_own_prog", False))
    model.block_mask_enabled = (bool(getattr(args, "block_mask", False))
                                or model.block_mask_no_own_prog)
    model.mv_readout = getattr(args, "mv_readout", "mean")
    if model.block_mask_enabled:
        if args.views < 2:
            raise ValueError("--block-mask with --views 1: there is one camera per instant, "
                             "so the block mask is the causal mask and measures nothing")
        model.install_block_mask_hook()
    if getattr(args, "view_embed", False):
        if args.views < 2:
            raise ValueError("--view-embed with --views 1: there is only one view, so "
                             "the tag is a constant and the arm measures nothing")
        import torch.nn as _nn
        h = model.rbm.model.language_model.config.hidden_size
        model.view_embed = _nn.Embedding(args.views, h).to(device)
        # zero-init, same house rule as tier A: step 0 is the untouched backbone
        _nn.init.zeros_(model.view_embed.weight)
        model.attach()          # R1 installs no hooks otherwise
    model.pose_head = None
    model.pose_random = (args.pose_source == "random")
    model.workspace_centre = args.workspace_centre
    if use_pose and args.pose_source == "learned":
        from anyviewmeter.models.pose_head import PoseHead
        model.pose_head = PoseHead().to(device)
        # THE ZERO-INIT HAS TO GO, and this is not a tuning preference.  Tier A ships with
        # encoder.out zeroed so the frozen trunk is untouched at step 0.  With a LEARNED
        # pose that zero also cuts the gradient to everything UPSTREAM of the encoder --
        # measured: |d loss / d(az,el,r,fov)| is exactly 0 at out.weight = 0 and 1e-1 at
        # N(0,1e-3).  So the pose head would sit at its random init while the encoder --
        # which does get gradient -- trains on those random poses, and by the time the head
        # could move, the encoder has had every reason to learn that its pose input is
        # noise.  A chicken-and-egg that ends in "ignore the camera", which is precisely
        # the failure this whole line of work is about.
        args.encoder_init = "small"
    if use_pose and args.encoder_init == "small":
        with torch.no_grad():
            model.injector.encoder.out.weight.normal_(0.0, 1e-3)
    if use_pose:
        model.attach()
    return model, use_pose


def add_lora(rbm, args):
    """LoRA on the last `n` decoder layers' Q/V, injected IN PLACE.

    ``inject_adapter_in_model`` rather than ``get_peft_model``: the latter wraps the
    model in a PeftModel, which moves ``model.language_model.layers`` one level deeper
    and silently breaks the forward hooks the pose pathway is installed through.
    """
    from peft import LoraConfig, inject_adapter_in_model
    n_layers = len(rbm.model.language_model.layers)
    first = n_layers - args.lora_layers
    pat = "|".join(str(i) for i in range(first, n_layers))
    # The official recipe puts LoRA on all seven projections including the MLP; the
    # attention and MLP submodules sit under different parents, so one regex over
    # self_attn alone can never match gate/up/down and silently trains fewer modules
    # than the flag asks for.
    # getattr, not attribute access: probe and analysis scripts build their own Namespace
    # by hand and adding a required field here silently breaks every one of them.
    mods = [m.strip() for m in getattr(args, "lora_modules", "q_proj,v_proj").split(",")
            if m.strip()]
    attn = [m for m in mods if m in ("q_proj", "k_proj", "v_proj", "o_proj")]
    mlp = [m for m in mods if m in ("gate_proj", "up_proj", "down_proj")]
    unknown = set(mods) - set(attn) - set(mlp)
    if unknown:
        raise ValueError(f"unknown --lora-modules entries: {sorted(unknown)}")
    alts = ([rf"self_attn\.({'|'.join(attn)})"] if attn else []) + \
           ([rf"mlp\.({'|'.join(mlp)})"] if mlp else [])
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                     lora_dropout=getattr(args, "lora_dropout", 0.0), bias="none",
                     target_modules=rf".*language_model\.layers\.({pat})\.(" +
                                    "|".join(alts) + ")")
    inject_adapter_in_model(cfg, rbm)
    n = 0
    for name, p in rbm.named_parameters():
        p.requires_grad = "lora_" in name
        n += p.numel() if "lora_" in name else 0
    for p in rbm.progress_head.parameters():      # the head must move with the labels
        p.requires_grad = True
    if getattr(args, "success_head", False) and hasattr(rbm, "success_head"):
        for p in rbm.success_head.parameters():
            p.requires_grad = True
    heads = list(rbm.progress_head.parameters())
    if getattr(args, "success_head", False) and hasattr(rbm, "success_head"):
        heads += list(rbm.success_head.parameters())
    return n, sum(p.numel() for p in heads)


# -------------------------------------------------------------------- forward
def camera_random(meta, n, device, centre):
    """A camera drawn from the same ranges as a real one, but keyed on the CLIP ID.

    The control the tier-A story needs and has never had on the wide cone.  The learned
    -pose arm showed a 0.44M CNN emitting four numbers with NO relation to the true camera
    (probe: 50-92 deg angular error, 0% within 10 deg) can match true-pose conditioning.
    That raises a question the task numbers alone cannot answer: how much of P1's gain is
    the GEOMETRY, and how much is simply having a per-clip side channel into a frozen
    trunk?  This arm has the channel and no information: deterministic per clip, so it is
    a stable code rather than noise, and independent of both the image and the true pose.
    """
    import hashlib
    h = hashlib.sha256(f"{meta.get('traj','')}|{meta.get('cam','')}".encode()).digest()
    r = np.random.default_rng(int.from_bytes(h[:8], "little"))
    from anyviewmeter.geometry.plucker import look_at_camera
    t = lambda v: torch.full((n,), float(v), device=device)
    return look_at_camera(t(r.uniform(-np.pi, np.pi)), t(r.uniform(0.05, 1.05)),
                          t(r.uniform(0.42, 0.95)), t(r.uniform(0.60, 1.15)),
                          torch.as_tensor(centre, device=device, dtype=torch.float32), 256, 256)


def camera_from_head(model, frames, device):
    """Learned camera for one clip: the head sees the pixels, nothing else.

    ``meta`` is deliberately NOT passed -- if the true camera reached this function the
    experiment would be measuring nothing.  The workspace centre is a rig constant and is
    taken from the model, not from the clip.
    """
    from anyviewmeter.geometry.plucker import look_at_camera
    x = torch.as_tensor(frames, device=device)
    if x.dtype == torch.uint8:
        x = x.float().div(255.0)
    x = x.permute(0, 3, 1, 2) if x.shape[-1] == 3 else x
    az, el, r, fov = model.pose_head(x)
    centre = torch.as_tensor(model.workspace_centre, device=device, dtype=torch.float32)
    return look_at_camera(az, el, r, fov, centre, x.shape[-2], x.shape[-1])


def success_targets(y, is_failure: bool, args):
    """Robometer's success labels + the frame mask that decides which ones count.

    Two rules, both taken from their trainer rather than invented here:
      * a failure trajectory is all-zeros for every frame (compute_success_labels)
      * a frame enters the loss when progress < min_success OR it is a positive; the
        band in between is ambiguous and is dropped -- except that a failure/suboptimal
        trajectory has every frame forced in via quality_mask.
    """
    if is_failure:
        lab = torch.zeros_like(y)
        mask = torch.ones_like(y)                       # quality_mask
    else:
        lab = (y >= args.max_success).to(y.dtype)
        mask = ((y < args.min_success) | (lab > 0.5)).to(y.dtype)
    return lab, mask


def success_loss(logits, lab, mask):
    """Class-balanced BCE, weighting positives and negatives equally as they do."""
    logits = logits.clamp(-50.0, 50.0).reshape(-1)
    npos = (lab * mask).sum()
    nneg = ((1 - lab) * mask).sum()
    if mask.sum() < 1 or npos < 1 or nneg < 1:
        pw = torch.ones((), device=logits.device, dtype=logits.dtype)
    else:
        pw = (nneg / npos).to(logits.dtype)
    l = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, lab.to(logits.dtype), pos_weight=pw, reduction="none")
    return (l * mask).sum() / mask.sum().clamp_min(1.0)


def forward_probs(model, frames, meta, device, use_pose: bool, training: bool,
                  grad: bool = True):
    """-> (T, N_BINS) probabilities for one clip from one camera."""
    model._view_slots = None
    inp = model.build_inputs(frames, meta["prompt"], device)
    if use_pose:
        if getattr(model, "pose_head", None) is not None:
            cam = camera_from_head(model, frames, device)
        elif getattr(model, "pose_random", False):
            cam = camera_random(meta, len(frames), device, model.workspace_centre)
        else:
            cam = cam_params_of(meta, len(frames), device)
        model.set_pose(cam, training=training)
    else:
        model.clear_pose()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        _, logits = model(inp, return_logits=True)
        return logits.float().softmax(-1)


def train(model, store, device, args, use_pose: bool, log):
    params_head, params_pose, params_gate = [], [], []
    for n, p in model.rbm.named_parameters():
        if p.requires_grad:
            params_head.append(p)
    if use_pose:
        for n, p in model.injector.named_parameters():
            if not p.requires_grad:
                continue
            (params_gate if n.endswith("gate") else params_pose).append(p)

    pinned = args.gate_open is not None and bool(params_gate)
    if pinned:
        # GATE PINNED.  The learned gate is an experiment about the OPTIMISER: given the
        # choice it closes, even when held open for the first `gate_freeze` steps.
        # Removing the choice asks the different question -- with the injection
        # guaranteed, can the branch make pose pay?  Watch inj%, not the gate: a pinned
        # block can still make itself irrelevant by driving proj to zero.
        for p in params_gate:
            p.requires_grad_(False)
        params_gate = []

    groups = [dict(params=params_head, lr=args.lr_lora)]
    if params_pose:
        groups.append(dict(params=params_pose, lr=args.lr_pose))
    if params_gate:
        groups.append(dict(params=params_gate, lr=args.lr_gate))
    if getattr(model, "view_embed", None) is not None:
        groups.append(dict(params=list(model.view_embed.parameters()), lr=args.lr_pose))
    if getattr(model, "pose_head", None) is not None:
        # Trained from the progress loss like everything else here.  It gets its own group
        # only so the learning rate can differ: it is a from-scratch CNN sitting in front
        # of a frozen 4B trunk, and the pose lr suits it better than the LoRA lr.
        groups.append(dict(params=list(model.pose_head.parameters()), lr=args.lr_pose))
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warm = max(1, int(0.05 * args.steps))

    def base(s):
        return ((s + 1) / warm if s < warm
                else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))

    # GATE FREEZE.  Every parameter in the pose branch sits behind the factor tanh(g), so
    # its gradient is proportional to the gate -- and the gate's own lr is 3.3x the
    # branch's.  That is a race the gate wins: if it closes before the branch has learned
    # anything worth injecting, the branch then learns at a rate proportional to a
    # shrinking gate, stays useless, and the gate keeps closing.  The first 12k-step run
    # showed exactly that shape (0.050 -> 0.070 by step 200, then monotone decay to
    # ~0.002).  Holding the gate open for the first `gate_freeze` steps gives the branch a
    # fixed, non-zero injection to prove itself under; after that the gate is released and
    # the model is free to shut it -- so a gate that still decays afterwards is a finding
    # about the pose signal rather than about the optimiser.
    lambdas = [base] * len(groups)
    if params_gate and args.gate_freeze:
        lambdas[-1] = lambda s: 0.0 if s < args.gate_freeze else base(s)
        log(f"    gate frozen at tanh(g)={args.gate_init} for the first "
            f"{args.gate_freeze} steps")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)

    bw = bin_weights(store, device, args.bin_weight_alpha)
    log("    bin weights " + " ".join(f"{v:.2f}" for v in bw.tolist()))

    rng = random.Random(args.seed)
    model.rbm.model.eval()                      # frozen trunk stays in eval
    model.rbm.progress_head.train()
    if use_pose:
        model.injector.train()

    hist, t0 = [], time.time()
    best = None
    n_bad = 0
    for step in range(args.steps):
        lam = lambda_view_at(step, args.view_warmup, args.view_ramp, args.lambda_view)
        # an unlabelled (failure) trajectory contributes ONLY the consistency term, so
        # drawing one while lambda_view is still 0 would be a step with no loss at all
        key = rng.choice(store.keys)
        for _ in range(20):
            if store.labeled(key) or lam > 0:
                break
            key = rng.choice(store.keys)
        cams = store.cams_in(key)
        a, b = rng.sample(cams, 2)
        if args.swap and step % 2:
            a, b = b, a
        steps_t = sample_steps(rng, args.n_src_frames, args.frames)

        fa, ma = store.load(key, a, steps_t)
        fb, mb = store.load(key, b, steps_t)
        has_y = ma["progress"] is not None

        p_b = None
        if args.views > 1:
            # Both cameras go into ONE attention sequence, so this is genuine multi-view
            # inference rather than two single-view passes averaged afterwards.  It also
            # rules out the consistency loss, which is defined between two SEPARATE
            # forward passes and has nothing to relate here.
            p_a = forward_probs_multi(model, [fa, fb], [ma, mb], device, use_pose,
                                      True, grad=True)
        else:
            if lam > 0:
                p_b = forward_probs(model, fb, mb, device, use_pose, True,
                                    grad=False).detach()
            p_a = forward_probs(model, fa, ma, device, use_pose, True, grad=True)

        if has_y:
            y = torch.tensor(ma["progress"], dtype=torch.float32, device=device)
            q = soft_bin_targets(y)
            # per-frame weight = expected bin weight under the soft target, so a frame
            # in a rare progress band counts for more than one on the crowded plateau
            wt = (q * bw).sum(-1)
            l_bin = -(wt * (q * p_a.clamp_min(1e-8).log()).sum(-1)).mean()
            l_reg = (wt * ((p_a * bin_centres(device)).sum(-1) - y) ** 2).mean()
        else:
            l_bin = p_a.new_zeros(())
            l_reg = p_a.new_zeros(())
        l_view = js_divergence(p_a, p_b) if p_b is not None else p_a.new_zeros(())
        l_succ = p_a.new_zeros(())
        if args.success_head and has_y:
            sl = getattr(model, "last_success_logits", None)
            if sl is None:
                raise RuntimeError("--success-head but the backbone returned no success "
                                   "logits; the head is not on this checkpoint's "
                                   "progress path")
            # multi-view emits one logit per view-frame; the targets are per timestep
            if args.views > 1:
                sl = sl.reshape(-1, args.views).mean(1)
            lab, smask = success_targets(y, bool(ma.get("is_failure")), args)
            l_succ = success_loss(sl, lab, smask)
        loss = l_bin + 0.5 * l_reg + lam * l_view + args.lambda_success * l_succ

        # A single non-finite step permanently poisons the weights -- AdamW propagates
        # NaN into the moments and every later step is NaN too, which is indistinguishable
        # in the log from "the model diverged".  Skip the step instead, and say so.
        if not torch.isfinite(loss):
            n_bad += 1
            if n_bad <= 5:
                log(f"    !! non-finite loss at step {step} ({key}, cams {a}/{b}, "
                    f"bin={float(l_bin)} reg={float(l_reg)} view={float(l_view)}, "
                    f"p_a finite={bool(torch.isfinite(p_a).all())}) -- step skipped")
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()
        trainable = [p for g in groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gn):
            n_bad += 1
            if n_bad <= 5:
                log(f"    !! non-finite grad norm at step {step} ({key}) -- step skipped")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        hist.append(dict(step=step, loss=float(loss), bin=float(l_bin), reg=float(l_reg),
                         view=float(l_view), succ=float(l_succ), lam=lam, gnorm=float(gn)))

        if step % args.log_every == 0 or step == args.steps - 1:
            w = hist[-args.log_every:]
            # Peak allocation since the last log, not since process start: the OOM
            # question is about the steady state, and the very first step allocates
            # optimiser moments that make a since-start peak unrepresentative.
            # Same role as tier A's inj=[patch:..]: a zero-initialised tag that never
            # leaves zero trains, logs and evaluates exactly like a working one.
            ve = ""
            if getattr(model, "view_embed", None) is not None:
                ve = (f" ve={model.last_view_rel * 100:.1f}%"
                      f"/|w|={float(model.view_embed.weight.norm()):.4f}")
            mem = ""
            if torch.cuda.is_available():
                mem = f" mem={torch.cuda.max_memory_allocated() / 2**30:.2f}G"
                torch.cuda.reset_peak_memory_stats()
            g = ""
            if use_pose:
                gv = (model.injector.gate_values() if hasattr(model.injector, "gate_values") else {})
                g = (" gates=[" + " ".join(f"{v:+.3f}" for v in gv.values()) + "]"
                     if gv and not pinned else "")
                if hasattr(model.injector, "gate_telemetry"):
                    al = {k: v for k, v in model.injector.gate_telemetry().items()
                          if k.endswith("/alpha")}
                    # alpha starts at 0 so the trunk is bit-exact at step 0; unlike
                    # tier A's zero-init it still receives gradient (dL/dalpha =
                    # <dL/dq, pe_q>, non-zero because pe_q is not), so a flat 0 here
                    # means the geometry is being ignored, not that it cannot move.
                    g += " a=[" + " ".join(f"{v:+.3f}" for v in al.values()) + "]"
                g += (" inj=[" + " ".join(f"{k}:{v * 100:.1f}%" for k, v in
                                          model.injector.injection_ratios().items()) + "]")
            log(f"[{step:5d}/{args.steps}] loss={np.mean([h['loss'] for h in w]):.4f} "
                f"bin={np.mean([h['bin'] for h in w]):.4f} "
                f"reg={np.mean([h['reg'] for h in w]):.4f} "
                f"view={np.mean([h['view'] for h in w]):.4f} "
                f"succ={np.mean([h.get('succ', 0.0) for h in w]):.4f} lam={lam:.2f} "
                f"|g|={np.mean([h['gnorm'] for h in w]):.2f}{g}{ve}{mem} "
                f"{(time.time()-t0)/(step+1):.2f}s/step")

        if args.val_every and step and step % args.val_every == 0:
            sel = tuple(args.val_group.split(","))
            v = evaluate(model, args.val_store, device, args, use_pose,
                         groups_only=sel, max_traj=args.val_trajs)
            missing = [g for g in sel if g not in v]
            if missing:
                raise SystemExit(f"--val-group {args.val_group}: {missing} not in this "
                                 f"battery (has {sorted(v)}).  Model selection would "
                                 f"silently score nothing, so this is fatal.")
            # pool by labelled-frame count: with a graded ladder the selection group is
            # several rungs, and an unweighted mean would let the smallest one steer it
            wsel = np.array([v[g]["n_labeled"] for g in sel], dtype=float)
            wsel = wsel / max(wsel.sum(), 1e-8)
            d = {k: float(np.sum(np.array([v[g][k] for g in sel]) * wsel))
                 for k in ("mae", "tau", "pred_std", "label_std")}
            mae, spread = d["mae"], d["pred_std"] / max(d["label_std"], 1e-8)
            # ANTI-COLLAPSE GUARD.  Selecting on MAE alone actively rewards the
            # degenerate solution: the previous run's chosen checkpoint predicted a near
            # constant 0.47 (pred std 0.02 vs label std 0.20) and won on MAE and S_view
            # precisely because it had stopped varying.  A checkpoint whose predictions
            # span less than `collapse_floor` of the label spread is disqualified, not
            # scored -- the same logic as the standing E4 gate.
            ok = spread >= args.collapse_floor
            better = ok and (best is None or mae < best[0])
            log(f"    val {args.val_group} MAE {mae:.4f}  tau {d['tau']:+.3f}  "
                f"spread {spread:.2f}" + ("" if ok else "  COLLAPSED, not eligible")
                + ("  <- best" if better else ""))
            if better:
                best = (mae, step, {k: v_.detach().cpu().clone()
                                    for k, v_ in state_to_save(model, use_pose).items()})
            model.rbm.progress_head.train()
            if use_pose:
                model.injector.train()
    if n_bad:
        log(f"    !! {n_bad}/{args.steps} steps skipped as non-finite")
    return hist, best


def state_to_save(model, use_pose: bool):
    sd = {f"rbm.{k}": v for k, v in model.rbm.state_dict().items()
          if "lora_" in k or k.startswith("progress_head")
          or k.startswith("success_head")}
    if use_pose:
        sd.update({f"inj.{k}": v for k, v in model.injector.state_dict().items()})
    if getattr(model, "view_embed", None) is not None:
        sd.update({f"view_embed.{k}": v for k, v in model.view_embed.state_dict().items()})
    if getattr(model, "pose_head", None) is not None:
        # Without this the learned pose is thrown away at the end of the run, and the
        # probe that decides whether it is GEOMETRY or just a useful image code cannot be
        # run at all -- which is the only thing that makes a --pose-source learned result
        # interpretable.  Cost one line, cost of omitting it one whole run.
        sd.update({f"pose_head.{k}": v for k, v in model.pose_head.state_dict().items()})
    return sd


# ------------------------------------------------------------------ evaluation
def kendall_tau(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 2:
        return 0.0
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    return float(np.triu(dx * dy, 1).sum() / (0.5 * n * (n - 1)))


@torch.no_grad()
def evaluate(model, store, device, args, use_pose: bool, groups_only=None,
             max_traj: int = 0, pose_mode: str = "correct", log=None):
    """Per camera-group MAE / Kendall tau / S_view / E_pair on full-length clips.

    ``pose_mode`` drives the causal diagnostics: the checkpoint never changes, only
    what the pose pathway is handed.
    """
    model.rbm.eval()
    if use_pose:
        model.injector.eval()
    keys = store.keys[:max_traj] if max_traj else store.keys
    per_group = defaultdict(lambda: dict(ae=[], tau=[], preds={}, ys=[]))

    rng = np.random.default_rng(args.seed)
    for key in keys:
        for cam in store.cams_in(key):
            g = store.cam_group[cam]
            if groups_only and g not in groups_only:
                continue
            frames, meta = store.load(key, cam)
            # failures carry no progress target under the Robometer-style policy, so
            # they score CONSISTENCY (S_view / E_pair need no ground truth) but are
            # excluded from MAE and tau rather than being scored against an invented curve
            y = (np.asarray(meta["progress"], dtype=np.float64)
                 if meta["progress"] is not None else None)
            order = None
            if pose_mode == "black_rgb":
                frames = np.zeros_like(frames)
            elif pose_mode == "shuffled_frames":
                order = rng.permutation(len(frames))
                frames = frames[order]
                y = y[order] if y is not None else None
            mm = meta
            if pose_mode == "shuffled_pose":
                other = store.keys[(store.keys.index(key) + 1) % len(store.keys)]
                ocams = store.cams_in(other)
                _, om = store.load(other, ocams[rng.integers(len(ocams))], [0])
                mm = dict(meta, cam_params=om["cam_params"])
            elif pose_mode.startswith("perturbed"):
                deg = float(pose_mode.split(":")[1])
                mm = dict(meta, cam_params=perturb_cam(meta["cam_params"], deg, rng))
            up = use_pose and pose_mode != "zero_pose"
            if args.views > 1:
                # Pair each battery camera with the FIXED canonical view, so the group
                # label still describes the camera that varies -- the deployment shape
                # this is meant to model is one fixed overhead camera plus one that
                # moves, not two cameras that move together.
                partner = args.view_partner
                if partner not in store.cams_in(key) or partner == cam:
                    continue
                f2, m2 = store.load(key, partner)
                # The partner has to receive the SAME frame transform: an unshuffled
                # second view would quietly hand the shuffled-frames diagnostic the
                # correct temporal order back, and the diagnostic would report that
                # frame order does not matter.
                if pose_mode == "black_rgb":
                    f2 = np.zeros_like(f2)
                elif pose_mode == "shuffled_frames":
                    f2 = f2[order]
                p = forward_probs_multi(model, [frames, f2], [mm, m2], device, up,
                                        False, grad=False)
            else:
                p = forward_probs(model, frames, mm, device, up, False, grad=False)
            yhat = (p * bin_centres(device)).sum(-1).float().cpu().numpy()
            d = per_group[g]
            if y is not None:
                d["ae"].append(np.abs(yhat - y))
                d["tau"].append(kendall_tau(y, yhat))
                d["ys"].append(y)
            d["preds"].setdefault(key, {})[cam] = yhat

    out = {}
    for g, d in per_group.items():
        # S_view: spread across cameras at the SAME physical state, then averaged
        sv, ep = [], []
        for key, per_cam in d["preds"].items():
            M = np.stack([per_cam[c] for c in sorted(per_cam)])       # (V, T)
            if M.shape[0] < 2:
                continue
            sv.append(M.std(axis=0).mean())
            diff = np.abs(M[:, None, :] - M[None, :, :])
            iu = np.triu_indices(M.shape[0], 1)
            ep.append(diff[iu].mean())
        # Spread of the predictions against spread of the labels.  A model that has
        # collapsed to a constant scores a low MAE (the labels are concentrated) and a
        # perfect S_view (a constant is trivially view-invariant), so without this the
        # headline metrics reward exactly the degenerate solution.
        allp = np.concatenate([v for per in d["preds"].values() for v in per.values()])
        ally = np.concatenate(d["ys"]) if d["ys"] else np.array([np.nan])
        out[g] = dict(mae=float(np.mean(np.concatenate(d["ae"]))) if d["ae"] else float("nan"),
                      tau=float(np.mean(d["tau"])) if d["tau"] else float("nan"),
                      s_view=float(np.mean(sv)) if sv else float("nan"),
                      e_pair=float(np.mean(ep)) if ep else float("nan"),
                      pred_std=float(allp.std()), label_std=float(np.nanstd(ally)),
                      n_labeled=len(d["tau"]), n_traj=len(d["preds"]))
    return out


def perturb_cam(cp: dict, deg: float, rng) -> dict:
    """Rotate the extrinsic about a random axis by `deg` and shift the centre.

    The RGB is untouched, so this asks whether the model is reading the pose it is
    given or merely noticing that something was supplied.
    """
    E = np.array(cp["extrinsic_cv"], dtype=np.float64)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = np.deg2rad(deg)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
    E2 = E.copy()
    E2[:3, :3] = R @ E[:3, :3]
    E2[:3, 3] = E[:3, 3] + rng.normal(scale=0.05 * deg / 15.0, size=3)
    return dict(cp, extrinsic_cv=E2.tolist())


def frame_counter_baseline(store, args) -> Dict[str, dict]:
    """What a model that ignores pixels and reads only the frame index would score.

    Fit the mean label per timestep on TRAIN, then apply it everywhere.  Any arm that
    does not clear this is not doing perception, whatever its MAE looks like.
    """
    tr = ClipStore(args.data, "train", args.fail_policy)
    acc = np.zeros(args.n_src_frames)
    cnt = 0
    for key in tr.keys:
        _, m = tr.load(key, tr.cams_in(key)[0])
        if m["progress"] is None:
            continue
        acc += np.asarray(m["progress"])
        cnt += 1
    prior = acc / max(cnt, 1)
    out = defaultdict(lambda: dict(ae=[], tau=[]))
    for key in store.keys:
        for cam in store.cams_in(key):
            _, m = store.load(key, cam)
            if m["progress"] is None:
                continue
            y = np.asarray(m["progress"])
            out[store.cam_group[cam]]["ae"].append(np.abs(prior - y))
            out[store.cam_group[cam]]["tau"].append(kendall_tau(y, prior))
    # S_view/E_pair are identically 0 for a frame counter: it ignores the camera, so
    # every view of a state gets the same number.  That is exactly why S_view alone
    # can never be the criterion.
    return {g: dict(mae=float(np.mean(np.concatenate(d["ae"]))),
                    tau=float(np.mean(d["tau"])), s_view=0.0, e_pair=0.0,
                    n_labeled=len(d["tau"]), n_traj=0) for g, d in out.items()}


#: Report ORDER, not a whitelist -- ``fmt`` appends anything else the store contains,
#: so a battery with different group names (the azimuth ladder, the occlusion ladder)
#: still prints in full instead of silently showing nothing.
GROUPS = ["canonical", "id_random", "az_seen", "az_edge", "az_far",
          "ood_pose", "ood_fov", "occlusion", "occ_clear", "occ_occluded"]


def fmt(name: str, res: Dict[str, dict]) -> str:
    L = [f"{name}",
         f"{'group':<12} {'MAE':>8} {'tau':>8} {'S_view':>8} {'E_pair':>8} {'spread':>7} {'n_lab':>6}"]
    for g in GROUPS + [g for g in res if g not in GROUPS]:
        if g in res:
            r = res[g]
            sp = r.get("pred_std", float("nan")) / max(r.get("label_std", 1e-8), 1e-8)
            L.append(f"{g:<12} {r['mae']:8.4f} {r['tau']:8.3f} {r['s_view']:8.4f} "
                     f"{r['e_pair']:8.4f} {sp:7.2f} {r['n_labeled']:6d}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="P2", choices=["R0", "R1", "P1", "P2"])
    ap.add_argument("--data", default="/var/tmp/avm/data/pickcube_avm")
    ap.add_argument("--out", default="/var/tmp/avm/runs")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--n-src-frames", type=int, default=32)
    ap.add_argument("--train-trajs", type=int, default=0, help="0 = all")
    ap.add_argument("--val-trajs", type=int, default=8)
    ap.add_argument("--test-trajs", type=int, default=0)
    mask_options = ap.add_mutually_exclusive_group()
    mask_options.add_argument("--block-mask", action="store_true",
                    help="multi-view attention mask: causal across instants, bidirectional "
                         "among the views of one instant.  The default causal mask lets "
                         "the last camera of an instant see the others but not the "
                         "reverse, so the views are not interchangeable.")
    mask_options.add_argument("--block-mask-no-own-prog", action="store_true",
                    help="block-causal variant: each frame cannot attend to its own "
                         "progress token; other views and progress queries stay visible")
    ap.add_argument("--mv-readout", default="mean", choices=["mean", "last"],
                    help="fuse the per-view progress tokens of an instant by averaging, "
                         "or read only the last one")
    ap.add_argument("--view-embed", action="store_true",
                    help="R1 plus a learned per-view tag added to each frame's patches: "
                         "the camera INDEX and nothing else.  Separates 'knowing which "
                         "view a frame came from' from 'knowing the geometry', which a "
                         "ray map supplies together and tier A therefore cannot "
                         "attribute.")
    ap.add_argument("--rayqk-only", action="store_true",
                    help="with --inject rayqk, disable the tier-A patch add so only the "
                         "Q/K reciprocal-product term is left")
    ap.add_argument("--views", type=int, default=1,
                    help="cameras packed into ONE attention sequence per prediction.  "
                         "1 = the single-view setting every result so far used.  2 = "
                         "both views interleaved, which is also the only configuration "
                         "where the Plucker reciprocal product is non-zero.")
    ap.add_argument("--view-partner", default="canonical",
                    help="the fixed second camera used at eval time, so the reported "
                         "group still names the camera that varies")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-modules", default="q_proj,v_proj",
                    help="comma list from q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,"
                         "down_proj.  Robometer's own recipe is all seven at r=32; the "
                         "default here is deliberately smaller so the pose branch's "
                         "contribution stays measurable against LoRA's.")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-layers", type=int, default=9)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-pose", type=float, default=3e-4)
    ap.add_argument("--lr-gate", type=float, default=1e-3)
    ap.add_argument("--lambda-view", type=float, default=0.2)
    ap.add_argument("--view-warmup", type=int, default=100)
    ap.add_argument("--view-ramp", type=int, default=200)
    ap.add_argument("--inject", default="xattn",
                    choices=["xattn", "patch", "both", "rayqk"],
                    help="WHERE the camera enters.  xattn = tier C, a control sequence "
                         "the decoder cross-attends to; nothing in it says 'token i "
                         "looks along THIS ray', so the alignment must be learned inside "
                         "a frozen trunk, and measurably never is.  patch = tier A, rays "
                         "added onto the merged patch tokens: feature k lands on patch k "
                         "and the signal enters before the LLM.  This is the biggest "
                         "single lever measured -- bigger than the arm or the budget.")
    ap.add_argument("--gate-open", type=float, default=None,
                    help="pin tanh(gate) at this value for the WHOLE run and never train "
                         "it (1.0 = fully open).  Overrides --gate-init/--gate-freeze.  "
                         "Separates 'pose is useless' from 'the optimiser found the "
                         "switch cheaper than the signal' -- the learned gate closes "
                         "even when held open for the first --gate-freeze steps.")
    ap.add_argument("--workspace-centre", default="",
                    help="'x,y,z' the rig aims at.  Empty = read it from the dataset's "
                         "canonical camera target, which is what every camera looks at.")
    ap.add_argument("--encoder-init", default="zero", choices=["zero", "small"],
                    help="tier A zeroes encoder.out so the trunk is untouched at step 0. "
                         "'small' is required for --pose-source learned (the zero also "
                         "cuts gradient to the head) and is exposed separately so a "
                         "true-pose arm can be run with the SAME init -- otherwise any "
                         "learned-vs-true comparison is confounded by the init.")
    ap.add_argument("--pose-source", default="true", choices=["true", "learned", "random"],
                    help="'learned' replaces the stored extrinsics with a CNN's prediction "
                         "from the pixels, trained by the PROGRESS LOSS ONLY -- no pose "
                         "supervision anywhere.  Only interpretable alongside a probe of "
                         "the learned code against the true pose (see pose_head.py).")
    ap.add_argument("--pose-null", action="store_true",
                    help="feed the pose pathway ZEROS for the whole run, training AND "
                         "eval.  Same parameters, same injection, no camera.  The "
                         "control for --gate-open: if it scores the same, the gain was "
                         "trainable capacity and not geometry (it did, for tier C).")
    ap.add_argument("--occl-data", default="",
                    help="graded occlusion battery from scripts/render_occlusion_ladder.py, "
                         "scored after the test pass.  Held out, never trained on.")
    ap.add_argument("--occl-diag", action="store_true",
                    help="also score the ladder with the pose zeroed and with a wrong "
                         "camera.  Occlusion is where geometry has the best excuse to "
                         "matter, so the causal check belongs there too.")
    ap.add_argument("--occl-trajs", type=int, default=0, help="0 = all")
    ap.add_argument("--val-group", default="ood_pose",
                    help="camera group(s) model selection is scored on, comma separated "
                         "and pooled by labelled-frame count.  A name the battery does "
                         "not have is fatal, not silently ignored.  Azimuth ladder: "
                         "az_edge,az_far.")
    ap.add_argument("--diag-group", default="ood_pose",
                    help="group the causal diagnostics run on")
    ap.add_argument("--xattn-layers", default="27,31,35")
    ap.add_argument("--control-dim", type=int, default=256)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--gate-init", type=float, default=0.05)
    ap.add_argument("--gate-freeze", type=int, default=0,
                    help="hold the gate at --gate-init for this many steps so the pose "
                         "branch learns under a guaranteed non-zero injection")
    ap.add_argument("--pose-dropout", type=float, default=0.1)
    ap.add_argument("--only-kind", default="",
                    help="restrict TRAIN to one kind, e.g. 'success'.  Used by the "
                         "stage-1 overfit check, where a consistency-only failure "
                         "trajectory would give nothing to overfit to.")
    ap.add_argument("--fail-policy", default="mask",
                    choices=["mask", "zero_last", "robometer"],
                    help="how failure episodes are labelled.  'mask': no progress target "
                         "(they train/score consistency only).  'zero_last': all-zero "
                         "progress.  'robometer': the failure keeps a progress target and "
                         "the failure signal is carried by the SUCCESS head instead, "
                         "which is what Robometer itself does -- its progress ramp is "
                         "computed without ever looking at quality_label, and "
                         "compute_success_labels is what returns all-zeros for a "
                         "failure.  Requires --success-head.")
    ap.add_argument("--success-head", action="store_true",
                    help="train Robometer's own pretrained success head alongside "
                         "progress, with class-balanced BCE.  This is the head that "
                         "carries 'this trajectory failed' in the official recipe; "
                         "without it --fail-policy robometer throws the failure "
                         "information away entirely.")
    ap.add_argument("--lambda-success", type=float, default=1.0)
    ap.add_argument("--min-success", type=float, default=0.5,
                    help="Robometer's data.min_success: a frame enters the success loss "
                         "when its progress is below this OR it is labelled a success; "
                         "frames in between are ambiguous and are left out.")
    ap.add_argument("--max-success", type=float, default=1.0,
                    help="Robometer's dataset_success_percent: progress at or above this "
                         "makes a frame a positive.  1.0 is their value for simulation, "
                         "where trajectory ends are exact.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--swap", type=int, default=1)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--bin-weight-alpha", type=float, default=0.5,
                    help="progress-loss weight per bin ~ freq^-alpha.  0 disables. "
                         "Counteracts 63%% of frames sitting in bins 3-4.")
    ap.add_argument("--collapse-floor", type=float, default=0.40,
                    help="minimum pred_std/label_std for a checkpoint to be eligible "
                         "at model selection.  Guards against selecting the constant "
                         "predictor, which wins on MAE and S_view by not varying.")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--diag-trajs", type=int, default=15,
                    help="trajectories per causal-diagnostic pass.  8 modes x N trajs x "
                         "8 OOD cameras of 32-frame forwards adds up fast; 15 keeps the "
                         "pass under ~10 min while still giving 120 clips per mode.")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.workspace_centre:
        a.workspace_centre = [float(v) for v in a.workspace_centre.split(",")]
    else:
        _bat = json.load(open(f"{a.data}/index.json"))["battery"]
        a.workspace_centre = list(_bat[0]["target"])

    run = f"{a.out}/{a.arm}{a.tag}"
    os.makedirs(run, exist_ok=True)
    logf = open(f"{run}/train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda")

    store = ClipStore(a.data, "train", a.fail_policy)
    if a.only_kind:
        store.keys = [k for k in store.keys if store.kind[k] == a.only_kind]
    if a.train_trajs:
        store.keys = store.keys[:a.train_trajs]
    a.val_store = ClipStore(a.data, "val", a.fail_policy)
    test = ClipStore(a.data, "test", a.fail_policy)
    if a.test_trajs:
        test.keys = test.keys[:a.test_trajs]
    n_lab = sum(store.labeled(k) for k in store.keys)
    log(f"=== arm {a.arm} === train {len(store.keys)} trajs ({n_lab} with a progress "
        f"target, {len(store.keys)-n_lab} consistency-only), val {len(a.val_store.keys)}, "
        f"test {len(test.keys)}  fail-policy={a.fail_policy}")

    bundle = load_robometer(device=device)
    model, use_pose = build_model(a.arm, bundle, device, a)
    n_lora, n_head = (0, 0)
    if a.arm != "R0":
        n_lora, n_head = add_lora(model.rbm, a)
    model.enable_gradient_checkpointing(True)
    log(model.describe())
    log(f"trainable: LoRA {n_lora/1e6:.2f}M + head {n_head/1e6:.2f}M" +
        (f" + adapter {sum(p.numel() for p in model.adapter_parameters())/1e6:.2f}M"
         if use_pose else ""))

    # DO NOT REMOVE -- this forward is load-bearing, not just a check.
    #
    # Without it every step of training comes out NaN.  Measured: with LoRA injected and
    # gradient checkpointing enabled, if the FIRST grad-enabled forward happens inside
    # the training loop, the logits are non-finite from step 0 onward; running one
    # forward here first makes the identical loop train normally (loss 2.58 -> 1.48 over
    # 6 steps).  Something in the unsloth/transformers checkpointing path is initialised
    # lazily on first use and is wrong when that first use is the training step itself.
    # It reproduces on the CLI path and not from a harness that happened to forward once
    # beforehand, which is what made it look like an argument-parsing difference.
    #
    # It doubles as a fail-fast: a model emitting non-finite logits here would otherwise
    # burn the whole budget producing NaN, and NaN in the loss is indistinguishable in
    # the log from honest divergence.
    # The instability is INTERMITTENT: across otherwise identical runs the first forward
    # is sometimes non-finite and sometimes not (R1 hit it on one launch, P1 and P2 did
    # not on the next).  So retry rather than abort -- a later forward has always been
    # clean once an earlier one has run.  The in-loop non-finite skip is the second net.
    _k = next(k for k in store.keys if store.labeled(k))
    _f, _m = store.load(_k, store.cams_in(_k)[0], list(range(a.frames)))
    _p = None
    for attempt in range(4):
        _p = forward_probs(model, _f, _m, device, use_pose, False, grad=True)
        if torch.isfinite(_p).all():
            break
        log(f"self-test forward attempt {attempt}: NON-FINITE, retrying")
    log(f"self-test forward: finite={bool(torch.isfinite(_p).all())} "
        f"row0={_p[0].detach().float().cpu().numpy().round(3)}")
    if not torch.isfinite(_p).all():
        raise RuntimeError("model emits non-finite logits before training has started")
    # ``_p`` is the output of a GRAD-ENABLED forward, so holding the name alive holds
    # the whole autograd graph -- several GB that are never freed because ``main``
    # keeps the reference for the rest of the run.  On a 12 GB card that graph is the
    # difference between a 32-frame eval forward fitting and OOM-ing.
    del _p, _f, _m
    torch.cuda.empty_cache()

    results = {}
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location=device)
        load_state(model, sd, use_pose)
        log(f"loaded {a.ckpt}")
    elif not a.eval_only and a.arm != "R0":
        hist, best = train(model, store, device, a, use_pose, log)
        json.dump(hist, open(f"{run}/loss_history.json", "w"))
        if best is not None:
            torch.save(best[2], f"{run}/best.pt")
            log(f"best val {a.val_group} MAE {best[0]:.4f} @ step {best[1]} "
                f"-> {run}/best.pt")
            load_state(model, best[2], use_pose)
        else:
            torch.save(state_to_save(model, use_pose), f"{run}/final.pt")

    log("\n--- TEST ---")
    results["test"] = evaluate(model, test, device, a, use_pose)
    log(fmt(f"{a.arm} test", results["test"]))

    if use_pose:
        # Both lines or neither: a gate that stayed open with inj% ~ 0 is a branch that
        # zeroed its own projection, which is the same finding wearing a different hat.
        gv, ir = (model.injector.gate_values() if hasattr(model.injector, "gate_values") else {}), model.injector.injection_ratios()
        results["gates"] = dict(tanh_g=gv, injection_ratio=ir,
                                pinned=a.gate_open is not None, inject=a.inject,
                                pose_null=a.pose_null)
        if gv:
            log("\ngates      " + "  ".join(f"L{i}={v:+.3f}" for i, v in sorted(gv.items())))
        log("injection  " + "  ".join(f"{k}={v * 100:5.2f}%" for k, v in sorted(ir.items()))
            + "   (||injected|| / ||host||, last forward)")

    if a.occl_data:
        # The shipped `occlusion` group does not occlude -- pulling its azimuth out of
        # the wedge where the robot base fills the frame also pulled it out of the ARM's
        # path, and it scores BETTER than canonical.  The ladder does occlude, in matched
        # occluded/clear pairs.  Scored through the same evaluate() as everything else;
        # for the per-rung breakdown run scripts/score_occlusion.py against the same
        # checkpoint.
        occl = ClipStore(a.occl_data, "test", fail_policy=a.fail_policy)
        modes = ("correct", "zero_pose", "shuffled_pose") if (use_pose and a.occl_diag) \
            else ("correct",)
        for mode in modes:
            r = evaluate(model, occl, device, a, use_pose, pose_mode=mode,
                         max_traj=a.occl_trajs)
            results["occlusion_ladder" + ("" if mode == "correct" else f"/{mode}")] = r
            log(fmt(f"\n{a.arm} occlusion ladder"
                    + ("" if mode == "correct" else f"  [pose={mode}]"), r))

    results["frame_counter"] = frame_counter_baseline(test, a)
    log("\n" + fmt("frame-counter baseline (mean label per timestep, ignores pixels)",
                   results["frame_counter"]))

    if a.diagnostics:
        log("\n--- CAUSAL DIAGNOSTICS (same checkpoint, only the input changes) ---")
        # The pose perturbations are meaningless without a pose pathway, but black_rgb
        # and shuffled_frames are not: they ask whether the arm reads the image at all,
        # and that question needs an answer for the RGB baseline too -- otherwise
        # "black_rgb beats the real image" looks like a property of pose conditioning
        # when it may just be what the backbone does.
        modes = (("correct", "zero_pose", "shuffled_pose", "perturbed:5", "perturbed:15",
                  "perturbed:30", "black_rgb", "shuffled_frames") if use_pose
                 else ("correct", "black_rgb", "shuffled_frames"))
        for mode in modes:
            r = evaluate(model, test, device, a, use_pose, groups_only=(a.diag_group,),
                         pose_mode=mode, max_traj=a.diag_trajs)
            results[f"diag/{mode}"] = r
            g = r[a.diag_group]
            log(f"  {mode:20s} {a.diag_group} MAE {g['mae']:.4f}  "
                f"tau {g['tau']:+.3f}  S_view {g['s_view']:.4f}")

    json.dump(results, open(f"{run}/results.json", "w"), indent=2, default=float)
    log(f"\nwrote {run}/results.json")


def load_state(model, sd, use_pose: bool):
    """Load a checkpoint written by EITHER trainer.

    ``remote_train/train_r1_p1.py`` is a standalone single file with its own PoseBranch,
    and its module layout differs from this package's injector: tier A sits under
    ``patch.encoder.*`` there and at ``encoder.*`` here, because here PatchAddInjector is
    a BASE class of CrossAttnInjector rather than a member.  That file's docstring warns
    its ``inj.*`` keys are not interchangeable -- so convert them, rather than failing on
    every checkpoint this project has produced so far.

    The injector load is non-strict, because the package also carries tier-B parameters
    the standalone file never built.  Non-strict must not be allowed to mean "silently
    restored nothing", so it is checked both ways: unexpected keys are fatal (the layouts
    have diverged further than this knows about), and a zero-initialised tier-A encoder
    that is STILL all-zero afterwards means the weights never arrived.
    """
    ph = {k[len("pose_head."):]: v for k, v in sd.items() if k.startswith("pose_head.")}
    if ph and getattr(model, "pose_head", None) is not None:
        model.pose_head.load_state_dict(ph)
    rbm = {k[4:]: v for k, v in sd.items() if k.startswith("rbm.")}
    inj = {k[4:]: v for k, v in sd.items() if k.startswith("inj.")}
    if rbm:
        model.rbm.load_state_dict(rbm, strict=False)
    if not (inj and use_pose):
        return
    inj = {(k[len("patch."):] if k.startswith("patch.") else k): v for k, v in inj.items()}
    missing, unexpected = model.injector.load_state_dict(inj, strict=False)
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} checkpoint key(s) the injector has no home for, e.g. "
            f"{list(unexpected)[:3]} -- the two layouts have diverged further than "
            f"load_state knows about")
    enc = getattr(model.injector, "encoder", None)
    # --rayqk-only DELIBERATELY leaves tier A off and its encoder frozen at the zero
    # init, so "still all-zero" is the correct state there and this guard, which exists
    # to catch a silently-unrestored checkpoint, would fire on a healthy run.
    if (enc is not None and any(k.startswith("encoder.") for k in inj)
            and getattr(model, "patch_inject", True)):
        if float(enc.out.weight.abs().max()) == 0.0:
            raise RuntimeError(
                "tier-A encoder is still all-zero after loading, so nothing was "
                f"restored ({len(missing)} missing key(s))")


if __name__ == "__main__":
    sys.exit(main())
