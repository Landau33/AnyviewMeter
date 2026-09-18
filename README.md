# AnyviewMeter

A progress-reward model conditioned on **camera pose**, built to test one specific
question: *how* does Plücker geometry have to enter a pretrained VLM before the
model actually uses it?

Structure mirrors `robometer/` deliberately, so a Robometer checkpoint and an
AnyviewMeter checkpoint stay comparable term for term.

---

## Requirements

This repository holds the AnyviewMeter code only. It is built on Robometer and does not
run on its own:

- The `robometer` package must be importable (`robometer.utils.save`,
  `robometer.utils.setup_utils`, `robometer.data.dataset_types`). Install the Robometer
  source with `pip install -e <path-to-robometer>` or put it on `PYTHONPATH`.
- The backbone checkpoint is `robometer/Robometer-4B` on Hugging Face.
- Some scripts still default to data and output paths of the original workspace
  (`/home/yuang/ws_jepa/multicam_ws/outputs/...`); pass `--data` / `--out` explicitly.

Commands below are run from the repository root.

---

## Why three injection tiers instead of one

Camera pose is a **low-frequency, low-dimensional** signal competing with a very
high-capacity pretrained visual stream, and a large model is free to ignore it.

VD3D's ablation is the warning we are designing around: zero-initialised Plücker
features added straight onto patch tokens produced *almost no* camera
controllability, and only a ControlNet-style cross-attention pathway made the
conditioning bite. So "inject Plücker somewhere in the middle" is a hypothesis,
not a plan — and **which injection is strong enough for a progress model is itself
the non-trivial contribution.**

So the tiers live behind one interface and the ablation is a config flag:

| tier | config | what it adds | why it's here |
|---|---|---|---|
| **A** `patch_add` | `tier_a_patch_add.yaml` | zero-init Plücker → patch tokens | the VD3D-refuted baseline, kept as the **negative control** |
| **B** `cam_token` | `tier_b_cam_token.yaml` | A + a global `<cam>` register token | rays are *local*, "which viewpoint" is *global* — UniScale splits it the same way |
| **C** `cross_attn` | `tier_c_cross_attn.yaml` | B + zero-gated cross-attention into LLM layers | the recipe VD3D found actually works |
| **D** `qk_pe` | — | B + a ray positional encoding on each injected layer's **Q and K** | the one form that is not a value-stream write at all |
| — | `pose_blind_control.yaml` | nothing | identical code path, injector removed |

Tier D arrived after the tier-C result, from SCoPE (arXiv 2606.27345). A/B/C all write
pose into the *value stream* — they change what a token's vector says. D writes into
the *address space*: the ray is added to the pretrained attention's queries and keys,
so geometry becomes part of how tokens find each other. SCoPE's ablation reports that
deleting the content↔geometry cross-terms costs more than any other component, with the
claim that this coupling cannot be represented by a V-side or cross-attention pathway.
Tier C *is* a V-side cross-attention, and tier C is the tier we measured failing here —
so this is the one structural form we had not tried, and it comes with an argument for
why the ones we did try were the wrong shape.

If B and C do not beat A, the "simple injection is not enough" claim has **no
evidence** and the method section must say so. That is the point of keeping A.

Every tier is the **exact identity** on the backbone at initialisation (zero-init
projections and gates). That is a deliberate constraint: it makes the tiers
comparable, since none of them perturbs pretrained features until learning moves
them, and it makes any gain attributable.

Tier D is the first one where that constraint costs nothing. The bind everywhere else
is that zero-initialising the scalar which *scales* a branch also starves everything
behind it — tier C's gates random-walked at ~0.012 while 9.5M parameters trained at 1%
of their nominal rate, which is why `gate_init` had to be raised to 0.05 and exact
identity given up. In tier D the two roles come apart: `alpha` carries the zero (so
the layer is bit-exactly the pretrained attention at step 0, and `d(out)/d(alpha) = pe
≠ 0` lifts it off on the first step), while `E_q = E_k = [I_6 | 0]` starts at the
**Plücker reciprocal product** — not a small random point but the exact geometric
quantity. Silent and correct, rather than silent and dead.

---

## The one number to know before reading any tier-D result

Two rays through the same camera centre always intersect, so their reciprocal product
is **identically zero**. Our clips are single-camera sequences, so tier D's pure-geometry
term contributes a constant 0 across the whole sequence and only the content↔geometry
terms can do any work. `tests/test_ngi.py` and `remote_train/smoke_qk_pe.py` both pin
this as an assertion rather than leaving it in prose.

Making that term live needs several cameras packed into one attention sequence. The data
already supports it — 25 cameras per trajectory — so it is a collator change, not a
re-render, and it is the obvious next experiment on this line.

---

## Two places where we deliberately depart from SCoPE

**The moment fed to `E` is raw, not normalised.** SCoPE's Normalize-Gate-Inject sends
`(d, m̂, s)` through `E`. That normalisation destroys the coplanarity zero the whole
method is named after: `d_i·m̂_j + d_j·m̂_i` vanishes only when the two magnitudes happen
to match. Measured on one of our own cameras, the same-camera term is **3e-8 raw and
3.9e-1 normalised**. It also costs SE(3) invariance, since `‖m‖` depends on where the
world origin sits. So `moment_mode="raw"` is the default — `E` reads the true moment and
`s` still goes to the gate, which is where the scale information is wanted. `"unit"` is
kept as a flag for anyone training across datasets.

That default is only defensible because we measured the thing normalisation exists to
fix. `scripts/moment_scale_report.py` reports `s = log‖m‖` per camera group against the
training range, and on both `pickcube_avm` and `pickcube_avm_wide` every group sits
**inside** it, with the camera moving `s` by only ~25% of what perspective does within a
single frame. Moment magnitude is not a distribution-shift problem here, so do not
attribute out-of-cone conditioning failures to it.

**The gate is a function of the input, not a free scalar.** Our tier-C finding was that
a learned scalar gate closes, and that `--gate-open` answers a different question than
the one we asked — it removes the switch instead of making the switch informative. The
`ScaleGate` used by tiers C (`gate_mode="scale"`) and D is per-channel and driven by the
ray's log-magnitude: shutting it off means learning an MLP that outputs large negatives
for every camera in the data, and a gate that closes for only *some* `s` is a reading
rather than one bit. Its `gate_span` is reported beside its mean, because a mean alone
cannot tell "sat at its initialisation" from "learned to sit half open".

---

## The geometry

For the ray through pixel `(u,v)`:

```
d = normalize(Rᵀ K⁻¹ [u,v,1]ᵀ)     direction, world frame
c = -Rᵀ t                          camera centre, world frame
m = c × d                          moment
r = (d, m)                         6 channels, image-aligned
```

Convention is OpenCV, matching ManiSkill's `intrinsic_cv` / `extrinsic_cv`:
`extrinsic` is world→camera `[R|t]`, `+z` forward. `tests/test_plucker.py` pins
this against real recorded data — `c = -Rᵀt` reproduces the renderer's independently
stored eye position to 2e-3.

Two details that are easy to get wrong and produce plausible-but-incorrect maps:

- **Maps are sampled at *token* centres**, not built at full resolution and pooled.
  Pooling averages ray directions across a token's footprint, which is wrong near
  the image edge where directions fan out.
- **Grid must match the vision tower.** Qwen3-VL is patch-16 with a 2×2 merge, so one
  token covers 32×32 px and a 256 px frame is an **8×8** grid. A mismatch raises
  loudly rather than broadcasting silently.

---

## The standing gate (inherited, non-negotiable)

`evals/diagnostics.py` carries phase A's ternary diagnostic forward:

| metric | meaning | direction |
|---|---|---|
| **S_view** | Δτ(best camera − worst camera) | should go **down** |
| **H_time** | \|τ\| small on frozen clips; τ negative on reversed/shuffled | hold or improve |
| **P_obj** | τ(success) − τ(object-frozen failure) | hold or improve |

**The hard rule:** a model that ignores pixels and counts frames is *trivially*
viewpoint-invariant. Phase A measured that **Robometer — the checkpoint we finetune
from — already carries a position prior** (|τ| up to +0.65 on 32 byte-identical
frames, and *positive* τ on temporally shuffled clips). So an S_view improvement is
meaningless alone: it must come with H_time and P_obj not degrading, or the model
has simply decayed further into that prior. `gate_verdict()` refuses such a
checkpoint, and `tests/test_pipeline.py` shows it doing so.

This is also why `pose_consistency_weight` defaults to **0**. That loss is the
direct signal for viewpoint robustness and the cheapest way to satisfy it is to
stop looking at the image — read `trainers/avm_trainer.py` before raising it.

---

## Layout

```
anyviewmeter/
  geometry/     camera.py      CameraParams, conventions, projection
                plucker.py     ray maps, pose descriptor, self-checks
  models/       avm.py         AnyviewMeter: backbone + heads + pose injection
                heads.py       progress / success / preference (+ pose probe)
                qk_attach.py   tier D's patched attention forward
                injection/     base.py  interface, registry, PluckerEncoder
                               tiers.py A / B / C
                               ngi.py   D: ScaleGate, RayPE, the reciprocal product
  data/         dataset_types.py   Trajectory + CameraView + MultiViewSample
                datasets/multicam.py   reads the phase-A rendered clips
                collators/avm_collator.py
  trainers/     avm_trainer.py losses, optimiser groups
  evals/        diagnostics.py S_view / H_time / P_obj + the gate
  configs/      experiment_configs.py + one YAML per tier
scripts/        smoke_test.py  moment_scale_report.py
tests/          test_plucker.py test_injection.py test_ngi.py
                test_model.py test_pipeline.py
```

---

## Running it

Nothing here needs a GPU or a weights download — the tests and the smoke run use a
stub backbone with the same surface AnyviewMeter relies on, so the wiring under
test is the real wiring.

```bash
python tests/test_plucker.py     # geometry correctness  (11)
python tests/test_injection.py   # tiers A/B/C           (12)
python tests/test_ngi.py         # tier D + the NGI feature (19)
python tests/test_model.py       # model wiring          (13)
python tests/test_pipeline.py    # collator/loss/gate    (17)
python scripts/smoke_test.py     # end-to-end, real clips
```

Or all at once:

```bash
python tests/run_all.py
```

Data comes from the phase-A renders, which already carry `intrinsic_cv` /
`extrinsic_cv` per clip — the Plücker inputs are present in what we have, no
re-render needed:

```
/home/yuang/ws_jepa/multicam_ws/outputs/clips/<task>/<success|failure>/<traj>__<cam>.npz
```

25 cameras per trajectory (5 semantic + 20 sampled sweep poses). The sweep poses are
the point: they make pose a **continuous** input rather than a 5-way categorical.

---

## Status

Built and tested: geometry, the three tiers, model wiring, collator, losses, the
gate. **Not yet done** — real Qwen3-VL loading (the model takes the backbone as an
argument, so this is a loader, not a redesign), the distributed training loop, and
any actual training run. `holdout_cameras` is wired but no transfer number exists
yet.
