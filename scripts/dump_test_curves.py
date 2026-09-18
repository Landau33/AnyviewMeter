"""Run train_viewpoint's own eval-only test pass and keep the per-clip progress curves.

train_viewpoint.evaluate() reduces every clip to group metrics and throws the curves
away.  This wrapper leaves that code untouched: it records which (trajectory, camera)
each forward belongs to by watching ClipStore.load, and saves yhat and the label next
to the results.json the normal eval writes.  Because the full test pass still runs,
the printed test metrics must reproduce the checkpoint's recorded results.json -- that
is the check that the reconstructed flags built the same model.

usage: python dump_test_curves.py OUT.npz  <train_viewpoint args ... --eval-only>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
out_path, sys.argv = sys.argv[1], [sys.argv[0]] + sys.argv[2:]
_argv = sys.argv[1:]
VIEWS = int(_argv[_argv.index("--views") + 1]) if "--views" in _argv else 1
PARTNER = _argv[_argv.index("--view-partner") + 1] if "--view-partner" in _argv else "canonical"

import train_viewpoint as tv  # noqa: E402

curves, pending, capture = {}, [], [False]
_load, _fp, _fpm, _eval = tv.ClipStore.load, tv.forward_probs, tv.forward_probs_multi, tv.evaluate


def load(self, key, cam, *args, **kw):
    frames, meta = _load(self, key, cam, *args, **kw)
    if capture[0]:
        # Two-view evaluate() loads the partner camera as a camera of its own and then
        # skips it with no forward.  Left in place, that unconsumed load became the
        # identity of the NEXT camera: on pickcube_avm_wide600_rnd every oodfov00
        # prediction was saved under "canonical" (found 2026-09-11 from 120 phantom
        # canonical curves and 0 oodfov00 ones).  Drop it when the next camera arrives.
        if (VIEWS > 1 and len(pending) == 1 and pending[0][1] == PARTNER
                and pending[0][0] == key and cam != PARTNER):
            pending.clear()
        pending.append((key, cam, meta.get("progress")))
    return frames, meta


def record(p):
    if capture[0] and pending:
        key, cam, y = pending[0]      # the first unconsumed load is the varying camera
        yhat = (p * tv.bin_centres(p.device)).sum(-1).float().cpu().numpy()
        curves[f"pred|{key}|{cam}"] = yhat
        if y is not None:
            curves[f"label|{key}|{cam}"] = np.asarray(y, dtype=np.float32)
    pending.clear()
    return p


def evaluate(*a, **kw):
    capture[0] = True
    try:
        return _eval(*a, **kw)
    finally:
        capture[0] = False


tv.ClipStore.load = load
# Eval only, so every forward runs without a graph: main()'s grad-enabled self-test
# forward is what OOMs a 12 GB card, and no checkpoint is trained here.
tv.forward_probs = lambda *a, **kw: record(_fp(*a, **{**kw, "grad": False}))
tv.forward_probs_multi = lambda *a, **kw: record(_fpm(*a, **{**kw, "grad": False}))
tv.evaluate = evaluate

if __name__ == "__main__":
    tv.main()
    np.savez_compressed(out_path, **curves)
    print(f"saved {len(curves)} arrays -> {out_path}")
