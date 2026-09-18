"""Collate the ablation arms into one table and apply the E4 gate to each.

Every arm shares the same frozen baseline (the same object with pose switched off),
so the baseline column is a consistency check as much as a reference: if two arms
disagree on it, something about the eval is non-deterministic and the comparison is
void.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anyviewmeter.evals.diagnostics import gate_verdict  # noqa: E402

ARMS = {"A_patch": "A  patch_add (+patch)",
        "C_xattn": "C  cross_attn",
        "C_full": "C  cross_attn +patch"}


def fmt(c, width=22):
    if not c or c.get("mean") is None:
        return f"{'--':>{width}}"
    return f"{c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]".rjust(width)


def scalar(c):
    return None if not c or c.get("mean") is None else c["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(f"{a.root}/*/comparison.json")):
        runs[os.path.basename(os.path.dirname(p))] = json.load(open(p))
    if not runs:
        print(f"no comparison.json under {a.root}")
        return 1

    for split in ("train_cams", "holdout_cams"):
        print(f"\n{'=' * 104}\n{split}\n{'=' * 104}")
        hdr = (f"{'arm':24}{'S_view':>22}{'P_obj':>22}{'stuck':>9}{'revd':>9}"
               f"{'shuf':>9}  verdict")
        print(hdr + "\n" + "-" * len(hdr))

        base = None
        for arm, r in runs.items():
            b = r.get(f"baseline/{split}")
            if b and base is None:
                base = b
                ht = b.get("H_time") or {}
                print(f"{'baseline (frozen)':24}{fmt(b.get('S_view'))}{fmt(b.get('P_obj'))}"
                      f"{scalar(ht.get('stuck')) or 0:+9.3f}{scalar(ht.get('reversed')) or 0:+9.3f}"
                      f"{scalar(ht.get('shuffled')) or 0:+9.3f}  --")
            elif b and base:
                d = abs((scalar(b.get("S_view")) or 0) - (scalar(base.get("S_view")) or 0))
                if d > 1e-6:
                    print(f"  !! {arm} measured a different baseline S_view "
                          f"(delta {d:.4f}) -- evaluation is not deterministic")

        for arm in sorted(runs, key=lambda k: list(ARMS).index(k) if k in ARMS else 99):
            r = runs[arm]
            v = r.get(f"avm/{split}")
            if not v:
                continue
            ht = v.get("H_time") or {}
            verdict = gate_verdict(v, r.get(f"baseline/{split}"))["verdict"]
            print(f"{ARMS.get(arm, arm):24}{fmt(v.get('S_view'))}{fmt(v.get('P_obj'))}"
                  f"{scalar(ht.get('stuck')) or 0:+9.3f}{scalar(ht.get('reversed')) or 0:+9.3f}"
                  f"{scalar(ht.get('shuffled')) or 0:+9.3f}  {verdict}")

    print(f"\n{'=' * 104}\ncross-attention gates (near zero = the backbone declined the pose signal)")
    for arm, r in runs.items():
        g = r.get("gates") or {}
        if g:
            print(f"  {ARMS.get(arm, arm):24}" +
                  " ".join(f"L{k}={v:+.3f}" for k, v in sorted(g.items(), key=lambda kv: int(kv[0]))))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
