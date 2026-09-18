#!/usr/bin/env bash
# Full pose-injection ablation.  Needs ~24GB; on a 12GB card only the
# --no-patch-inject / 4-frame corner of this fits (see the 4070S run in
# outputs/avm32 for what that degenerates into).
#
# Four arms, each trained from the same frozen Robometer-4B:
#   A_patch   tier A, Plucker added to patch tokens only   (the VD3D-refuted one)
#   C_xattn   tier C, zero-gated cross-attention only
#   C_full    tier C + patch injection                      (both pathways)
#   blind     no injection at all                           (control)
#
# 'blind' is not a trained arm -- it is the frozen backbone, and it is what every
# other arm is measured against.  It appears in each report automatically because
# evaluate() runs the same object with clear_pose().
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/miniconda3/envs/robometer/bin/python}
OUT=${OUT:-$HOME/avm_work/results}
# CUDA enumerates by compute capability by default, which is the REVERSE of
# nvidia-smi's PCI order on rog: the 4090 is nvidia-smi index 1 but cuda:0.
# Pinning CUDA_DEVICE_ORDER makes the index mean what nvidia-smi says.
GPU=${GPU:-1}                       # nvidia-smi index of the 4090 on rog
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STEPS=${STEPS:-3000}
FRAMES=${FRAMES:-16}
EVAL_FRAMES=${EVAL_FRAMES:-32}
EVAL_TRAJS=${EVAL_TRAJS:-10}
TRAIN_TRAJS=${TRAIN_TRAJS:-40}
LAYERS=${LAYERS:-11,17,23,29,35}

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU

run () {
  local name=$1; shift
  echo "=============================================================="
  echo "ARM: $name   $*"
  echo "=============================================================="
  $PY scripts/finetune_robometer.py \
      --steps "$STEPS" --frames "$FRAMES" --eval-frames "$EVAL_FRAMES" \
      --eval-trajs "$EVAL_TRAJS" --train-trajs "$TRAIN_TRAJS" \
      --xattn-layers "$LAYERS" --log-every 100 \
      --out "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name.log"
}

run C_full  --injector cross_attn --patch-inject
run C_xattn --injector cross_attn
run A_patch --injector patch_add  --patch-inject

echo
echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
$PY scripts/summarize_ablation.py --root "$OUT"
