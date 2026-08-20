#!/usr/bin/env bash
# Плотные срезы в этапе B: два фолда, всё остальное как в базовом evt-прогоне.
set -uo pipefail
cd /workspace/ecup
O=artifacts/neural
python -u scripts/train_gapgru.py --max-len 192 --epochs 12 --batch-size 2048 \
  --init-from "$O/pretrain_evt" --freeze-epochs 2 --eval-every 2 \
  --dense-dir "$O/dense" --dense-frac 0.5 \
  --ckpt "$O/dense_ckpt" --tb "$O/tb" --out "$O/gapgru_dense.json" \
  > "$O/logs/dense_train.log" 2>&1
tail -12 "$O/logs/dense_train.log"
echo "ГОТОВО"
