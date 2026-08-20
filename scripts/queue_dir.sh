#!/usr/bin/env bash
# Очередь: направленный лосс, затем недостающие боевые сиды.
set -uo pipefail
cd /workspace/ecup
O=artifacts/neural

echo "[1/2] направленный лосс, два фолда"
python -u scripts/train_gapgru.py --max-len 192 --epochs 12 --batch-size 2048 \
  --init-from "$O/pretrain_evt" --freeze-epochs 2 --loss dir --accum 4 \
  --eval-every 2 --ckpt "$O/dir_ckpt" --tb "$O/tb" --out "$O/gapgru_dir.json" \
  > "$O/logs/dir.log" 2>&1
tail -10 "$O/logs/dir.log"

echo "[2/2] недостающие боевые сиды"
for S in 7 2026; do
  python -u scripts/train_gapgru.py --max-len 192 --epochs 12 --batch-size 2048 \
    --init-from "$O/pretrain_evt" --freeze-epochs 2 --seed "$S" --production \
    --ckpt "$O/prod_s${S}" --out "$O/prod_s${S}.json" > "$O/logs/prod_s${S}.log" 2>&1
  echo "сид $S готов"
done
echo "ВСЁ ГОТОВО"
