#!/usr/bin/env bash
# Факторизация остатка: что именно исправляет последовательностный сигнал —
# вероятность покупки или условную сумму. Раздельно, иначе при положительном
# результате механизм снова будет неразличим.
set -uo pipefail
cd /workspace/ecup
O=artifacts/neural
for R in p m pm; do
  echo "=== residual=$R ==="
  python -u scripts/train_gapgru.py --max-len 192 --epochs 12 --batch-size 2048 \
    --init-from "$O/pretrain_evt" --freeze-epochs 2 --eval-every 2 \
    --residual "$R" --ckpt "$O/res_${R}_ckpt" --tb "$O/tb" \
    --out "$O/gapgru_res_${R}.json" 2>&1 | tee "$O/logs/res_${R}.log" | tail -12
done
echo "ВСЁ ГОТОВО"
