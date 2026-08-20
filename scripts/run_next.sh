#!/usr/bin/env bash
# Три задачи по порядку: чистое сравнение архитектур, затем боевые модели.
#
# 1. ARCH=cyc при ТЕХ ЖЕ настройках, что и evt (6 эпох A, 12 B). Тогда
#    разница относится только к токенам покупочных циклов, а не к трём
#    изменениям сразу, как в прошлом прогоне.
# 2. Три боевые модели evt на ВСЕХ якорях (198…378) с разными сидами.
#    Усреднение Δz по сидам — та же логика, что дала выигрыш в ансамбле
#    деревьев: снижает шумовую компоненту направления, не трогая сигнальную.
set -euo pipefail
cd /workspace/ecup
OUT=artifacts/neural
mkdir -p "$OUT/logs"

echo "### 1. чистое сравнение: cyc при настройках evt"
ARCH=cyc EPOCHS_A=6 EPOCHS_B=12 EVAL_EVERY=2 bash scripts/run_cluster.sh \
  2>&1 | tee "$OUT/logs/cyc_clean.log" | tail -6

echo "### 2. боевые модели evt на всех якорях"
for SEED in 42 7 2026; do
  echo "--- сид $SEED ---"
  python scripts/train_gapgru.py --max-len 192 --epochs 12 --batch-size 2048 \
      --init-from "$OUT/pretrain_evt" --freeze-epochs 2 --seed "$SEED" \
      --production --ckpt "$OUT/prod_s${SEED}" --tb "$OUT/tb" \
      --out "$OUT/prod_s${SEED}.json" 2>&1 | tail -4
done
echo "### готово"
