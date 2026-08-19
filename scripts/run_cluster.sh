#!/usr/bin/env bash
# Полный конвейер на GPU: токены → этап A (претрейн) → этап B (остаток) → отчёт.
#
#   bash scripts/run_cluster.sh          # полный масштаб
#   MAXLEN=96 USERS=80000 bash scripts/run_cluster.sh   # уменьшенный прогон
#
# Каждый шаг идемпотентен: токены и срезы кэшируются, чекпоинты пишутся
# каждую эпоху. После прерывания spot-инстанса скрипт запускается повторно
# и продолжает с того места, где кэш уже есть.
set -euo pipefail
MAXLEN=${MAXLEN:-192}
USERS=${USERS:-0}
EPOCHS_A=${EPOCHS_A:-6}
EPOCHS_B=${EPOCHS_B:-12}
BS=${BS:-2048}
# ARCH=evt — Gap-GRU + многозапросная голова (контроль)
# ARCH=cyc — то же плюс вторая шкала времени: покупочные циклы
ARCH=${ARCH:-evt}
CYC=$([ "$ARCH" = cyc ] && echo "--cycles" || echo "")
OUT=artifacts/neural
mkdir -p "$OUT/logs"

echo "### архитектура: $ARCH"
echo "### 1/3 токены (max_len=$MAXLEN)"
python scripts/build_tokens.py --max-len "$MAXLEN" \
    ${USERS:+--users "$USERS"} --out "$OUT/tokens" --cycles \
    2>&1 | tee "$OUT/logs/tokens.log"

# Предел пуржинга свой у каждого фолда: срез допустим, только если его
# целевое окно заканчивается не позже якоря подбора α этого фолда.
for FOLD in 0 1; do
    LIMIT=$([ "$FOLD" = 0 ] && echo 318 || echo 348)
    echo "### 2/3 этап A, фолд $FOLD (срезы до $((LIMIT-30)))"
    python scripts/pretrain_gapgru.py --limit-anchor "$LIMIT" \
        --max-len "$MAXLEN" --epochs "$EPOCHS_A" --batch-size "$BS" \
        $CYC --out "$OUT/pretrain_${ARCH}_fold${FOLD}.pt" \
        2>&1 | tee "$OUT/logs/pretrain_${ARCH}_f${FOLD}.log"
done

echo "### 3/3 этап B: остаток к ансамблю, два фолда"
python scripts/train_gapgru.py --max-len "$MAXLEN" --epochs "$EPOCHS_B" \
    --batch-size "$BS" --init-from "$OUT/pretrain_${ARCH}" --freeze-epochs 2 \
    $CYC --ckpt "$OUT/gapgru_${ARCH}_ckpt" --out "$OUT/gapgru_${ARCH}.json" \
    2>&1 | tee "$OUT/logs/train_${ARCH}.log"

echo "### результат"
python -c "
import json; r = json.load(open('$OUT/gapgru_${ARCH}.json'))
for x in r: print(f\"  фолд {x['fold']}: {x['shape_base']:.5f} -> {x['shape_corrected']:.5f}  выигрыш {x['gain']:+.5f}  alpha={x['alpha']:+.4f}\")
print('  критерий:', 'ПРОЙДЕН' if all(x['gain'] > 0.0002 for x in r) else 'не пройден')"
