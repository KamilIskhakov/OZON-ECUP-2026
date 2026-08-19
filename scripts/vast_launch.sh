#!/usr/bin/env bash
# Поиск, аренда и заливка на vast.ai. Ключ берётся из .env (в git не попадает).
#
#   bash scripts/vast_launch.sh search          # посмотреть предложения
#   bash scripts/vast_launch.sh create <id>     # арендовать
#   bash scripts/vast_launch.sh upload <id>     # залить код и данные
#   bash scripts/vast_launch.sh ssh <id>        # подключиться
#   bash scripts/vast_launch.sh kill <id>       # остановить (деньги идут за время!)
set -euo pipefail
cd "$(dirname "$0")/.."
export VAST_API_KEY="$(grep '^API-KEY=' .env | cut -d= -f2- | tr -d '[:space:]')"
VAST=$(command -v vastai || echo "$(dirname "$(command -v python)")/vastai")
IMAGE=${IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}
DISK=${DISK:-80}

# Фильтр под наш профиль нагрузки. Узкое место — не арифметика, а 192
# последовательных запуска ядер на шаг рекуррентности, поэтому топовая
# карта не нужна. Зато нужен CPU: токены собираются на месте из панели
# в 30.6 млн строк, и это работа polars, а не GPU.
FILTER=${FILTER:-'reliability>0.98 num_gpus=1 gpu_ram>=16 cpu_ram>=32 cpu_cores>=8 disk_space>=100 inet_down>=300 rentable=true dph_total<0.5'}

case "${1:-search}" in
  search)
    $VAST search offers "$FILTER" -o 'dph_total' --limit "${2:-10}"
    ;;
  create)
    ID=${2:?нужен id предложения}
    $VAST create instance "$ID" --image "$IMAGE" --disk "$DISK" \
        --ssh --direct --onstart-cmd 'touch /workspace/.ready'
    echo "инстанс создаётся; статус: $VAST show instances"
    ;;
  upload)
    ID=${2:?нужен id инстанса}
    read -r HOST PORT < <($VAST ssh-url "$ID" | sed -E 's#ssh://root@([^:]+):([0-9]+)#\1 \2#')
    echo "заливка на $HOST:$PORT"
    # Токены НЕ везём: 12 ГБ против 172 МБ панели, из которой они собираются.
    rsync -az --info=progress2 -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
      --exclude '.git' --exclude 'artifacts/neural/tokens' \
      --exclude 'artifacts/neural/pretrain' --exclude '.env' \
      --exclude '__pycache__' --exclude '*.ipynb' \
      ./ "root@$HOST:/workspace/ecup/"
    echo "готово; дальше: bash scripts/vast_launch.sh ssh $ID"
    ;;
  ssh)
    ID=${2:?нужен id инстанса}
    read -r HOST PORT < <($VAST ssh-url "$ID" | sed -E 's#ssh://root@([^:]+):([0-9]+)#\1 \2#')
    exec ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
      -t 'cd /workspace/ecup && bash -l'
    ;;
  kill)
    $VAST destroy instance "${2:?нужен id инстанса}"
    ;;
  *) echo "неизвестная команда: $1"; exit 1 ;;
esac
