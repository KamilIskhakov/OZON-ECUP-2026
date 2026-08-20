#!/usr/bin/env bash
# Просмотр обучения на кластере через TensorBoard.
#
#   bash scripts/tensorboard.sh <host> <port>
#
# Поднимает tensorboard на инстансе и туннель к нему. Открывать
# http://localhost:6006 — вкладка SCALARS, там же кривая holdout/gain.
#
# Главная кривая — не loss, а holdout/gain: переносимый выигрыш на якоре
# подбора α. Именно по ней выбирается эпоха, потому что лосс включает
# вспомогательные головы и падает даже когда поправка перестала улучшаться.
set -euo pipefail
HOST=${1:-202.122.49.242}
PORT=${2:-64938}
KEY=${KEY:-$HOME/.ssh/vast_ecup}
SSH_OPTS="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"

ssh -p "$PORT" $SSH_OPTS "root@$HOST" \
  'pip install -q tensorboard 2>/dev/null; pkill -f "tensorboard --logdir" 2>/dev/null || true;
   cd /workspace/ecup && nohup tensorboard --logdir artifacts/neural/tb --port 6006 \
     --bind_all > /tmp/tb.log 2>&1 & sleep 4; echo "tensorboard поднят"'

echo "туннель: http://localhost:6006  (Ctrl-C чтобы закрыть)"
exec ssh -N -L 6006:localhost:6006 -p "$PORT" $SSH_OPTS "root@$HOST"
