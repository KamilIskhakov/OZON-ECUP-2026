#!/usr/bin/env bash
# Развёртывание на инстансе vast.ai. Запускать из корня репозитория.
#
#   bash scripts/vast_bootstrap.sh
#
# Ожидается образ с CUDA (pytorch/pytorch:*-cuda*-runtime). Всё, что нужно
# доставить с локальной машины, — репозиторий, data_start/train.parquet
# (172 МБ) и artifacts/neural/oof_a*.npz (~20 МБ). Токены собираются здесь:
# заливать их бессмысленно, полный набор весит 13.7 ГБ.
set -euo pipefail

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== зависимости ==="
pip install -q --upgrade pip
pip install -q polars==1.43.2 lightgbm==4.7.0 catboost==1.2.10 \
               numpy scikit-learn scipy pyarrow
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA недоступна — инстанс без GPU"
print(f"torch {torch.__version__} · CUDA {torch.version.cuda} · "
      f"{torch.cuda.get_device_name(0)} · "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} ГБ")
PY

echo "=== данные ==="
test -f data_start/train.parquet || { echo "нет data_start/train.parquet"; exit 1; }
ls artifacts/neural/oof_a*.npz >/dev/null 2>&1 || { echo "нет OOF-файлов"; exit 1; }
python - <<'PY'
import sys; sys.path.insert(0, "src")
import warnings; warnings.filterwarnings("ignore")
from ecup import load_panel, SplitConfig
df = load_panel()
sp = SplitConfig(max_history=300, n_train_anchors=6, with_state=True)
print(f"панель {df.height:,} строк · якоря {sp.train_anchors()} · "
      f"валидация {sp.val_anchor}")
PY
df -h . | tail -1 | awk '{print "диск: свободно " $4 " (нужно ~25 ГБ на токены)"}'
echo "готово"
