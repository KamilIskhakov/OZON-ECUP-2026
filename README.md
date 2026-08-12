# OZON E-CUP 2026 · Search LTV

Прогноз суммарного GMV пользователя за 30 дней (**2026-02-14 … 2026-03-15**) по истории
дневной активности. Метрика — RMSLE, 250 000 пользователей.

| Документ | О чём |
| :--- | :--- |
| [SOLUTION.md](SOLUTION.md) | текущее решение, результат на лидерборде, проверенные гипотезы |
| [ANALYSIS.md](ANALYSIS.md) | выжимка анализа данных: три находки, метрика, признаки, причинность |
| [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) | полный разведочный анализ, 40 графиков |
| [notebooks/02_report.ipynb](notebooks/02_report.ipynb) | отчёт с выводами формул |

## Установка

```bash
conda env create -f environment.yml
conda activate ecup
```

Либо в существующее окружение:

```bash
pip install -r requirements.txt
# macOS: LightGBM без libomp не импортируется
conda install -c conda-forge llvm-openmp   # или: brew install libomp
```

## Данные

Положить в `data_start/`:

```text
data_start/
├── train.parquet        30 631 006 строк, 180 MB
└── sample_submit.csv    250 000 строк
```

`data_work/` и `artifacts/` создаются автоматически: первый — кеш панели
(даункаст 4090 → 1139 MB, считается один раз), второй — сабмиты и результаты свипов.

## Запуск

```python
import sys; sys.path.insert(0, "src")
from ecup import load_panel, run_validation, make_submission, SplitConfig

df = load_panel()                                    # ~30 с в первый раз, потом мгновенно
split = SplitConfig(max_history=300, n_train_anchors=6)

run_validation(df=df, split=split)                   # ~4 мин
make_submission(split=split, a_p=0.0, a_m=0.07)       # ~5 мин → artifacts/submission_v2.csv
```

Эксперименты:

```python
from ecup import sweep_history, sweep_anchors
sweep_history(history_grid=(90, 168, 240, 300, 365), n_train_anchors=4, df=df)
sweep_anchors(anchor_grid=(1, 2, 4, 6, 9), max_history=300, df=df)
```

## Требования к машине

Разрабатывалось на MacBook, 18 GB RAM, 11 ядер, без GPU. Панель после даункаста
занимает 1.1 GB, обучение на 7 якорях (1.2 млн примеров × 116 признаков) — около 5 минут.
