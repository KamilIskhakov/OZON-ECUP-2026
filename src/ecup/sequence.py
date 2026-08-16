"""Sequence-энкодер дневной истории: TCN и GRU с multi-task головами.

Зачем вообще. Все проверенные представления траекторий (LSA по биграммам,
недельный SVD, исходы соседей) дали около нуля поверх агрегатов — но все они
**неконтролируемые**: сжимают дисперсию, а не сигнал под задачу. Первая
компонента недельного SVD забирает «размер пользователя», потому что это
главная дисперсия, а не потому что это главный предиктор.

Прямой тест остатков: сырой дневной ряд (120 дней × 5 каналов), поданный
в GBDT, объясняет остатки текущей модели с R² = +0.00081 против −0.00124
на перемешанном контроле. Сигнал есть. И это нижняя граница: GBDT видит
«день −37» независимым признаком, у него нет ни трансляционной инвариантности,
ни многомасштабного пулинга — ровно того, ради чего берут свёртки и attention.

Многозадачность нужна не ради дополнительных выходов, а чтобы представление
описывало механизм поведения, а не подгоняло одно шумное число: главный лосс
на log1p(y) слишком разрежен, чтобы одному учить энкодер.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# Каналы дневного вектора. Флаги каналов и индикатор активности дают сети
# отличить «день без строки» от «день со строкой, но без активности».
SEQ_CHANNELS: tuple[str, ...] = ("gmv", "to_ord", "to_cart", "searches")
N_CHANNELS = len(SEQ_CHANNELS) + 3          # + активность, search, cat


def build_sequences(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    span: int = 180,
    dtype: str = "float16",
) -> np.ndarray:
    """(N, span, C) — дневная история, выровненная по правому краю на якорь.

    Последний столбец времени всегда сам якорь, поэтому сети не нужно
    догадываться, где «сейчас»: recency закодирован позицией.
    """
    n = len(users)
    h = df.filter(pl.col("user_id").is_in(users)
                  & pl.col("d").is_between(anchor - span + 1, anchor))
    pos = {u: i for i, u in enumerate(users.to_numpy())}
    ri = np.fromiter((pos[u] for u in h["user_id"].to_numpy()), dtype=np.int64, count=h.height)
    ci = (h["d"].to_numpy() - (anchor - span + 1)).astype(np.int64)

    out = np.zeros((n, span, N_CHANNELS), dtype=dtype)
    for k, c in enumerate(SEQ_CHANNELS):
        out[ri, ci, k] = np.log1p(h[c].to_numpy().astype("float32"))
    out[ri, ci, len(SEQ_CHANNELS)] = 1.0                                  # день присутствует
    out[ri, ci, len(SEQ_CHANNELS) + 1] = h["search"].to_numpy().astype("float32")
    out[ri, ci, len(SEQ_CHANNELS) + 2] = h["cat"].to_numpy().astype("float32")
    return out


def build_targets(y: np.ndarray, df: pl.DataFrame, anchor: int,
                  users: pl.Series, horizon: int = 30,
                  center_level: bool = True) -> dict[str, np.ndarray]:
    """Главная цель плюс вспомогательные: они заставляют энкодер понять механизм.

    `center_level` вычитает уровень якоря из z, как это делает GBDT-ветка.
    Без него сеть выучивает уровень обучающих окон (около 2.62) и применяет
    его к валидационному (2.34) — метрика тогда меряет смещение, а не форму,
    и выглядит это как переобучение, хотя дело в другом.
    """
    t = (
        df.filter(pl.col("d").is_between(anchor + 1, anchor + horizon))
          .group_by("user_id")
          .agg(n_buy=(pl.col("gmv") > 0).sum(), n_ord=pl.col("to_ord").sum(),
               n_act=pl.len())
    )
    aux = (pl.DataFrame({"user_id": users}).join(t, on="user_id", how="left")
             .with_columns(pl.exclude("user_id").fill_null(0)).sort("user_id"))
    z = np.log1p(y)
    level = float(z[y > 0].mean()) if center_level and (y > 0).any() else 0.0
    # вычитаем условный уровень только у покупавших: у нулей z и так ноль,
    # и сдвигать их означало бы ломать точечную массу
    z = np.where(y > 0, z - level, 0.0)
    return {
        "z": z.astype("float32"),
        "level": np.float32(level),
        "p": (y > 0).astype("float32"),
        "n_buy": np.log1p(aux["n_buy"].to_numpy()).astype("float32"),
        "n_ord": np.log1p(aux["n_ord"].to_numpy()).astype("float32"),
        "n_act": np.log1p(aux["n_act"].to_numpy()).astype("float32"),
    }


@dataclass
class SeqConfig:
    span: int = 180
    hidden: int = 96
    embed_dim: int = 48
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)   # рецептивное поле ~127 дней
    dropout: float = 0.1
    lr: float = 2e-3
    batch_size: int = 512
    epochs: int = 8
    arch: str = "tcn"                                   # "tcn" или "gru"
    # Веса вспомогательных задач. Главная — z, остальные регуляризуют
    # представление, поэтому берутся заметно меньшими.
    aux_weights: dict = field(default_factory=lambda: {"p": 0.5, "n_buy": 0.3,
                                                       "n_ord": 0.2, "n_act": 0.2})
    seed: int = 42


def make_model(cfg: SeqConfig):
    """Энкодер с общим представлением и несколькими головами."""
    import torch
    from torch import nn

    class TCNBlock(nn.Module):
        def __init__(self, ch, k, d, p):
            super().__init__()
            self.pad = (k - 1) * d
            self.conv1 = nn.Conv1d(ch, ch, k, dilation=d)
            self.conv2 = nn.Conv1d(ch, ch, k, dilation=d)
            self.norm = nn.GroupNorm(4, ch)
            self.drop = nn.Dropout(p)

        def forward(self, x):
            # причинная свёртка: паддинг только слева, будущее не подглядывается
            h = self.conv1(nn.functional.pad(x, (self.pad, 0)))
            h = self.drop(nn.functional.gelu(h))
            h = self.conv2(nn.functional.pad(h, (self.pad, 0)))
            return nn.functional.gelu(self.norm(x + self.drop(h)))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(N_CHANNELS, cfg.hidden, 1)
            if cfg.arch == "tcn":
                self.body = nn.Sequential(*[TCNBlock(cfg.hidden, cfg.kernel, d, cfg.dropout)
                                            for d in cfg.dilations])
                self.gru = None
            else:
                self.body = nn.Identity()
                self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
            self.attn = nn.Linear(cfg.hidden, 1)
            self.proj = nn.Sequential(nn.Linear(cfg.hidden * 2, cfg.embed_dim), nn.GELU())
            self.heads = nn.ModuleDict({k: nn.Linear(cfg.embed_dim, 1)
                                        for k in ("z", "p", "n_buy", "n_ord", "n_act")})

        def embed(self, x):                      # x: (B, T, C)
            h = self.inp(x.transpose(1, 2))      # (B, H, T)
            if self.gru is None:
                h = self.body(h).transpose(1, 2)
            else:
                h, _ = self.gru(h.transpose(1, 2))
            # внимание по времени: сеть сама решает, какие эпизоды важны,
            # плюс последнее состояние — оно кодирует «что прямо сейчас»
            w = torch.softmax(self.attn(h), dim=1)
            return self.proj(torch.cat([(w * h).sum(1), h[:, -1]], dim=1))

        def forward(self, x):
            e = self.embed(x)
            return e, {k: head(e).squeeze(-1) for k, head in self.heads.items()}

    torch.manual_seed(cfg.seed)
    return Encoder()


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_encoder(
    Xtr: np.ndarray,
    ytr: dict[str, np.ndarray],
    Xva: np.ndarray,
    yva: dict[str, np.ndarray],
    cfg: SeqConfig | None = None,
    verbose: bool = True,
):
    """Обучить энкодер. Возвращает (модель, история валидации по главной цели)."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    cfg = cfg or SeqConfig()
    dev = pick_device()
    model = make_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    keys = ["z", "p", "n_buy", "n_ord", "n_act"]
    best_rmse, best_state = float("inf"), None
    ds = TensorDataset(torch.from_numpy(Xtr),
                       *[torch.from_numpy(ytr[k]) for k in keys])
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(dl))
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def evaluate():
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xva), 4096):
                xb = torch.from_numpy(Xva[i:i + 4096]).to(dev).float()
                _, o = model(xb)
                outs.append(o["z"].cpu().numpy())
        model.train()
        zp = np.concatenate(outs)
        return float(np.sqrt(np.mean((zp - yva["z"]) ** 2)))

    hist = []
    for ep in range(cfg.epochs):
        tot = 0.0
        for batch in dl:
            xb = batch[0].to(dev).float()
            tg = {k: batch[1 + i].to(dev).float() for i, k in enumerate(keys)}
            opt.zero_grad(set_to_none=True)
            _, o = model(xb)
            loss = mse(o["z"], tg["z"]) + cfg.aux_weights["p"] * bce(o["p"], tg["p"])
            for k in ("n_buy", "n_ord", "n_act"):
                loss = loss + cfg.aux_weights[k] * mse(o[k], tg[k])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.item())
        rmse = evaluate(); hist.append(rmse)
        if verbose:
            print(f"  эпоха {ep+1}/{cfg.epochs}  loss {tot/len(dl):.4f}  "
                  f"RMSE(z) на валидации {rmse:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  восстановлена лучшая эпоха: RMSE(z) {best_rmse:.5f}")
    return model, hist


def extract_embeddings(model, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Представление пользователя для подачи в GBDT."""
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(dev).float()
            out.append(model.embed(xb).cpu().numpy())
    model.train()
    return np.concatenate(out).astype("float32")
