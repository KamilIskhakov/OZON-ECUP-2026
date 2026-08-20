"""Gap-GRU с query-attention и остаточной головой к прогнозу ансамбля.

Идея опирается на то, что уже измерено, а не на общие соображения. Лучшие
семьи признаков у нас — состояния с забыванием: EWMA с тремя фиксированными
δ дала +0.0021, дисконтированные байесовские фильтры +0.0028. Обе имеют вид

    S_t = δ S_{t-1} + x_t

с ЗАДАННОЙ наперёд скоростью забывания. GRU обобщает это до скорости,
зависящей от входа и состояния, а gap-decay добавляет зависимость от паузы:

    γ_j = exp[-ReLU(W_γ φ(Δt_j) + b_γ)],     h̄_{j-1} = γ_j ⊙ h_{j-1}

Существенно, что γ_j ∈ R^d, а не скаляр: разные координаты памяти забывают
с разной скоростью. Одна может держать денежный масштаб месяцами, другая —
намерение купить несколько дней. Это ровно обучаемый многомерный EWMA,
и именно поэтому здесь ожидается больше, чем от TCN с фиксированным
рецептивным полем.

Attention решает другую задачу, не ту же самую. GRU отвечает «в каком
состоянии пользователь сейчас», h_n. Один запрос ко всей памяти отвечает
«какой эпизод прошлого похож на нынешнее состояние». Полного self-attention
здесь нет сознательно: при 192 токенах он добавил бы параметры и дисперсию,
не добавив конкретного indutive bias.

Голова остаточная и инициализирована нулями: в начале обучения Δz = 0 и
модель тождественно равна текущему ансамблю. Единственный способ уменьшить
лосс — объяснить его ошибку. Заново открывать recency, GMV90 и AOV
не требуется, они уже в z0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch  # noqa: F401  (нужен для Parameter в make_model)

import numpy as np


@dataclass
class GapGRUConfig:
    n_features: int = 20
    max_len: int = 192
    d_in: int = 64
    hidden: int = 96
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 32
    # Обучаемые запросы по поведенческим факторам вместо одного пулинга.
    # Разложение взято из BTYD, но параметры не постулируются: сеть сама
    # решает, какие эпизоды нужны для оценки частоты, а какие — для чека.
    # Один запрос вынужден одним вектором обслуживать все головы сразу;
    # четыре специализируются.
    queries: tuple[str, ...] = ("freq", "amount", "intent", "activity")
    # Ветка покупочных циклов: маленький трансформер по 16 токенам.
    use_cycles: bool = False
    n_cycle_features: int = 8
    max_cycles: int = 16
    cycle_dim: int = 64
    cycle_layers: int = 2
    mlp: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    # Штраф на величину поправки: если в последовательности нет убедительного
    # нового сигнала, дешевле не трогать хороший базовый прогноз.
    lambda_delta: float = 0.05
    # z_abs нужна на этапе A: там нет базового прогноза, и энкодер учится
    # предсказывать сам уровень поведения. На этапе B её вес снижается —
    # она становится регуляризатором представления, а не целью.
    aux_weights: dict = field(default_factory=lambda: {
        "p": 0.2, "n_buy": 0.1, "n_ord": 0.05, "z_abs": 0.1})
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 12
    seed: int = 42


def make_model(cfg: GapGRUConfig):
    import torch
    from torch import nn

    class GapGRULayer(nn.Module):
        """GRU, у которого состояние затухает по паузе перед каждым шагом."""

        def __init__(self, d_in, hidden):
            super().__init__()
            self.cell = nn.GRUCell(d_in, hidden)
            # φ(Δt) = [log1p(Δt), log1p(Δt)²] — мягкая нелинейность по паузе
            self.decay = nn.Linear(2, hidden)
            nn.init.zeros_(self.decay.bias)
            nn.init.uniform_(self.decay.weight, 0.0, 0.1)
            self.hidden = hidden

        def forward(self, x, gap, mask):
            phi = torch.stack([gap, gap * gap], dim=-1)          # (B, L, 2)
            # γ ∈ (0, 1]: ReLU гарантирует неотрицательный показатель,
            # то есть затухание, а не рост состояния на паузе
            gamma = torch.exp(-torch.relu(self.decay(phi)))      # (B, L, H)
            h = x.new_zeros(x.shape[0], self.hidden)
            out = []
            # unbind, а не x[:, t] внутри цикла. Срез по времени — отдельная
            # операция автограда, и её обратный проход создаёт ПОЛНОРАЗМЕРНЫЙ
            # нулевой тензор (B, L, H), чтобы рассеять в него градиент одного
            # шага. При 192 шагах на слой это 192 аллокации по 151 МБ и столько
            # же сложений. unbind делает то же разбиение одной операцией,
            # обратный проход которой — один stack. Замер на A4000, батч 2048:
            # обратный проход 1143 мс против 78.5 мс, суммарно 9.7 раза.
            for xt, gt, mt in zip(x.unbind(1), gamma.unbind(1), mask.unbind(1)):
                # паддинг слева не должен двигать состояние
                h = torch.where(mt.unsqueeze(-1), self.cell(xt, gt * h), h)
                out.append(h)
            return torch.stack(out, dim=1)

    class GapGRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(cfg.n_features, cfg.d_in), nn.GELU())
            dims = [cfg.d_in] + [cfg.hidden] * cfg.n_layers
            self.layers = nn.ModuleList(
                [GapGRULayer(dims[i], dims[i + 1]) for i in range(cfg.n_layers)])
            self.norm = nn.LayerNorm(cfg.hidden)
            dh = cfg.n_heads * cfg.d_head
            self.nq = len(cfg.queries)
            # Каждый запрос — свой обучаемый вектор ПЛЮС проекция текущего
            # состояния и прогноза ансамбля: «что в прошлом релевантно этому
            # фактору при том, что модель предсказывает сейчас».
            self.q_emb = nn.Parameter(torch.zeros(self.nq, cfg.hidden + 3))
            nn.init.normal_(self.q_emb, std=0.02)
            self.q = nn.Linear(cfg.hidden + 3, dh)
            self.k = nn.Linear(cfg.hidden, dh)
            self.v = nn.Linear(cfg.hidden, dh)
            self.age_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(),
                                          nn.Linear(16, cfg.n_heads))
            self.cyc = None
            d_cyc = 0
            if cfg.use_cycles:
                # Циклы — вторая шкала времени. Дневная ветка отвечает
                # «что происходит сейчас», циклическая — «как обычно устроен
                # ритм этого пользователя». Последовательности 8,9,8,10,27
                # и 12,11,10,11,10 имеют близкую медиану интервала и разную
                # динамику; агрегаты эту разницу стирают.
                self.cyc_in = nn.Sequential(
                    nn.Linear(cfg.n_cycle_features, cfg.cycle_dim), nn.GELU())
                self.cyc_pos = nn.Parameter(torch.zeros(cfg.max_cycles, cfg.cycle_dim))
                nn.init.normal_(self.cyc_pos, std=0.02)
                enc = nn.TransformerEncoderLayer(
                    cfg.cycle_dim, nhead=4, dim_feedforward=cfg.cycle_dim * 2,
                    dropout=cfg.dropout, batch_first=True, norm_first=True)
                self.cyc = nn.TransformerEncoder(enc, cfg.cycle_layers)
                d_cyc = cfg.cycle_dim
            layers, d = [], cfg.hidden + dh * self.nq + d_cyc + 3
            for w in cfg.mlp:
                layers += [nn.Linear(d, w), nn.GELU(), nn.Dropout(cfg.dropout)]
                d = w
            self.trunk = nn.Sequential(*layers)
            self.head_dz = nn.Linear(d, 1)
            nn.init.zeros_(self.head_dz.weight); nn.init.zeros_(self.head_dz.bias)
            self.aux = nn.ModuleDict({k: nn.Linear(d, 1) for k in cfg.aux_weights})

        def forward(self, x, gap, age, mask, prior, cycles=None):
            """prior — (z0, расхождение семейств, длина), нормированные снаружи."""
            h = self.inp(x)
            for layer in self.layers:
                h = layer(h, gap, mask)
            H = self.norm(h)                                    # (B, L, Hd)
            h_n = H[:, -1]                                      # правый край = якорь
            B, L, _ = H.shape
            base = torch.cat([h_n, prior], -1).unsqueeze(1)     # (B, 1, Hd+3)
            q = self.q(base + self.q_emb.unsqueeze(0))          # (B, nq, dh)
            q = q.view(B, self.nq, cfg.n_heads, cfg.d_head)
            k = self.k(H).view(B, L, cfg.n_heads, cfg.d_head)
            v = self.v(H).view(B, L, cfg.n_heads, cfg.d_head)
            s = torch.einsum('bqhd,blhd->bqhl', q, k) / (cfg.d_head ** 0.5)
            s = s + self.age_bias(age.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(1)
            s = s.masked_fill(~mask[:, None, None, :], float('-inf'))
            a = torch.softmax(s, dim=-1)
            c = torch.einsum('bqhl,blhd->bqhd', a, v).reshape(B, -1)
            feats = [h_n, c, prior]
            if self.cyc is not None:
                if cycles is None:
                    raise ValueError("модель собрана с use_cycles, но циклы не поданы")
                hc = self.cyc(self.cyc_in(cycles) + self.cyc_pos.unsqueeze(0))
                feats.insert(2, hc[:, -1])                      # последний цикл = ближайший
            z = self.trunk(torch.cat(feats, -1))
            dz = self.head_dz(z).squeeze(-1)
            return dz, {k: head(z).squeeze(-1) for k, head in self.aux.items()}

    torch.manual_seed(cfg.seed)
    return GapGRUNet()


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
