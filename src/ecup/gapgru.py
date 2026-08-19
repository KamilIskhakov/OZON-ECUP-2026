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
    mlp: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    # Штраф на величину поправки: если в последовательности нет убедительного
    # нового сигнала, дешевле не трогать хороший базовый прогноз.
    lambda_delta: float = 0.05
    aux_weights: dict = field(default_factory=lambda: {
        "p": 0.2, "n_buy": 0.1, "n_ord": 0.05})
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
            B, L, _ = x.shape
            phi = torch.stack([gap, gap * gap], dim=-1)          # (B, L, 2)
            # γ ∈ (0, 1]: ReLU гарантирует неотрицательный показатель,
            # то есть затухание, а не рост состояния на паузе
            gamma = torch.exp(-torch.relu(self.decay(phi)))      # (B, L, H)
            h = x.new_zeros(B, self.hidden)
            out = []
            for t in range(L):
                h_dec = gamma[:, t] * h
                h_new = self.cell(x[:, t], h_dec)
                m = mask[:, t].unsqueeze(-1)
                # паддинг слева не должен двигать состояние
                h = torch.where(m, h_new, h)
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
            # запрос строится из текущего состояния И из прогноза ансамбля:
            # «что в прошлом релевантно тому, что модель предсказывает сейчас»
            self.q = nn.Linear(cfg.hidden + 3, dh)
            self.k = nn.Linear(cfg.hidden, dh)
            self.v = nn.Linear(cfg.hidden, dh)
            self.age_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(),
                                          nn.Linear(16, cfg.n_heads))
            layers, d = [], cfg.hidden + dh + 3
            for w in cfg.mlp:
                layers += [nn.Linear(d, w), nn.GELU(), nn.Dropout(cfg.dropout)]
                d = w
            self.trunk = nn.Sequential(*layers)
            self.head_dz = nn.Linear(d, 1)
            nn.init.zeros_(self.head_dz.weight); nn.init.zeros_(self.head_dz.bias)
            self.aux = nn.ModuleDict({k: nn.Linear(d, 1) for k in cfg.aux_weights})

        def forward(self, x, gap, age, mask, prior):
            """prior = (logit p0, m0, z0), нормированные снаружи."""
            h = self.inp(x)
            for layer in self.layers:
                h = layer(h, gap, mask)
            H = self.norm(h)                                    # (B, L, Hd)
            h_n = H[:, -1]                                      # правый край = якорь
            B, L, _ = H.shape
            q = self.q(torch.cat([h_n, prior], -1)).view(B, cfg.n_heads, cfg.d_head)
            k = self.k(H).view(B, L, cfg.n_heads, cfg.d_head)
            v = self.v(H).view(B, L, cfg.n_heads, cfg.d_head)
            s = torch.einsum('bhd,blhd->bhl', q, k) / (cfg.d_head ** 0.5)
            s = s + self.age_bias(age.unsqueeze(-1)).permute(0, 2, 1)
            s = s.masked_fill(~mask.unsqueeze(1), float('-inf'))
            a = torch.softmax(s, dim=-1)
            c = torch.einsum('bhl,blhd->bhd', a, v).reshape(B, -1)
            z = self.trunk(torch.cat([h_n, c, prior], -1))
            dz = self.head_dz(z).squeeze(-1)
            return dz, {k: head(z).squeeze(-1) for k, head in self.aux.items()}

    torch.manual_seed(cfg.seed)
    return GapGRUNet()


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
