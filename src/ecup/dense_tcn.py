"""Каузальный TCN на плотной дневной решётке.

Причинность обеспечивается ЛЕВЫМ паддингом: выход в позиции t зависит
только от входов до t включительно. Рецептивное поле при дилатациях
1..128 и ядре 3 составляет 1 + 2*(1+2+4+...+128) = 511 дней, то есть
покрывает всю решётку 384.

Голова поправки инициализирована нулями: на первом шаге dz тождественно
ноль, прогноз равен базовому, и единственный способ уменьшить лосс —
объяснить его ошибку. Тот же приём, что в Gap-GRU.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from torch import nn


@dataclass
class TCNConfig:
    n_features: int = 14
    max_len: int = 384
    channels: int = 64
    kernel: int = 3
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
    n_prior: int = 3
    dropout: float = 0.1
    lr: float = 2e-3
    batch_size: int = 256
    epochs: int = 12
    seed: int = 42
    lambda_delta: float = 0.05
    direct: bool = False   # True — самостоятельный прогноз z, а не поправка
    init_bias: float = 0.0


class Block(nn.Module):
    def __init__(self, ch, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(ch, ch, k, dilation=d)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=d)
        # BatchNorm, а НЕ GroupNorm: последняя нормирует внутри каждого
        # примера по каналам и времени, то есть приводит каждого
        # пользователя к нулевому среднему и стирает информацию об уровне —
        # ровно ту, ради которой модель и строится. На первом запуске это
        # дало std(dz) = 0 и неподвижный лосс.
        self.n1, self.n2 = nn.BatchNorm1d(ch), nn.BatchNorm1d(ch)
        self.do = nn.Dropout(p)

    def forward(self, x):
        h = torch.nn.functional.pad(x, (self.pad, 0))
        h = self.do(torch.relu(self.n1(self.c1(h))))
        h = torch.nn.functional.pad(h, (self.pad, 0))
        h = self.do(torch.relu(self.n2(self.c2(h))))
        return x + h


class DenseTCN(nn.Module):
    def __init__(self, cfg: TCNConfig):
        super().__init__()
        self.inp = nn.Conv1d(cfg.n_features, cfg.channels, 1)
        self.blocks = nn.ModuleList(
            [Block(cfg.channels, cfg.kernel, d, cfg.dropout) for d in cfg.dilations])
        # Пулинг по трём срокам: последний день, последний месяц, всё окно.
        # Так модель получает и мгновенное состояние, и агрегаты разной
        # давности, не теряя их в одном усреднении.
        self.head = nn.Sequential(
            nn.Linear(cfg.channels * 3 + cfg.n_prior, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        if cfg.direct:
            # Для самостоятельного прогноза нулевой старт бессмысленен:
            # обычная инициализация весов, смещение — в среднее z.
            nn.init.constant_(self.head[-1].bias, cfg.init_bias)
        else:
            # Для поправки нулевой старт даёт тождественное равенство базе.
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, prior):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        last = h[:, :, -1]
        m30 = h[:, :, -30:].mean(-1)
        allm = h.mean(-1)
        return self.head(torch.cat([last, m30, allm, prior], 1)).squeeze(-1)


def make_tcn(cfg: TCNConfig) -> DenseTCN:
    torch.manual_seed(cfg.seed)
    return DenseTCN(cfg)
