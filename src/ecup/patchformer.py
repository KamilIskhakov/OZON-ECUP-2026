"""Многомасштабный трансформер над патчами: второй сильный предиктор.

Роль отличается от Gap-GRU. Тот видит событийную динамику последних дней;
этот видит СОВМЕСТНО недельную и годовую структуру. Именно совместности
не хватало: годовая форма извлекалась вручную заданной гармоникой, а
свежая динамика — рекуррентной сетью, и связать их было нечем.

Три решения выведены из измерений, а не из общих соображений.

Разложение level/shape подаётся явно: измеренный supressor-эффект показал,
что уровень маскирует форму (ковариация выросла втрое после его удаления),
а линейный Ridge по блокам отдал все веса уровневым признакам. Надеяться,
что attention разложит сам, оснований нет.

Календарная фаза каждого патча считается от центра ЦЕЛЕВОГО окна — так
была зафиксирована фаза годового фильтра, который сработал.

Голова поправки инициализирована нулями: на старте модель тождественна
деревьям плюс уже найденные поправки, и единственный способ уменьшить
лосс — объяснить их общую ошибку.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatchFormerConfig:
    n_ch: int = 16                 # 8 каналов × (исходный + центрированный)
    n_pos: int = 5
    n_prior: int = 7
    d_model: int = 128
    n_blocks: int = 4
    n_heads: int = 4
    n_queries: int = 4
    mlp: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    aux: dict = field(default_factory=lambda: {
        "z7": 0.05, "z14": 0.1, "z30": 1.0, "c30": 0.05,
        "n_ord": 0.05, "n_buy": 0.05, "z_s": 0.05, "z_c": 0.05})
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 20
    seed: int = 42


def make_patchformer(cfg: PatchFormerConfig):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # у потоков разный масштаб времени, поэтому проекции раздельные
            self.pw = nn.Linear(cfg.n_ch + cfg.n_pos, cfg.d_model)
            self.pm = nn.Linear(cfg.n_ch + cfg.n_pos, cfg.d_model)
            self.stream = nn.Parameter(torch.zeros(2, cfg.d_model))
            nn.init.normal_(self.stream, std=0.02)
            blk = nn.TransformerEncoderLayer(
                cfg.d_model, cfg.n_heads, cfg.d_model * 2, cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(blk, cfg.n_blocks)
            self.q = nn.Parameter(torch.zeros(cfg.n_queries, cfg.d_model))
            nn.init.normal_(self.q, std=0.02)
            self.qproj = nn.Linear(cfg.n_prior, cfg.d_model)
            self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                              dropout=cfg.dropout, batch_first=True)
            layers, d = [], cfg.d_model * cfg.n_queries + cfg.n_prior
            for w in cfg.mlp:
                layers += [nn.Linear(d, w), nn.GELU(), nn.Dropout(cfg.dropout)]
                d = w
            self.trunk = nn.Sequential(*layers)
            self.head = nn.Linear(d, 1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            self.aux = nn.ModuleDict({k: nn.Linear(d, 1) for k in cfg.aux})

        def forward(self, xw, xm, pw, pm, prior):
            B = xw.shape[0]
            tw = self.pw(torch.cat([xw, pw.expand(B, -1, -1)], -1)) + self.stream[0]
            tm = self.pm(torch.cat([xm, pm.expand(B, -1, -1)], -1)) + self.stream[1]
            h = self.enc(torch.cat([tw, tm], 1))          # 38 токенов обоих масштабов
            # запросы смещены прогнозом деревьев: «что они ещё не знают»
            q = self.q.unsqueeze(0).expand(B, -1, -1) + self.qproj(prior).unsqueeze(1)
            c, _ = self.attn(q, h, h, need_weights=False)
            z = self.trunk(torch.cat([c.reshape(B, -1), prior], -1))
            return self.head(z).squeeze(-1), {k: hd(z).squeeze(-1)
                                              for k, hd in self.aux.items()}

    torch.manual_seed(cfg.seed)
    return Net()
