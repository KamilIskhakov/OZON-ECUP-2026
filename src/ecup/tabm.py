"""TabM: параметрически эффективный ансамбль MLP над готовыми признаками.

Обоснование из наших же данных, а не из литературы: единственное, что
надёжно работало — разнообразие семейств (+0.0019 суммарно), а не более
сложная обработка последовательности. При этом 183 инженерных признака
обрабатывались ровно двумя функциональными классами, LightGBM и CatBoost;
нейросети использовались только как sequence-модели.

Ядро: k членов ансамбля делят матрицу W, различаясь покомпонентными
адаптерами r_k на входе и s_k на выходе (BatchEnsemble). Стоимость
почти как у одной сети, а ошибки членов декоррелированы.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TabMConfig:
    n_features: int = 183
    k: int = 32
    hidden: tuple[int, ...] = (512, 512, 512)
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    epochs: int = 30
    seed: int = 42


def make_tabm(cfg: TabMConfig):
    import torch
    from torch import nn

    class BELinear(nn.Module):
        """Общая матрица плюс покомпонентные адаптеры на вход и выход."""
        def __init__(self, d_in, d_out, k):
            super().__init__()
            self.lin = nn.Linear(d_in, d_out)
            self.r = nn.Parameter(torch.ones(k, d_in))
            self.s = nn.Parameter(torch.ones(k, d_out))
            self.b = nn.Parameter(torch.zeros(k, d_out))
            # инициализация адаптеров знакопеременная: иначе члены ансамбля
            # стартуют тождественными и декоррелироваться им неоткуда
            with torch.no_grad():
                self.r.copy_(torch.randint(0, 2, self.r.shape).float()*2 - 1)
                self.s.copy_(torch.randint(0, 2, self.s.shape).float()*2 - 1)

        def forward(self, x):                    # x: (B, k, d_in)
            return self.lin(x * self.r) * self.s + self.b

    class TabM(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [cfg.n_features] + list(cfg.hidden)
            self.layers = nn.ModuleList(
                [BELinear(dims[i], dims[i+1], cfg.k) for i in range(len(dims)-1)])
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in cfg.hidden])
            self.drop = nn.Dropout(cfg.dropout)
            self.head = BELinear(cfg.hidden[-1], 1, cfg.k)

        def forward(self, x):                    # x: (B, d)
            h = x.unsqueeze(1).expand(-1, cfg.k, -1)
            for lin, nrm in zip(self.layers, self.norms):
                h = self.drop(nn.functional.gelu(nrm(lin(h))))
            return self.head(h).squeeze(-1)      # (B, k) — по одному на члена

    torch.manual_seed(cfg.seed)
    return TabM()
