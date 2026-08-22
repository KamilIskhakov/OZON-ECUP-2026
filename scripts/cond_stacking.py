"""Условное стекинг: есть ли сегменты, где экспертов надо перевешивать.

Вопрос узкий и не про «ещё один learner над 183 признаками»:
существуют ли устойчивые группы пользователей, где уже имеющиеся
модели систематически стоит смешивать иначе, чем в среднем.

Работаем ТОЛЬКО в координатах разногласия q_k = z_k - z0 и их
взаимодействиях с несколькими заранее названными переменными.
Метамодель — Ridge, то есть детерминированная: после сегодняшнего
замера sigma = 2-2.8e-4 у парных обучений деревьев любая
стохастическая метамодель просто утонула бы в собственном шуме.

Перенос настоящий walk-forward: 288+318 -> 348, затем 288+318+348 -> 378.
"""
import sys, warnings, gc; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from ecup import SplitConfig, load_panel, build_anchor
ANCH = (288, 318, 348, 378)
XCOLS = ('r_gmv', 'avail_history', 'act_days_90')     # давность, возраст истории, активность
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
X = np.load('artifacts/neural/xgb_walkforward.npz')
S = {}
for A in ANCH:
    o = np.load(f'artifacts/neural/oofpm_a{A}.npz')
    v = build_anchor(df, A, sp, None)
    f = pl.DataFrame({'user_id': v.X['user_id'].to_numpy(),
                      **{c: v.X[c].to_numpy().astype('float64') for c in XCOLS}})
    t = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']), 'z0': o['z0'],
                      'lgb': o['z0_lgb'], 'cb': o['z0_cb'], 'p0': o['p0'],
                      'h': X[f'h_{A}'], 'd': X[f'd_{A}']}).join(f, on='user_id', how='inner')
    assert len(t) == len(o['user_id'])
    g = {k: t[k].to_numpy() for k in t.columns if k != 'user_id'}
    E = np.column_stack([g['lgb'], g['cb'], g['h'], g['d']])
    q = np.column_stack([g['cb'] - g['lgb'], g['h'] - g['z0'],
                         g['d'] - g['z0'], E.std(1)])
    mods = np.column_stack([g[c] for c in XCOLS] + [g['p0']])
    mods = (mods - mods.mean(0)) / (mods.std(0) + 1e-9)
    Q = np.column_stack([q] + [q * mods[:, [j]] for j in range(mods.shape[1])])
    S[A] = dict(z=g['z'], z0=g['z0'], e=g['z'] - g['z0'], Q=Q, q=q)
    print(f'якорь {A}: {len(g["z"]):,} пользователей · координат {Q.shape[1]} · '
          f'база {S[A]["e"].std():.5f}', flush=True)
    del v; gc.collect()


KEY = 'Q'

def fit(tr, alpha):
    A_ = np.vstack([S[a][KEY][:, COLS] for a in tr]); b = np.concatenate([S[a]['e'] for a in tr])
    mu, sd = A_.mean(0), A_.std(0) + 1e-12
    An = (A_ - mu) / sd; An = np.column_stack([np.ones(len(An)), An])
    G = An.T @ An; G[1:, 1:] += alpha * np.eye(G.shape[0] - 1)
    return np.linalg.solve(G, An.T @ b), mu, sd


def apply(a, c):
    w, mu, sd = c
    Z = np.column_stack([np.ones(len(S[a][KEY])), (S[a][KEY][:, COLS] - mu) / sd])
    return Z @ w


# q = [cb-lgb, h-z0, d-z0, spread]; Q = q и его 4 взаимодействия
VAR = {
    'всё, условно':        ('Q', list(range(20))),
    'всё, БЕЗ условий':    ('q', [0, 1, 2, 3]),
    'только cb-lgb, усл.': ('Q', [0, 4, 8, 12, 16]),
    'только XGB, усл.':    ('Q', [1, 2, 5, 6, 9, 10, 13, 14, 17, 18]),
    'только spread, усл.': ('Q', [3, 7, 11, 15, 19]),
}
for nm, (KEY, COLS) in VAR.items():
    globals()['KEY'], globals()['COLS'] = KEY, COLS
    print(f'\n--- {nm} · координат {len(COLS)}')
    print(f'{"обучено → оценено":>22}{"база":>10}{"после":>10}{"выигрыш":>10}{"alpha":>10}')
    for tr, te in ((ANCH[:2], ANCH[2]), (ANCH[:3], ANCH[3])):
    # alpha выбирается ВНУТРИ обучающих якорей: последний из них — holdout,
    # оценочный якорь при выборе гиперпараметра не участвует
        best, ba = None, None
        for al in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            c = fit(tr[:-1], al); v = (S[tr[-1]]['e'] - apply(tr[-1], c)).std()
            if best is None or v < best: best, ba = v, al
        c = fit(tr, ba); p = apply(te, c)
        s0 = S[te]['e'].std(); s1 = (S[te]['e'] - p).std()
        print(f'{f"{list(tr)} → {te}":>22}{s0:>10.5f}{s1:>10.5f}{s0-s1:>+10.5f}{ba:>10.0f}')
