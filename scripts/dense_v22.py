"""Плотный бэкбон против разреженного при РАВНОМ числе моделей.

Бэкбон v22 состоит из тридцати моделей. Заменить его на шесть плотных
значило бы поменять две вещи сразу: плотность якорей и размер ансамбля.
Поэтому обучается парный контроль — те же шесть моделей при stride 30.

Никаких новых признаков и гиперпараметров: спецификация v22 не меняется,
меняется только шаг якорей.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, HurdleGBDT, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue, rmsle)
from ecup.dataset import anchor_offsets
from ecup.catboost_model import HurdleCatBoost, CatBoostConfig
df = load_panel(); OUT = Path('artifacts/neural'); SEEDS = (42, 7, 2026)
CAPS = {'lgb': (250, 150), 'cb': (950, 1550)}      # из профиля по валидационному якорю

def run(stride, n_anch):
    sp = SplitConfig(max_history=300, n_train_anchors=n_anch, stride=stride, with_state=True)
    an = sp.train_anchors()
    Xtr_df, ytr, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    Xtr, feats = to_matrix(Xtr_df); del Xtr_df; gc.collect()
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    val = build_anchor(df, sp.val_anchor, sp, None)
    Xva, _ = to_matrix(val.X, feats); z = np.log1p(val.y)
    print(f'stride {stride}: якорей {len(an)}, строк {len(ytr):,}', flush=True)
    res = {}
    for fam in ('lgb', 'cb'):
        zs = []
        for s in SEEDS:
            t0 = time.perf_counter()
            if fam == 'lgb':
                cfg = ModelConfig(seed=s, early_stopping_rounds=None); cls = HurdleGBDT
                cfg.clf_params['n_estimators'], cfg.reg_params['n_estimators'] = CAPS['lgb']
            else:
                cfg = CatBoostConfig(seed=s, early_stopping_rounds=None); cls = HurdleCatBoost
                cfg.clf_params['iterations'], cfg.reg_params['iterations'] = CAPS['cb']
            m = cls(config=cfg).fit(Xtr, ytr, feature_names=feats, sample_weight=w,
                                    z_offset=zo, clf_init=ci)
            p, mm = m.predict_parts(Xva, p_target=last.p_bar, m_offset=last.l_plus)
            zs.append(np.log1p(hurdle_glue(p, np.clip(mm, 0, None))))
            print(f'  {fam} сид {s}: shape {(z-zs[-1]).std():.5f} · '
                  f'{time.perf_counter()-t0:.0f}с', flush=True)
            del m; gc.collect()
        res[fam] = np.mean(zs, 0)
    del Xtr, Xva; gc.collect()
    return res, z

r10, z = run(10, 18)
r30, _ = run(30, 6)
E = np.load('/tmp/cb_ens2.npz')
zl_full = np.mean([E[k] for k in E.files if k.startswith('lgb')],0)
zc_full = np.mean([E[k] for k in E.files if k.startswith(('cb_','brd','dpw'))],0)
d16 = np.load(OUT/'dz_a378.npz')['dz']; dann = np.load('/tmp/d_annual_378.npy')
CORR = 0.35*(d16-d16.mean()) + 0.0104*(dann-dann.mean())   # те же поправки, что в v22
def sh(v): return float((z-v).std())
back = {'плотный 3+3': 0.4*r10['lgb']+0.6*r10['cb'],
        'разреженный 3+3': 0.4*r30['lgb']+0.6*r30['cb'],
        'полный v22 (30 моделей)': 0.4*zl_full+0.6*zc_full}
print(f'\n{"бэкбон":<26} {"без поправок":>14} {"с поправками v22":>18}')
for nm, b in back.items():
    print(f'  {nm:<24} {sh(b):>14.5f} {sh(b+CORR):>18.5f}')
np.savez(OUT/'dense_backbone_378.npz', lgb10=r10['lgb'], cb10=r10['cb'],
         lgb30=r30['lgb'], cb30=r30['cb'], z=z)
base = back['полный v22 (30 моделей)'] + CORR
print(f'\nсмесь плотного бэкбона с полным v22:')
for lam in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
    v = (1-lam)*base + lam*(back['плотный 3+3']+CORR)
    print(f'  lambda {lam:.2f}: shape {sh(v):.5f}'
          + ('   <- текущий v22' if lam == 0 else ''))
