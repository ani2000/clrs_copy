"""Are the feature labels shown on the page true? For each algorithm, recompute precision / recall / F1 of the
top labelled features on FRESH graphs (new seed), using the same predicates, then compare with feature_labels.json."""
import json, os, os, sys
import numpy as np, jax, clrs
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP); os.chdir(APP)
from clrs.examples import run as clrs_run
clrs_run.FLAGS(['x'])
import server_128 as S
from config import ALGOS
K = 8
ONLY = os.environ.get('ONLY')
worst_all = 0
for algo in (ONLY.split(',') if ONLY else ALGOS):
    meta = S.META[algo]; spec = clrs.SPECS[algo]; r = S.REG[algo]
    t = np.load(f'algos/{algo}/transcoder.npz'); fl = json.load(open(f'algos/{algo}/feature_labels.json'))
    tgt, ttype = meta['target'], meta['target_type']
    src_field = next((n for n, (st, loc, ty) in spec.items() if st == 'input' and loc == 'node' and ty == 'mask_one'), None)
    rng = np.random.RandomState(4242)
    train, *_ = clrs_run.create_samplers(rng=rng, train_lengths=[16], algorithms=[algo], val_lengths=[16], test_lengths=[16],
                                         train_batch_size=32, val_batch_size=32, test_batch_size=32)
    X, P, Nx, SRC = [], [], [], []
    for bi in range(int(os.environ.get('NB', 12))):
        fb = next(train[0])
        hints = {h.name: np.asarray(h.data) for h in fb.features.hints}; th = hints[tgt]
        src = np.asarray(fb.features.inputs[[i.name for i in fb.features.inputs].index(src_field)].data) if src_field else None
        _, _, hid, z, msgs = r['net'].apply(r['params'], jax.random.PRNGKey(bi + 99), [fb.features], True, 0, True, False)
        z, msgs = np.asarray(z), np.asarray(msgs); T, B, N, _ = z.shape; L = np.asarray(fb.features.lengths)
        for b in range(B):
            for s in range(min(T, int(L[b]) - 1)):
                for n in range(N):
                    X.append(np.concatenate([z[s, b, n], msgs[s, b, n]])); P.append(th[s, b, n]); Nx.append(th[s + 1, b, n])
                    SRC.append(1.0 if src is not None and n == src[b].argmax() else 0.0)
    X = np.array(X); pre = (X - t['mu_x']) / t['sd_x'] @ t['w_enc'] + t['b_enc']; act = np.maximum(pre, 0)
    top = np.argsort(-act, 1)[:, :K]; mask = np.zeros_like(act); np.put_along_axis(mask, top, 1, 1); feats = act * mask
    prev, nxt, is_src = np.array(P, dtype=object), np.array(Nx, dtype=object), np.array(SRC)
    preds = {'is_source': is_src == 1} if src_field else {}
    if ttype in ('mask', 'mask_one'):
        p0, n0 = prev.astype(float), nxt.astype(float)
        preds.update({'stays 0': (p0 == 0) & (n0 == 0), 'newly 1': (p0 == 0) & (n0 == 1), 'stays 1': (p0 == 1) & (n0 == 1), 'newly 0': (p0 == 1) & (n0 == 0)})
    elif ttype == 'scalar':
        p0, n0 = prev.astype(float), nxt.astype(float)
        preds.update({'increases': n0 > p0 + 1e-6, 'decreases': n0 < p0 - 1e-6, 'stays same': np.abs(n0 - p0) <= 1e-6})
    else:
        n0 = np.stack(nxt).argmax(-1)
        for c in range(np.stack(nxt).shape[-1]): preds[f'becomes class {c}'] = n0 == c
    preds.update({'not ' + k: ~v for k, v in list(preds.items())})
    rows = [x for x in fl['rows'] if x.get('f1')][:10]
    diffs = []
    for x in rows:
        fires = feats[:, x['feature']] > 0; pr = preds[x['label']]
        tp = (fires & pr).sum(); p = tp / max(1, fires.sum()); rc = tp / max(1, pr.sum()); f1 = 2 * p * rc / max(1e-9, p + rc)
        diffs.append(abs(f1 - x['f1']))
    worst = max(diffs); worst_all = max(worst_all, worst)
    print(f"{algo:30s} {len(X):7d} fresh sites | top-10 labels: mean |ΔF1| {np.mean(diffs):.3f}, worst {worst:.3f}"
          f"  (e.g. f{rows[0]['feature']} '{rows[0]['label']}' stored F1 {rows[0]['f1']:.3f})", flush=True)
print('worst |ΔF1| over all algorithms:', round(worst_all, 3))
