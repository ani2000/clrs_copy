"""Export each algorithm's MPNN (encoders, processor, HINT decoders) as a small
binary weight bundle + JSON manifest, and build the compact per-graph
'instance' the in-browser forward pass (mpnn.js) consumes.

Bundle layout: models/<algo>.json (manifest) + models/<algo>.bin (float32 LE).
Only what the browser needs to reproduce (z, msgs) is exported: output-stage
decoders are omitted because hints are fed back from HINT decoders only.
"""
import json, os, pickle, sys
import numpy as np, clrs
from config import CKPT_DIR, ALGOS

HIDDEN = 128
PROC = 'net/mpnn_aggr_clrs_processor'
ENC = 'net/~_construct_encoders_decoders'
# processor layer roles (creation order in PGNCapture.__call__)
PROC_ROLES = {'linear': 'm1', 'linear_1': 'm2', 'linear_2': 'me', 'linear_3': 'mg',
              'linear_4': 'o1', 'linear_5': 'o2'}


def _ckpt(algo):
    return pickle.load(open(f'{CKPT_DIR}/{algo}/best.pkl', 'rb'))['params']


def export_model(algo):
    spec = clrs.SPECS[algo]
    p = _ckpt(algo)
    chunks, off = [], 0

    def put(arr):
        nonlocal off
        a = np.asarray(arr, dtype='<f4').ravel()
        chunks.append(a)
        start = off
        off += a.size
        return start

    def lin(key):
        v = p[key]
        din, dout = v['w'].shape
        return dict(w=put(v['w']), b=put(v['b']), din=int(din), dout=int(dout))

    man = dict(algo=algo, H=HIDDEN, fields=[], proc={})
    for role_key, role in PROC_ROLES.items():
        man['proc'][role] = lin(f'{PROC}/{role_key}')
    man['proc']['mlp0'] = lin(f'{PROC}/mlp/~/linear_0')
    man['proc']['mlp1'] = lin(f'{PROC}/mlp/~/linear_1')
    ln = p[f'{PROC}/layer_norm']
    man['proc']['ln'] = dict(scale=put(ln['scale']), offset=put(ln['offset']))

    for name, (stage, loc, typ) in spec.items():
        f = dict(name=name, stage=stage, loc=loc, type=typ)
        if stage in ('input', 'hint'):
            ekeys = [k for k in p if k.startswith(f'{ENC}/algo_0_{name}_enc_linear') and
                     (k == f'{ENC}/algo_0_{name}_enc_linear' or k[len(f'{ENC}/algo_0_{name}_enc_linear'):].startswith('_'))]
            ekeys.sort(key=lambda k: (len(k), k))
            f['enc'] = [lin(k) for k in ekeys]
            f['K'] = f['enc'][0]['din']            # 1 for scalar-like, class count for categorical
        if stage == 'hint':
            dkeys = [k for k in p if k.startswith(f'{ENC}/algo_0_{name}_dec_linear')]
            dkeys.sort(key=lambda k: (len(k), k))
            f['dec'] = [lin(k) for k in dkeys]
        man['fields'].append(f)
    man['nfloats'] = off
    return man, np.concatenate(chunks)


def write_all(outdir='models'):
    os.makedirs(outdir, exist_ok=True)
    for a in ALGOS:
        man, blob = export_model(a)
        json.dump(man, open(f'{outdir}/{a}.json', 'w'), separators=(',', ':'))
        blob.astype('<f4').tofile(f'{outdir}/{a}.bin')
        print(f'{a:32s} {blob.size * 4 / 1e6:.2f} MB')


def _tolist(a):
    a = np.asarray(a)
    return a.tolist()


def build_instance(algo, n, p, seed, meta):
    """Compact instance from CLRS's own sampler (same graphs as /api/graph)."""
    for tries in range(8):
        sampler, _ = clrs.build_sampler(algo, num_samples=1, seed=seed, length=n, p=(p,))
        fb = sampler.next()
        L = int(np.asarray(fb.features.lengths)[0])
        if L >= 3:
            break
        seed = seed + 1000 + tries
    inputs = {d.name: _tolist(np.asarray(d.data)[0]) for d in fb.features.inputs}
    hints0 = {d.name: _tolist(np.asarray(d.data)[0, 0]) for d in fb.features.hints}
    names_i = [i.name for i in fb.features.inputs]
    adj = np.asarray(fb.features.inputs[names_i.index('adj')].data)[0]
    source = None
    if meta['source_field']:
        s_in = np.asarray(fb.features.inputs[names_i.index(meta['source_field'])].data)[0]
        source = int(np.argmax(s_in))
    target_true = None
    if meta['target_stage'] == 'hint':
        names_h = [h.name for h in fb.features.hints]
        tt = np.asarray(fb.features.hints[names_h.index(meta['target'])].data)[:, 0]
        if meta['target_type'] == 'categorical':
            tt = tt.argmax(-1).astype(float)
        target_true = tt[:L].tolist()
    return dict(n=int(adj.shape[0]), length=L, seed=seed, adj=adj.tolist(), source=source,
                target_true=target_true, inputs=inputs, hints0=hints0), fb


if __name__ == '__main__':
    write_all(sys.argv[1] if len(sys.argv) > 1 else 'models')
