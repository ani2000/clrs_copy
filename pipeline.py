"""Generic Phase 2+3+4 pipeline for the real 128-hidden checkpoints_v3, all
12 graph algorithms. Reuses capture_net.CaptureNet unmodified (it's already
algorithm-agnostic, parameterized only by `spec`). No training happens here
-- these checkpoints are the ones from Final_clrs30_graph_algos_mpnn_v3.ipynb.
"""
import json, pickle, time, sys
import numpy as np, jax, jax.numpy as jnp, haiku as hk, clrs, optax
from clrs.examples import run as clrs_run
from capture_net import CaptureNet, get_capture_processor_factory
from config import CKPT_DIR  # single source of truth -- checkpoints_v3 only

clrs_run.FLAGS(['x'])
HIDDEN, F, K = 128, 192, 8
N_EPISODES, BATCH, TRAIN_N, SEED = 1500, 32, 16, 9001
RESULTS_DIR = "/Users/anindya/CLRS/new run/final results and notebook(v3 versions)/results_v3"
OUT_DIR = "algos"

TARGETS = {
    'bfs': 'reach_h', 'dfs': 'color', 'bellman_ford': 'd', 'dijkstra': 'd',
    'dag_shortest_paths': 'd', 'mst_prim': 'key', 'mst_kruskal': 'u',
    'topological_sort': 'topo_head_h', 'articulation_points': 'is_cut_h',
    'bridges': 'color', 'strongly_connected_components': 'color', 'floyd_warshall': 'k',
}


def run_algo(algo, log):
    t0 = time.time()
    spec = clrs.SPECS[algo]
    target_name = TARGETS[algo]
    target_stage, target_loc, target_type = spec[target_name]
    assert target_loc == 'node'
    is_output_target = (target_stage == 'output')

    rng = np.random.RandomState(SEED)
    train_sampler, _v, _vn, _tb, _tc, spec_list = clrs_run.create_samplers(
        rng=rng, train_lengths=[TRAIN_N], algorithms=[algo], val_lengths=[TRAIN_N],
        test_lengths=[TRAIN_N], train_batch_size=BATCH, val_batch_size=BATCH, test_batch_size=BATCH)
    sampler = train_sampler[0]

    fb0 = next(sampler)
    nb_dims = {}
    for inp in fb0.features.inputs: nb_dims[inp.name] = inp.data.shape[-1]
    for hint in fb0.features.hints: nb_dims[hint.name] = hint.data.shape[-1]
    for outp in fb0.outputs: nb_dims[outp.name] = outp.data.shape[-1]

    def net_fn(features_list, repred, algorithm_index, return_hints, return_all_outputs):
        return CaptureNet([spec], HIDDEN, True, True, get_capture_processor_factory(True, 8),
                          False, 'xavier_on_scalars', 0.0, 0.0, 'soft', [nb_dims], 1, False)(
            features_list, repred, algorithm_index, return_hints, return_all_outputs)
    net = hk.transform(net_fn)
    params = net.init(jax.random.PRNGKey(0), [fb0.features], True, 0, True, False)
    ckpt = pickle.load(open(f'{CKPT_DIR}/{algo}/best.pkl', 'rb'))['params']
    if set(params) != set(ckpt):
        raise ValueError(f'param mismatch for {algo}: {set(params) ^ set(ckpt)}')
    params = hk.data_structures.merge(params, ckpt)

    # source-like input (mask_one), if this algorithm has one, for a bonus predicate
    source_field = next((n for n, (st, loc, t) in spec.items() if st == 'input' and loc == 'node' and t == 'mask_one'), None)

    rows_x, rows_y, rows_prev, rows_next, rows_src = [], [], [], [], []
    n_batches = -(-N_EPISODES // BATCH)
    for bi in range(n_batches):
        fb = next(sampler)
        names = [h.name for h in fb.features.hints]
        t_stage_data = (np.asarray(fb.features.hints[names.index(target_name)].data) if target_stage == 'hint'
                        else None)  # (T,B,N[,C])
        lengths = np.asarray(fb.features.lengths)
        src_data = None
        if source_field:
            iname = [i.name for i in fb.features.inputs]
            src_data = np.asarray(fb.features.inputs[iname.index(source_field)].data)  # (B,N)

        out_preds, hint_preds, hiddens, z, msgs = net.apply(
            params, jax.random.PRNGKey(bi + 1), [fb.features], True, 0, True, False)
        z, msgs, hid_all = np.asarray(z), np.asarray(msgs), np.asarray(hiddens)  # (T,B,N,dim)
        Tp, B, N, _ = z.shape
        for b in range(B):
            L = int(lengths[b])
            for t in range(min(Tp, L - 1)):
                for n in range(N):
                    rows_x.append(np.concatenate([z[t, b, n], msgs[t, b, n]]))
                    rows_y.append(hid_all[t, b, n])
                    if target_stage == 'hint':
                        rows_prev.append(t_stage_data[t, b, n])
                        rows_next.append(t_stage_data[t + 1, b, n])
                    else:
                        rows_prev.append(np.nan); rows_next.append(np.nan)
                    rows_src.append(1.0 if (src_data is not None and n == src_data[b].argmax()) else 0.0)
    X = np.array(rows_x, dtype=np.float32)
    Y = np.array(rows_y, dtype=np.float32)
    prev = np.array(rows_prev, dtype=object)
    nxt = np.array(rows_next, dtype=object)
    is_src = np.array(rows_src, dtype=np.float32)
    log(f'{algo}: {len(X)} rows extracted ({time.time()-t0:.1f}s)')

    # ---- transcoder ----
    # Cap dataset size: DFS-family algorithms (dfs, articulation_points, bridges,
    # scc) produce 10-20x more node-step rows per episode than bfs-like ones
    # (long backtracking traversals), which would blow up training time for no
    # real accuracy gain -- these are i.i.d. activation samples, not sequences,
    # so random subsampling is representative.
    MAX_TRAIN, MAX_VAL = 80000, 10000
    rng2 = np.random.RandomState(0)
    idx = rng2.permutation(len(X))
    n_val = min(max(2000, len(X) // 10), MAX_VAL)
    val_idx, tr_idx = idx[:n_val], idx[n_val:][:MAX_TRAIN]
    Xtr, Ytr, Xval, Yval = X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]
    log(f'{algo}: training on {len(Xtr)}/{len(X)} rows (capped), val {len(Xval)}')
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-6

    def init_params(key):
        k1, k2 = jax.random.split(key)
        return dict(w_enc=jax.random.normal(k1, (X.shape[1], F)) * (1 / np.sqrt(X.shape[1])),
                   b_enc=jnp.zeros(F), w_dec=jax.random.normal(k2, (F, HIDDEN)) * (1 / np.sqrt(F)),
                   b_dec=jnp.mean(Ytr, axis=0))

    def topk_feats(p, x):
        act = jax.nn.relu(x @ p['w_enc'] + p['b_enc'])
        _, topi = jax.lax.top_k(act, K)
        mask = jnp.zeros_like(act).at[jnp.arange(act.shape[0])[:, None], topi].set(1.0)
        return act * mask

    def forward(p, x):
        feats = topk_feats(p, x)
        return feats @ p['w_dec'] + p['b_dec'], feats

    def loss_fn(p, xb, yb):
        pred, _ = forward(p, xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def update(p, opt_state, xb, yb):
        grads = jax.grad(loss_fn)(p, xb, yb)
        updates, opt_state = opt.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state

    params_t = init_params(jax.random.PRNGKey(0))
    Xtr_n, Xval_n = (Xtr - mu_x) / sd_x, (Xval - mu_x) / sd_x
    Xtr_j, Ytr_j, Xval_j, Yval_j = jnp.array(Xtr_n), jnp.array(Ytr), jnp.array(Xval_n), jnp.array(Yval)
    BATCH_T, EPOCHS = 256, 300
    n_batches_t = max(1, len(Xtr_j) // BATCH_T)
    sched = optax.cosine_decay_schedule(5e-3, EPOCHS * n_batches_t)
    opt = optax.adam(sched)
    opt_state = opt.init(params_t)
    var_y = float(jnp.mean((Yval_j - jnp.mean(Yval_j, axis=0)) ** 2))
    best_fvu, best_params = 1e9, params_t
    key = jax.random.PRNGKey(1)
    for ep in range(EPOCHS):
        key, k = jax.random.split(key)
        perm = jax.random.permutation(k, len(Xtr_j))
        for bi in range(n_batches_t):
            b = perm[bi * BATCH_T:(bi + 1) * BATCH_T]
            params_t, opt_state = update(params_t, opt_state, Xtr_j[b], Ytr_j[b])
        if ep % 30 == 0 or ep == EPOCHS - 1:
            vpred, _ = forward(params_t, Xval_j)
            fvu = float(jnp.mean((vpred - Yval_j) ** 2)) / var_y
            if fvu < best_fvu: best_fvu, best_params = fvu, params_t
    params_t = best_params
    log(f'{algo}: transcoder FVU={best_fvu:.4f} ({time.time()-t0:.1f}s total)')

    # ---- feature labels (generic predicates from the target field's own before/after, by type) ----
    Xn = (X - mu_x) / sd_x
    pre = Xn @ np.array(params_t['w_enc']) + np.array(params_t['b_enc'])
    act = np.maximum(pre, 0)
    topi = np.argsort(-act, axis=1)[:, :K]
    mask = np.zeros_like(act); np.put_along_axis(mask, topi, 1.0, axis=1)
    feats = act * mask

    predicates = {'is_source': is_src == 1} if source_field else {}
    if target_stage == 'hint':
        if target_type in ('mask', 'mask_one'):
            p0, n0 = prev.astype(float), nxt.astype(float)
            predicates.update({
                'stays 0': (p0 == 0) & (n0 == 0), 'newly 1': (p0 == 0) & (n0 == 1),
                'stays 1': (p0 == 1) & (n0 == 1), 'newly 0': (p0 == 1) & (n0 == 0),
            })
        elif target_type == 'scalar':
            p0, n0 = prev.astype(float), nxt.astype(float)
            predicates.update({'increases': n0 > p0 + 1e-6, 'decreases': n0 < p0 - 1e-6,
                              'stays same': np.abs(n0 - p0) <= 1e-6})
        elif target_type == 'categorical':
            n0 = np.stack(nxt).argmax(-1)
            for c in range(np.stack(nxt).shape[-1]):
                predicates[f'becomes class {c}'] = n0 == c
    predicates.update({'not ' + k: ~v for k, v in list(predicates.items())})

    N_rows = feats.shape[0]
    rows = []
    for f in range(F):
        fires = feats[:, f] > 0
        n_fire = int(fires.sum())
        if n_fire < 20:
            rows.append(dict(feature=f, label='(rare)', precision=None, recall=None, f1=None, n_fire=n_fire))
            continue
        best = None
        for name, pred in predicates.items():
            tp = int((fires & pred).sum())
            if tp == 0: continue
            precision, recall = tp / n_fire, tp / max(1, int(pred.sum()))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if best is None or f1 > best[3]: best = (name, precision, recall, f1)
        if best is None:
            rows.append(dict(feature=f, label='(unlabeled)', precision=None, recall=None, f1=None, n_fire=n_fire))
        else:
            name, p, r, f1 = best
            rows.append(dict(feature=f, label=name, precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3), n_fire=n_fire))
    rows.sort(key=lambda r: -(r['f1'] or 0))

    import os
    os.makedirs(f'{OUT_DIR}/{algo}', exist_ok=True)
    np.savez(f'{OUT_DIR}/{algo}/transcoder.npz', w_enc=np.array(params_t['w_enc']), b_enc=np.array(params_t['b_enc']),
            w_dec=np.array(params_t['w_dec']), b_dec=np.array(params_t['b_dec']), mu_x=mu_x, sd_x=sd_x, F=F, K=K, fvu=best_fvu)
    json.dump({'rows': rows, 'total_sites': N_rows}, open(f'{OUT_DIR}/{algo}/feature_labels.json', 'w'))

    try:
        res = json.load(open(f'{RESULTS_DIR}/{algo}__mpnn.json'))
        val_f1, test_f1 = res.get('best_val_f1'), res.get('test_f1')
    except Exception:
        val_f1 = test_f1 = None

    dec_key = f'net/~_construct_encoders_decoders/algo_0_{target_name}_dec_linear'
    W_dec = ckpt[dec_key]
    meta = dict(algo=algo, target=target_name, target_stage=target_stage, target_type=target_type,
               is_output=is_output_target, source_field=source_field, hidden=HIDDEN, F=F, K=K,
               fvu=best_fvu, val_f1=val_f1, test_f1=test_f1, n_rows=N_rows,
               W_dec=W_dec['w'].tolist(), b_dec=W_dec['b'].tolist(), elapsed_s=time.time() - t0)
    json.dump(meta, open(f'{OUT_DIR}/{algo}/meta.json', 'w'))
    log(f'{algo}: DONE, FVU={best_fvu:.4f}, val_f1={val_f1}, test_f1={test_f1}, {meta["elapsed_s"]:.1f}s')
    return meta


if __name__ == '__main__':
    def log(msg):
        print(msg, flush=True)
    algos = sys.argv[1:] or list(TARGETS)
    all_meta = []
    for a in algos:
        try:
            all_meta.append(run_algo(a, log))
        except Exception as e:
            import traceback
            log(f'{a}: FAILED: {e}\n{traceback.format_exc()}')
    json.dump(all_meta, open('batch_summary.json', 'w'), indent=2)
    log('BATCH DONE')
