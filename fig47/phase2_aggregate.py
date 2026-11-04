"""Phase 2, aggregated: for each algorithm, sample N independent in-distribution
graphs, pick a transitioning (node, step) in each, FREEZE that example's own
predicted supernode breaks (pre-registration stays PER-EXAMPLE, not once per
algorithm), run the intervention sweep, and report the CONFIRMED fraction as a
percentage over the whole set -- the paper's own "70% of cases" style claim,
not an eyeballed 1-3 example anecdote.
"""
import json, pickle, time
import numpy as np, jax, haiku as hk, clrs
from clrs.examples import run as clrs_run
from capture_net import CaptureNet, get_capture_processor_factory
from graph_engine import build_attribution, to_supernodes, run_intervention_group
from phase1 import SUPERNODE_MAP, strip_negation, load_weights, HIDDEN, CKPT_DIR, ALGOS

clrs_run.FLAGS(['x'])
TRAIN_N = 16
N_EXAMPLES = 15
FACTOR = -2.0
BASE_SEED = 20260924


def build_net_and_sampler(algo, spec, seed):
    rng = np.random.RandomState(seed)
    train_sampler, _v, _vn, _tb, _tc, spec_list = clrs_run.create_samplers(
        rng=rng, train_lengths=[TRAIN_N], algorithms=[algo], val_lengths=[TRAIN_N],
        test_lengths=[TRAIN_N], train_batch_size=8, val_batch_size=8, test_batch_size=8)
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
    assert set(params) == set(ckpt)
    params = hk.data_structures.merge(params, ckpt)
    return net, params, sampler, fb0


def examples_from_batch(algo, meta, fb, net, params, rng_key):
    """Run one forward pass on the whole batch, return a list of per-example G
    dicts (one per batch index with length>=3), each with its own picked
    (node, step) transition."""
    out_preds, hint_preds, hiddens, z, msgs = net.apply(params, rng_key, [fb.features], True, 0, True, False)
    z, msgs, hiddens = np.asarray(z), np.asarray(msgs), np.asarray(hiddens)  # (T,B,N,dim)
    node_fts = z[:, :, :, :HIDDEN]
    names_i = [i.name for i in fb.features.inputs]
    adj_all = np.asarray(fb.features.inputs[names_i.index('adj')].data)
    lengths = np.asarray(fb.features.lengths)
    src_all = None
    if meta['source_field']:
        src_all = np.asarray(fb.features.inputs[names_i.index(meta['source_field'])].data).argmax(-1)

    tgt_all = None
    if meta['target_stage'] == 'hint':
        names_h = [h.name for h in fb.features.hints]
        tgt_all = np.asarray(fb.features.hints[names_h.index(meta['target'])].data)  # (T,B,N[,C])
        if meta['target_type'] == 'categorical':
            tgt_all = tgt_all.argmax(-1)

    B = adj_all.shape[0]
    out = []
    for b in range(B):
        L = int(lengths[b])
        if L < 3:
            continue
        adj = adj_all[b]
        source = int(src_all[b]) if src_all is not None else None
        best = None
        if tgt_all is not None:
            for step in range(L - 1):
                for n in range(adj.shape[0]):
                    prev, nxt = tgt_all[step, b, n], tgt_all[step + 1, b, n]
                    if not np.array_equal(prev, nxt):
                        best = (step, n)
        step_pick, node_pick = best if best else (L - 2, source if source is not None else 0)
        out.append(dict(n=adj.shape[0], length=L, adj=adj, source=source,
                        z=z[:L - 1, b], msgs=msgs[:L - 1, b], node_fts=node_fts[:L - 1, b],
                        picked_step=step_pick, picked_node=node_pick))
    return out


def run_one_example(algo, meta, label_by_feat, W, smap, G):
    step, node = G['picked_step'], G['picked_node']
    A = build_attribution(W, G, step, node)

    def mapped_label(fid):
        r = label_by_feat.get(fid)
        if not r or not r.get('label') or r['label'] in ('(unlabeled)', '(rare)'):
            return None
        sn = smap.get(strip_negation(r['label']))
        return {'label': sn} if sn else None

    supernode_edges, _ = to_supernodes(A['edges'], A, {f: mapped_label(f) for f in label_by_feat})
    into_target = [(s, w) for s, d, w in supernode_edges if d == A['target_name'] and s != 'bias']
    # PRE-REGISTRATION, per-example: freeze this example's predicted breaks
    # BEFORE running its own interventions (same discipline as Phase 1, just
    # re-done independently for every one of the N examples, not shared).
    predictions = {sn: ('decrease' if w > 0 else 'increase') for sn, w in into_target}

    sn_features = {}
    for f, r in label_by_feat.items():
        if not r.get('label') or r['label'] in ('(unlabeled)', '(rare)'):
            continue
        sn = smap.get(strip_negation(r['label']))
        if sn:
            sn_features.setdefault(sn, []).append(f)

    panels = []
    for sn, predicted in predictions.items():
        feats = sn_features.get(sn, [])
        if not feats:
            continue
        for clamp_step, trivial in [(step - 1, False), (step, True)]:
            if clamp_step < 0:
                continue
            res = run_intervention_group(W, G, A, feats, clamp_step, node, FACTOR)
            if res is None:
                continue
            actual = 'increase' if res['delta'] > 0 else 'decrease'
            verdict = 'TRIVIAL' if trivial else ('CONFIRMED' if actual == predicted else 'NOT_CONFIRMED')
            panels.append(dict(supernode=sn, predicted=predicted, actual=actual, verdict=verdict, trivial=trivial))
    return dict(target=A['target_name'], step=step, node=node, panels=panels)


def run_algo(algo):
    spec = clrs.SPECS[algo]
    meta = json.load(open(f'../algos/{algo}/meta.json'))
    fl = json.load(open(f'../algos/{algo}/feature_labels.json'))
    label_by_feat = {r['feature']: r for r in fl['rows']}
    smap = SUPERNODE_MAP.get(algo, {})
    W = load_weights(algo, meta)

    examples = []
    tries = 0
    while len(examples) < N_EXAMPLES and tries < 8:
        seed = BASE_SEED + (sum(ord(c) for c in algo) * 31) % 1000 + tries * 7919  # deterministic, not Python's randomized hash()
        net, params, sampler, fb0 = build_net_and_sampler(algo, spec, seed)
        fb = fb0
        for batch_i in range(3):  # a couple of batches per seed if needed
            if len(examples) >= N_EXAMPLES:
                break
            batch_examples = examples_from_batch(algo, meta, fb, net, params, jax.random.PRNGKey(tries * 10 + batch_i))
            examples.extend(batch_examples)
            fb = next(sampler)
        tries += 1
    examples = examples[:N_EXAMPLES]

    all_panels = []
    for ex in examples:
        try:
            res = run_one_example(algo, meta, label_by_feat, W, smap, ex)
            all_panels.append(res)
        except Exception as e:
            print(f'{algo}: example failed: {e}')

    downstream = [(ex_i, p) for ex_i, ex in enumerate(all_panels) for p in ex['panels'] if not p['trivial']]
    trivial = [(ex_i, p) for ex_i, ex in enumerate(all_panels) for p in ex['panels'] if p['trivial']]
    n_conf = sum(1 for _, p in downstream if p['verdict'] == 'CONFIRMED')
    n_tot = len(downstream)
    frac = n_conf / n_tot if n_tot else None
    # per-example: did THIS example have >=1 downstream test, and was it (majority) confirmed?
    per_example_confirmed = []
    for ex in all_panels:
        ds = [p for p in ex['panels'] if not p['trivial']]
        if ds:
            per_example_confirmed.append(sum(1 for p in ds if p['verdict'] == 'CONFIRMED') / len(ds) >= 0.5)
    n_ex_conf = sum(per_example_confirmed)
    n_ex_tot = len(per_example_confirmed)

    result = dict(algo=algo, n_examples=len(all_panels), n_downstream_tests=n_tot, n_downstream_confirmed=n_conf,
                 downstream_confirmed_frac=frac, n_trivial_tests=len(trivial),
                 n_examples_majority_confirmed=n_ex_conf, n_examples_with_tests=n_ex_tot,
                 example_confirmed_frac=n_ex_conf / n_ex_tot if n_ex_tot else None)
    json.dump(result, open(f'phase2agg_{algo}.json', 'w'), indent=2, default=str)
    print(f"{algo:32s} examples={result['n_examples']:2d}  downstream {n_conf}/{n_tot} "
         f"({frac*100:.0f}%)" if n_tot else f"{algo:32s} examples={result['n_examples']:2d}  no downstream tests" +
         (f"  |  per-example majority-confirmed: {n_ex_conf}/{n_ex_tot}" if n_ex_tot else ""))
    return result


if __name__ == '__main__':
    import sys
    algos = sys.argv[1:] or ALGOS
    all_results = {}
    t0 = time.time()
    for a in algos:
        try:
            all_results[a] = run_algo(a)
        except Exception as e:
            import traceback
            print(f'{a}: FAILED: {e}')
            traceback.print_exc()
    json.dump(all_results, open('phase2agg_summary.json', 'w'), indent=2, default=str)
    print(f'\nPhase 2 (aggregated) done in {time.time()-t0:.0f}s.')
