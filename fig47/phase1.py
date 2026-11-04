"""Phase 1: distill per-algorithm supernodes (Fig 3/4 style) from the existing
labeled features, build the baseline attribution graph over them for one
Phase-0-clean example, and FREEZE predicted intervention breaks before Phase 2
ever runs (mirrors the paper's own discipline: "we choose our feature
labelings and supernode groupings prior to measuring perturbation results").
"""
import json, pickle, re, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, haiku as hk, clrs
from clrs.examples import run as clrs_run
from clrs._src import decoders
from capture_net import CaptureNet, get_capture_processor_factory
from graph_engine import build_attribution, to_supernodes
from config import CKPT_DIR, ALGOS  # single source of truth -- checkpoints_v3 only

clrs_run.FLAGS(['x'])
HIDDEN = 128
TRAIN_N = 16
SEED = 20260924

# Confirmed supernode mapping: raw predicate label (from feature_labels.json) -> supernode name.
# 'not X' predicates strip the negation and fall into the same bucket as X (disclosed
# simplification -- a feature labeled by negation is still telling us about that axis).
SUPERNODE_MAP = {
    'bfs': {'is_source': 'source', 'newly 1': 'frontier', 'stays 1': 'interior',
           'stays 0': 'unreached', 'newly 0': 'unreached'},
    'bellman_ford': {'is_source': 'source', 'increases': 'relax-candidate',
                     'decreases': 'relax-candidate', 'stays same': 'settled'},
    'dijkstra': {'is_source': 'source', 'increases': 'relax-candidate',
                'decreases': 'relax-candidate', 'stays same': 'settled'},
    'dag_shortest_paths': {'is_source': 'source', 'increases': 'relax-candidate',
                           'decreases': 'relax-candidate', 'stays same': 'settled'},
    'mst_prim': {'is_source': 'source', 'increases': 'frontier-edge',
                'decreases': 'frontier-edge', 'stays same': 'tree-interior'},
    'mst_kruskal': {'newly 1': 'frontier-edge', 'stays 1': 'tree-interior',
                    'stays 0': 'excluded-edge', 'newly 0': 'excluded-edge'},
    'topological_sort': {'newly 1': 'ready-to-emit', 'stays 1': 'emitted', 'stays 0': 'blocked'},
    # DFS-family (severe): color transitions map onto the white/gray/black state machine
    'dfs': {'becomes class 0': 'white(unvisited)', 'becomes class 1': 'gray(in-progress)',
           'becomes class 2': 'black(done)'},
    'articulation_points': {'is_source': 'source', 'newly 1': 'newly-flagged-cut',
                            'stays 1': 'confirmed-cut', 'stays 0': 'not-cut'},
    'bridges': {'becomes class 0': 'white(unvisited)', 'becomes class 1': 'gray(in-progress)',
               'becomes class 2': 'black(done)'},
    'strongly_connected_components': {'becomes class 0': 'white(unvisited)',
                                      'becomes class 1': 'gray(in-progress)', 'becomes class 2': 'black(done)'},
    'floyd_warshall': {'is_source': 'pivot'},  # no clean baseline (Phase 0); minimal mapping
}


def strip_negation(label):
    return label[4:] if label and label.startswith('not ') else label


def build_net(algo, meta, spec, seed=SEED):
    rng = np.random.RandomState(seed)
    train_sampler, _v, _vn, _tb, _tc, spec_list = clrs_run.create_samplers(
        rng=rng, train_lengths=[TRAIN_N], algorithms=[algo], val_lengths=[TRAIN_N],
        test_lengths=[TRAIN_N], train_batch_size=8, val_batch_size=8, test_batch_size=8)
    sampler = train_sampler[0]
    fb = next(sampler)
    nb_dims = {}
    for inp in fb.features.inputs: nb_dims[inp.name] = inp.data.shape[-1]
    for hint in fb.features.hints: nb_dims[hint.name] = hint.data.shape[-1]
    for outp in fb.outputs: nb_dims[outp.name] = outp.data.shape[-1]

    def net_fn(features_list, repred, algorithm_index, return_hints, return_all_outputs):
        return CaptureNet([spec], HIDDEN, True, True, get_capture_processor_factory(True, 8),
                          False, 'xavier_on_scalars', 0.0, 0.0, 'soft', [nb_dims], 1, False)(
            features_list, repred, algorithm_index, return_hints, return_all_outputs)
    net = hk.transform(net_fn)
    params = net.init(jax.random.PRNGKey(0), [fb.features], True, 0, True, False)
    ckpt = pickle.load(open(f'{CKPT_DIR}/{algo}/best.pkl', 'rb'))['params']
    assert set(params) == set(ckpt)
    params = hk.data_structures.merge(params, ckpt)
    return net, params, fb, sampler


def get_example(algo, net, params, fb, meta):
    out_preds, hint_preds, hiddens, z, msgs = net.apply(
        params, jax.random.PRNGKey(1), [fb.features], True, 0, True, False)
    z, msgs, hiddens = np.asarray(z)[:, 0], np.asarray(msgs)[:, 0], np.asarray(hiddens)[:, 0]
    node_fts = z[:, :, :HIDDEN]
    L = int(np.asarray(fb.features.lengths)[0])
    names_i = [i.name for i in fb.features.inputs]
    adj = np.asarray(fb.features.inputs[names_i.index('adj')].data)[0]
    source = None
    if meta['source_field']:
        s_in = np.asarray(fb.features.inputs[names_i.index(meta['source_field'])].data)[0]
        source = int(np.argmax(s_in))

    # Pick the most "interesting" (node, step): one where the target field
    # actually transitions, rather than a trivial always-true source case.
    best = None
    if meta['target_stage'] == 'hint':
        names_h = [h.name for h in fb.features.hints]
        tgt = np.asarray(fb.features.hints[names_h.index(meta['target'])].data)[:, 0]  # (T,N[,C])
        if meta['target_type'] == 'categorical':
            tgt = tgt.argmax(-1)
        for step in range(L - 1):
            for n in range(adj.shape[0]):
                prev, nxt = tgt[step, n], tgt[step + 1, n]
                changed = not np.array_equal(prev, nxt)
                if changed:
                    best = (step, n)
    step_pick, node_pick = best if best else (L - 2, source if source is not None else 0)

    G = dict(n=adj.shape[0], length=L, adj=adj, source=source,
            z=z[:L - 1], msgs=msgs[:L - 1], node_fts=node_fts[:L - 1],
            picked_step=step_pick, picked_node=node_pick)
    return G


def load_weights(algo, meta):
    t = dict(np.load(f'../algos/{algo}/transcoder.npz'))
    ck = pickle.load(open(f'{CKPT_DIR}/{algo}/best.pkl', 'rb'))['params']
    W_dec = ck[f'net/~_construct_encoders_decoders/algo_0_{meta["target"]}_dec_linear']
    return dict(w_enc=t['w_enc'], b_enc=t['b_enc'], w_dec=t['w_dec'], b_dec=t['b_dec'],
               mu_x=t['mu_x'], sd_x=t['sd_x'], F=int(t['F']), K=int(t['K']), hidden=HIDDEN,
               W_dec=W_dec['w'], b_dec_target=W_dec['b'], target=meta['target'])


def run_algo(algo):
    spec = clrs.SPECS[algo]
    meta = json.load(open(f'../algos/{algo}/meta.json'))
    fl = json.load(open(f'../algos/{algo}/feature_labels.json'))
    label_by_feat = {r['feature']: r for r in fl['rows']}
    W = load_weights(algo, meta)
    net = params = fb = sampler = G = None
    for tries in range(6):
        net, params, fb, sampler = build_net(algo, meta, spec, seed=SEED + tries * 1000)
        G = get_example(algo, net, params, fb, meta)
        if G['length'] >= 3:
            break
    if G['length'] < 3:
        return dict(algo=algo, error='degenerate example (length<3) after 6 seed retries')

    step = G['picked_step']
    node = G['picked_node']
    A = build_attribution(W, G, step, node)
    smap = SUPERNODE_MAP.get(algo, {})

    def label_by_feat_mapped(fid):
        r = label_by_feat.get(fid)
        if not r or not r.get('label') or r['label'] in ('(unlabeled)', '(rare)'):
            return None
        base = strip_negation(r['label'])
        sn = smap.get(base)
        return {'label': sn} if sn else None

    edges = A['edges']
    supernode_edges, rep_feature = to_supernodes(edges, A, {f: label_by_feat_mapped(f) for f in label_by_feat})
    # keep only edges into the target for the Fig-4-style summary + mean F1 per supernode
    into_target = [(s, d, w) for s, d, w in supernode_edges if d == A['target_name']]
    into_target.sort(key=lambda e: -abs(e[2]))

    mean_f1 = {}
    for sn in set(smap.values()):
        fs = [label_by_feat[f]['f1'] for f, r in label_by_feat.items()
             if r.get('label') and strip_negation(r['label']) in smap and smap[strip_negation(r['label'])] == sn
             and r.get('f1') is not None]
        if fs:
            mean_f1[sn] = float(np.mean(fs))

    # A clamp to FACTOR=-2x reverses AND amplifies an originally-positive
    # activation's contribution (delta ~ (FACTOR-1)*orig*gradient, and
    # FACTOR-1 < 0), so the predicted logit change under THIS specific
    # intervention is the OPPOSITE sign of the raw edge weight into target.
    predictions = {sn: ('decrease' if w > 0 else 'increase') for sn, _, w in into_target if sn != 'bias'}

    result = dict(algo=algo, target=A['target_name'], baseline_logit=A['logit'],
                 step=step, node=node, source=G['source'],
                 supernode_edges_into_target=[(s, round(w, 4)) for s, _, w in into_target],
                 mean_f1_per_supernode=mean_f1, n_edges_all=len(edges),
                 n_edges_supernode=len(supernode_edges))
    json.dump(result, open(f'fig4_baseline_{algo}.json', 'w'), indent=2)
    json.dump({'predictions': predictions, 'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S')},
              open(f'predictions_{algo}.json', 'w'), indent=2)
    print(f"{algo:32s} target={A['target_name']:<28s} supernodes->target: " +
         ", ".join(f"{s}({w:+.2f})" for s, w in result['supernode_edges_into_target'][:5]))
    return result


if __name__ == '__main__':
    import sys
    algos = sys.argv[1:] or ALGOS
    all_results = {}
    for a in algos:
        try:
            all_results[a] = run_algo(a)
        except Exception as e:
            import traceback
            print(f'{a}: FAILED: {e}')
            traceback.print_exc()
    json.dump(all_results, open('phase1_summary.json', 'w'), indent=2, default=str)
    print('\nPhase 1 done.')
