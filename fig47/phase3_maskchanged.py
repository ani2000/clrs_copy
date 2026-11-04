"""Step 3 (GROUNDTRUTH_AND_PHASE4_PROMPT.md): resolve the mask_changed
question. Re-derives the SAME 15 examples per algorithm (identical
deterministic seeds as phase2_aggregate.py / phase2_6.py -- verified
reproducible twice already), this time also capturing mask_changed and the
active feature IDs before/after each clamp, which were computed transiently
by run_intervention_group but never persisted.
"""
import json
import numpy as np, jax, clrs
from graph_engine import build_attribution, to_supernodes, reconstruct, encode
from phase1 import SUPERNODE_MAP, strip_negation, load_weights, HIDDEN, ALGOS
from phase2_aggregate import build_net_and_sampler, examples_from_batch, N_EXAMPLES, FACTOR, BASE_SEED


def run_one_example_with_mask(W, G, A, sn_features, predictions):
    step, node = A['step'], A['node']
    panels = []
    base_feats_cur, base_feats_prev = A['feats_cur'], A['feats_prev']
    for sn, predicted in predictions.items():
        feats = sn_features.get(sn, [])
        if not feats:
            continue
        for clamp_step, trivial in [(step - 1, False), (step, True)]:
            if clamp_step < 0:
                continue
            src = base_feats_cur if clamp_step == step else base_feats_prev
            active_before = sorted([int(f) for f in feats if src[f] > 0])
            if not active_before:
                continue
            H = W['hidden']
            z, msgs, node_fts = G['z'][step][node], G['msgs'][step][node], G['node_fts'][step][node]
            orig_vals = {f: float(src[f]) for f in active_before}

            def recompute(f_prev, f_cur):
                hidden_in = reconstruct(W, f_prev) if step > 0 else np.zeros(H)
                nxt = reconstruct(W, f_cur)
                h_t = np.concatenate([node_fts, hidden_in, nxt])
                return float(h_t @ A['decode_vec'] + A['decode_bias'])

            if clamp_step == step:
                nf = base_feats_cur.copy()
                for f, v in orig_vals.items(): nf[f] = FACTOR * v
                new_logit = recompute(base_feats_prev, nf)
                active_after = sorted([int(f) for f in range(W['F']) if nf[f] > 0])
                mask_changed = active_after != sorted([int(f) for f in range(W['F']) if base_feats_cur[f] > 0])
            else:
                nf1 = base_feats_prev.copy()
                for f, v in orig_vals.items(): nf1[f] = FACTOR * v
                hidden_in_alt = reconstruct(W, nf1)
                z_alt = np.concatenate([node_fts, hidden_in_alt])
                new_feats_cur = encode(W, np.concatenate([z_alt, msgs]))
                new_logit = recompute(nf1, new_feats_cur)
                active_after = sorted([int(f) for f in range(W['F']) if new_feats_cur[f] > 0])
                mask_changed = active_after != sorted([int(f) for f in range(W['F']) if base_feats_cur[f] > 0])
                active_before_cur = sorted([int(f) for f in range(W['F']) if base_feats_cur[f] > 0])

            delta = new_logit - A['logit']
            actual = 'increase' if delta > 0 else 'decrease'
            verdict = 'TRIVIAL' if trivial else ('CONFIRMED' if actual == predicted else 'NOT_CONFIRMED')
            # active_before_step / active_after_step always refer to the TARGET step's own
            # top-k membership (what actually feeds the decode), since that's what "a
            # different circuit taking over" would mean -- not the clamped step's own set.
            active_before_target_step = sorted([int(f) for f in range(W['F']) if base_feats_cur[f] > 0])
            panels.append(dict(supernode=sn, clamp_step=clamp_step, trivial=trivial, predicted=predicted,
                               actual=actual, verdict=verdict, delta=delta, mask_changed=bool(mask_changed),
                               active_before_target_step=active_before_target_step,
                               active_after_target_step=active_after if clamp_step != step else active_after,
                               clamped_features=active_before))
    return panels


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
        seed = BASE_SEED + (sum(ord(c) for c in algo) * 31) % 1000 + tries * 7919
        net, params, sampler, fb0 = build_net_and_sampler(algo, spec, seed)
        fb = fb0
        for batch_i in range(3):
            if len(examples) >= N_EXAMPLES:
                break
            batch_examples = examples_from_batch(algo, meta, fb, net, params, jax.random.PRNGKey(tries * 10 + batch_i))
            examples.extend(batch_examples)
            fb = next(sampler)
        tries += 1
    examples = examples[:N_EXAMPLES]

    def mapped_label(fid):
        r = label_by_feat.get(fid)
        if not r or not r.get('label') or r['label'] in ('(unlabeled)', '(rare)'):
            return None
        sn = smap.get(strip_negation(r['label']))
        return {'label': sn} if sn else None
    label_map = {f: mapped_label(f) for f in label_by_feat}

    all_examples = []
    for G in examples:
        step, node = G['picked_step'], G['picked_node']
        A = build_attribution(W, G, step, node)
        supernode_edges, _ = to_supernodes(A['edges'], A, label_map)
        into_target = [(s, w) for s, d, w in supernode_edges if d == A['target_name'] and s != 'bias']
        predictions = {sn: ('decrease' if w > 0 else 'increase') for sn, w in into_target}
        sn_features = {}
        for f, r in label_by_feat.items():
            lab = label_map.get(f)
            if lab:
                sn_features.setdefault(lab['label'], []).append(f)
        panels = run_one_example_with_mask(W, G, A, sn_features, predictions)
        all_examples.append(dict(target=A['target_name'], panels=panels))

    json.dump({'algo': algo, 'examples': all_examples}, open(f'phase3_maskchanged_{algo}.json', 'w'), indent=2, default=str)

    ds = [p for ex in all_examples for p in ex['panels'] if not p['trivial']]
    tab = {'CONFIRMED_maskchg': 0, 'CONFIRMED_same': 0, 'NOT_CONFIRMED_maskchg': 0, 'NOT_CONFIRMED_same': 0}
    for p in ds:
        key = f"{p['verdict']}_{'maskchg' if p['mask_changed'] else 'same'}"
        tab[key] += 1
    print(f"{algo:32s} n={len(ds):3d}  CONF+maskchg={tab['CONFIRMED_maskchg']:2d} CONF+same={tab['CONFIRMED_same']:2d}  "
         f"NOTCONF+maskchg={tab['NOT_CONFIRMED_maskchg']:2d} NOTCONF+same={tab['NOT_CONFIRMED_same']:2d}")
    return dict(algo=algo, table=tab, n=len(ds))


if __name__ == '__main__':
    import sys
    algos = sys.argv[1:] or ALGOS
    results = {}
    for a in algos:
        try:
            results[a] = run_algo(a)
        except Exception as e:
            import traceback
            print(f'{a}: FAILED: {e}')
            traceback.print_exc()
    json.dump(results, open('phase3_maskchanged_summary.json', 'w'), indent=2, default=str)
    print('\nStep 3 mask_changed collection done.')
