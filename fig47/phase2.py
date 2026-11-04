"""Phase 2: intervention sweep. For each algorithm, for each supernode with a
frozen (Phase 1) predicted direction, clamp its active features to -2x at
step T-1 (the REAL test: does the effect propagate to the target at step T,
after a fresh top-k reselection?) and separately at step T (TRIVIAL: directly
feeds the target's own decode, guaranteed by construction, reported but not
counted as validation, per the paper's own footnote on intervention layers).
"""
import json, pickle, re
import numpy as np, jax, haiku as hk, clrs
from clrs.examples import run as clrs_run
from capture_net import CaptureNet, get_capture_processor_factory
from graph_engine import build_attribution, run_intervention_group
from phase1 import SUPERNODE_MAP, strip_negation, build_net, get_example, load_weights, SEED, TRAIN_N, HIDDEN, CKPT_DIR, ALGOS

clrs_run.FLAGS(['x'])
FACTOR = -2.0


def run_algo(algo):
    spec = clrs.SPECS[algo]
    meta = json.load(open(f'../algos/{algo}/meta.json'))
    fl = json.load(open(f'../algos/{algo}/feature_labels.json'))
    label_by_feat = {r['feature']: r for r in fl['rows']}
    preds = json.load(open(f'predictions_{algo}.json'))['predictions']
    baseline = json.load(open(f'fig4_baseline_{algo}.json'))
    W = load_weights(algo, meta)
    smap = SUPERNODE_MAP.get(algo, {})

    net = params = fb = G = None
    for tries in range(6):
        net, params, fb, sampler = build_net(algo, meta, spec, seed=SEED + tries * 1000)
        G = get_example(algo, net, params, fb, meta)
        if G['length'] >= 3:
            break
    step, node = G['picked_step'], G['picked_node']
    A = build_attribution(W, G, step, node)
    assert A['target_name'] == baseline['target'], f'{algo}: example mismatch vs Phase 1 (non-determinism?)'

    # which raw features belong to each supernode
    sn_features = {}
    for f, r in label_by_feat.items():
        if not r.get('label') or r['label'] in ('(unlabeled)', '(rare)'):
            continue
        sn = smap.get(strip_negation(r['label']))
        if sn:
            sn_features.setdefault(sn, []).append(f)

    panels = []
    for sn, predicted in preds.items():
        if sn == 'bias' or sn not in sn_features:
            continue
        feats = sn_features[sn]
        for clamp_step, trivial in [(step - 1, False), (step, True)]:
            if clamp_step < 0:
                continue
            res = run_intervention_group(W, G, A, feats, clamp_step, node, FACTOR)
            if res is None:
                continue  # no active member of this supernode at this step
            actual = 'increase' if res['delta'] > 0 else 'decrease'
            verdict = 'TRIVIAL' if trivial else ('CONFIRMED' if actual == predicted else 'NOT_CONFIRMED')
            panels.append(dict(supernode=sn, predicted=predicted, actual=actual, verdict=verdict,
                               clamp_step=clamp_step, trivial=trivial, **{k: v for k, v in res.items() if k != 'orig_activations'}))
            print(f"{algo:24s} {sn:20s} step={clamp_step:<3d} {'(trivial)' if trivial else '(downstream)':13s} "
                 f"pred={predicted:<8s} actual={actual:<8s} {verdict:<13s} delta={res['delta']:+.3f} mask_changed={res['mask_changed']}")

    out = dict(algo=algo, target=A['target_name'], baseline_logit=A['logit'], panels=panels)
    json.dump(out, open(f'phase2_{algo}.json', 'w'), indent=2, default=str)
    return out


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
    json.dump(all_results, open('phase2_summary.json', 'w'), indent=2, default=str)
    print('\\nPhase 2 done.')
