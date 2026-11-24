"""Independent numpy re-implementation of the transcoder (float64, straight from the .npz files) evaluated on the SAME rounded inputs
that clrs-transcoder-reference.json stores. Writes clrs-edge-crosscheck.json for test1_port.js to compare against."""
import json, sys, numpy as np
src, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
g = json.load(open(ref))['golden']; res = {}
for a in g:
    T = {k: v.astype(np.float64) for k, v in np.load(f'{src}/{a}/transcoder.npz').items()}; k = int(T['k'])
    rows = []
    for r in g[a]['edge']['rows']:
        s = np.array(r['s']); x = (s - T['s_mu']) / T['s_scale']; pre = (x - T['b_pre']) @ T['W_enc'] + T['b_enc']
        thr = np.partition(pre, -k)[-k]; act = np.where(pre >= thr, np.maximum(pre, 0), 0)
        rows.append(dict(active_idx=[int(i) for i in np.nonzero(act)[0]], active_val=[float(act[i]) for i in np.nonzero(act)[0]],
                         y_hat=((act @ T['W_dec'] + T['b_dec']) * T['y_scale'] + T['y_mu']).tolist(), pre_first8=pre[:8].tolist()))
    res[a] = rows
json.dump(res, open(out, 'w'))
# how much does rounding the inputs to 6 significant digits move the reference itself?
worst = 0
for a in g:
    T = {k: v.astype(np.float64) for k, v in np.load(f'{src}/{a}/transcoder.npz').items()}
    for r in g[a]['edge']['rows']:
        s = np.array(r['s']); pre = ((s - T['s_mu']) / T['s_scale'] - T['b_pre']) @ T['W_enc'] + T['b_enc']
        i = np.array(r['active_idx']); worst = max(worst, float(np.max(np.abs(pre[i] - np.array(r['active_val'])) / np.abs(np.array(r['active_val'])))))
print('reference vs numpy float64 on the stored (rounded) inputs: worst relative difference of active values', f'{worst:.2e}')
