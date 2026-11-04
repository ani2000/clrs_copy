"""Does each checkpoint actually execute its own algorithm? Re-score every model with CLRS's own
evaluation (output metric) on fresh graphs at n=16 (validation size) and n=64 (OOD test size) and compare
with the training run's recorded best_val_f1 / test_f1 in results_v3/summary_v3.json."""
import json, os, os, sys, time
import numpy as np, jax, clrs
from clrs._src import decoders
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP); os.chdir(APP)
import server_128 as S
from config import ALGOS
SUM = json.load(open('/Users/anindya/CLRS/new run/final results and notebook(v3 versions)/results_v3/summary_v3.json'))
REC = {r['algo']: r for r in SUM['results']}
N_VAL, N_TEST = int(os.environ.get('NV', 64)), int(os.environ.get('NT', 16))

def score(algo, n, samples, seed):
    r = S.REG[algo]
    sampler, spec = clrs.build_sampler(algo, num_samples=samples, seed=seed, length=n)   # CLRS default p for this algorithm
    fb = sampler.next(samples)
    out = r['net'].apply(r['params'], jax.random.PRNGKey(0), [fb.features], True, 0, True, False)[0]
    pp = decoders.postprocess(spec, out, sinkhorn_temperature=0.1, sinkhorn_steps=50, hard=True)
    return clrs.evaluate(fb.outputs, pp)['score']

print(f"{'algorithm':30s} {'val(recorded)':>13s} {'val(now,n=16)':>13s} {'test(recorded)':>14s} {'test(now,n=64)':>14s}")
rows = []
ONLY = os.environ.get('ONLY'); OUTF = os.environ.get('OUTF', 'scores.json')
for a in (ONLY.split(',') if ONLY else ALGOS):
    t0 = time.time()
    v = score(a, 16, N_VAL, 1234)
    te = score(a, 64, N_TEST, 4321)
    rv, rt = REC[a]['best_val_f1'], REC[a]['test_f1']
    rows.append((a, rv, v, rt, te))
    print(f"{a:30s} {rv:13.3f} {v:13.3f} {rt:14.3f} {te:14.3f}   ({time.time()-t0:.0f}s)", flush=True)
json.dump(rows, open(OUTF, 'w'))
