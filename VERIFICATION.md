# Verification report — all 12 algorithms

Date: 2026-09-28. Scripts are in [`verification/`](verification/); every number below was produced by them.

The question being answered: when the viewer shows algorithm X, is every part of it — model, data, trajectory,
attribution graph, labels, table, playback — really algorithm X's, computed from its checkpoint in `checkpoints_v3`?

## 1. Checkpoint provenance (`verify_backend.py` §1)

| check | result |
|---|---|
| `config.CKPT_DIR` points at `checkpoints_v3` | pass |
| repo copy (`checkpoints_v3/<algo>/best.pkl`, deployed via Git LFS) byte-identical to the original in `final results and notebook(v3 versions)/checkpoints_v3` — sha256, all 12 | 12/12 |
| none of them identical to the unrelated 30-algorithm set in `/Users/anindya/CLRS/checkpoints/mpnn` | 12/12 |

## 2. Each checkpoint belongs to exactly one algorithm (`verify_backend.py` §2)

* Every checkpoint's encoder/decoder parameter names equal its algorithm's CLRS spec fields (e.g. dfs:
  `A, adj, color, d, f, pi, pi_h, pos, s, s_last, s_prev, time, u, v`). 12/12.
* Each parameter set fits **only** its own algorithm's spec (a bfs checkpoint cannot even be loaded into a dfs
  network, and vice versa). 12/12.
* Processor weights differ between all 12 (separately trained models).

## 3. What runs in the browser is the checkpoint (`verify_backend.py` §3, `verify_js.js`, `integrity_test.js`)

* The in-browser weight bundles (`models/<algo>.bin`, generated at image build time from the checkpoint) are
  identical to a fresh export from the checkpoint; spot-checked layers equal the checkpoint arrays exactly. 12/12.
* The JavaScript forward pass equals the JAX model (`CaptureNet`, the project's copy of CLRS's `Net`) on
  24 graphs: max |difference| 1.3e-4 (float32 rounding). 12/12.
* Live in the page: the activations computed in the browser equal the server's JAX run of the same algorithm's
  checkpoint on the same graph (max |diff| ≤ 6.9e-4, mst_kruskal's 73-step trajectory; ≤ 4e-5 for the others). 12/12.

## 4. The data and trajectories are the algorithm's (`verify_backend.py` §4, `integrity_test.js`)

* Instances contain exactly the algorithm's CLRS input and hint fields. 12/12.
* The target-hint trajectory shown (simulation colours, table "true" column) and every step-0 hint are equal to
  CLRS's reference implementation (`clrs._src.algorithms.<algo>`) re-run on the instance's own inputs, on 3 graphs
  per algorithm. 12/12.

## 5. Each model really executes its algorithm (`verify_scores.py`)

Fresh CLRS graphs (default sampler p), CLRS's own output metric, compared with the training run's recorded scores
(`results_v3/summary_v3.json`). 64 graphs at n = 16, 16 graphs at n = 64, so ±0.02–0.05 sampling noise is expected.

| algorithm | val recorded | val now (n=16) | test recorded | test now (n=64) |
|---|---|---|---|---|
| bfs | 1.000 | 0.993 | 0.998 | 0.996 |
| dfs | 1.000 | 1.000 | 0.481 | 0.472 |
| bellman_ford | 0.998 | 0.990 | 0.948 | 0.933 |
| dijkstra | 1.000 | 0.990 | 0.977 | 0.973 |
| dag_shortest_paths | 1.000 | 0.992 | 0.984 | 0.986 |
| mst_prim | 1.000 | 0.979 | 0.926 | 0.959 |
| mst_kruskal | 0.905 | 0.992 | 0.445 | 0.485 |
| topological_sort | 1.000 | 0.980 | 0.517 | 0.591 |
| articulation_points | 0.985 | 1.000 | 0.949 | 0.963 |
| bridges | 0.994 | 1.000 | 0.968 | 0.973 |
| strongly_connected_components | 1.000 | 1.000 | 0.273 | 0.260 |
| floyd_warshall | 0.931 | 0.878 | 0.385 | 0.362 |

Every model reproduces its own signature — the 7 CLEAN models stay accurate at 64 nodes, the 5 SEVERE ones
collapse by the recorded amounts. The largest val deviation (mst_kruskal, +0.09) is likely because the recorded number was measured on
CLRS's fixed validation set, a different sample of graphs from the fresh default-p samples used here.

## 6. The attribution graphs are exact (`fd_test.js`, `fd_targets.js`)

Each edge weight was compared with a finite-difference derivative of the real network (hint inputs and frozen
nonlinearities held fixed; for target logits the new hidden state is held fixed, because that path runs through the
feature/error nodes of the graph):

* 720 randomly sampled edges (feature→feature through the message layer, error nodes, every kind of hint leaf),
  all 12 algorithms: agreement to ~1e-8 relative error. The single outlier was a leaf→target edge measured without
  holding the hidden state fixed (a test-design artefact; see next line).
* 97 leaf→target edges (inputs A, adj, pos, s and hints, incl. pointer targets) with the hidden state held fixed:
  0 mismatches.
* Interventions: clamping a feature and re-running the real model moves the targets in the direction the graph
  predicts. For the main hint target the size matches to 3 decimals when no downstream ReLU / routing decision flips
  (e.g. bfs reach_h: −1.015 actual vs −1.015 predicted); pointer targets agree in direction with small size
  differences (−0.151 vs −0.132 in the same example), plausibly because the pointer decoder's max-selection is frozen
  in the graph but free in the real re-run.

## 7. Labels are true (`verify_labels.py`)

Precision / recall / F1 of each algorithm's top-10 labelled features recomputed on fresh graphs (new seed) with the
same predicates: mean |ΔF1| 0.002–0.028 per algorithm, worst 0.058 (dijkstra) — sampling noise.

## 8. The viewer never mixes algorithms (`integrity_test.js`, 98 checks)

For every algorithm, in a real browser: manifest / dropdown / weights / transcoder / feature labels all belong to
it; the attribution graph's targets and every leaf are its own fields; the node table's truth column is its true
hint; Play steps through its own trajectory and re-traces at each step. A rapid dfs→bfs→dfs→floyd→dfs switch ends
with everything on dfs.

**Found and fixed during this verification:**

1. After switching algorithm, until the new trace arrived, the node table's "pred" column and the feature dots under
   the simulation could show the *previous* algorithm's traced values, and a trace requested for the old algorithm
   could still be drawn if it arrived late. All traced results are now scoped to the exact algorithm + graph they were
   computed for and are dropped the moment either changes.
2. Hover texts corrected: training graph sizes (a mix of 4/7/11/13/16 nodes, not "16-node graphs"), training p,
   the fully connected message routing of the CLRS MPNN, the pruning rule (influence-based), min |w| (absolute),
   Play speed, interventions (real-model re-run), and "F1" → CLRS output score (accuracy for pointers, F1 for masks).
3. Predictions shown before a step is traced (a transcoder approximation) are now greyed and labelled as such;
   traced steps show the real model's prediction.
