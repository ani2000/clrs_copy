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

---

## Edge-site transcoder graph (the main page, `/`; the previous node-site version is kept at `/demo`)

Source of the transcoders: `clrs-transcoder-out/<algo>/transcoder.npz` (trained for the same `best.pkl` weights the tracer serves:
the 12 files in `checkpoints_v3` are byte-identical to the training run's). Converted by `clrs_export_edge_tc.py` to `edge_tc/`, which also
writes `edge_tc/clrs-edge-summary.json` from each `report.json` (the numbers in the page's note come from there). Only the edge site
(message input s_ij → message y_ij, D = 1280, k = 8) is used. The node-site transcoders are not used: their in-network retention is
0.13–0.49 for dfs, strongly_connected_components and floyd_warshall, and they collapse at n = 64 for several algorithms.

Engine: `clrs-edge-attrib.js`. Tests, all in `verification/clrs-edge/` (inputs: the 12 model bundles from `/models/<algo>.*`, 4 random
n = 16 graphs per algorithm from `/api/instance`, the reference file `clrs-transcoder-reference.json`):

| test | what it shows | result |
|---|---|---|
| `test1_port.js` (+ `test1_crosscheck.py`) | the JS transcoder gives the same active latents and values as an independent numpy float64 implementation on the reference inputs, for all 12 algorithms | identical indices, relative error 4e-13. Against the reference file's own stored outputs the difference is up to 1e-3, because its inputs are stored with 6 significant digits |
| `test2_quality.js` | the transcoder reconstructs the site model's real messages (FVU, sum‖y−ŷ‖² / sum‖y−mean y‖²) | 0.002–0.035 on all 12 (a transcoder that did not belong to this model would score about 1). Compared with the report's validation FVU it is 0.7–7× higher, because the tracer samples its own random graphs |
| `test4_fd.js` | (E0) an independent replay of the replacement model (transcoder output + error node, winners / σ / ReLU pattern frozen) reproduces the real hidden states and target logits; (FD) nudging one latent in the replay changes later pre-activations by exactly the edge weight / activation, and every target logit by exactly the total effect; (SUM) for every latent and target the sum of incoming edges plus the explicit bias equals its value | all 12 algorithms: E0 ≤ 1.4e-14, edge weights vs finite differences ≤ 5.5e-14 (about 15–40 thousand comparisons each), total effects ≤ 1.1e-13, sums ≤ 1e-13. The bias is computed from the constant terms, not chosen to make the sums work |
| `test6_regression.js` | the previous (node-site) graph, now served at `/demo`, is unchanged by this change | byte-identical nodes, edges, pruned graph and total effects for bfs, bellman_ford, dijkstra (n = 8, p = 0.4, seed = 42) |
| `test7_intervene.js` | the clamp panel runs for every algorithm and a factor of 1 changes nothing | 12/12 |
| `test8_replacement.js` | the transcoder works as planned INSIDE the network: the message MLP is replaced by the transcoder at every step (winners and hints recomputed), on the site's own graphs, and the predictions are compared with the unmodified network | all 12 algorithms: the network keeps 94–102% of its accuracy against the truth (the training report's retention is 97–100%; 3 graphs per algorithm here), prediction agreement 85–100% (dijkstra and mst_prim are the lowest) |
| `clrs_edge_latents.js` (+ `fetch_graphs.sh`) | measures what every latent does and writes `edge_tc/<algo>.latents.json` (see below) | 60 graphs per algorithm, 720 in total, no duplicates; deterministic (a second run gives identical tables); the sum test holds on every one of the ~85 thousand effect computations (worst 7e-14) |
| `test9_labels.js` | a second, simple implementation reproduces the stored precision / recall / F1 of the best named latents on the held-out graphs | all 12 algorithms, 40 latents checked: identical to 3 decimals |
| `report_interventions.js` | **not a gate**: how well does the graph's linear prediction of a clamp match the real model? | direction matched in 223 of 325 comparisons (69%), 44–77% depending on the algorithm. Holding the fed-back hints fixed makes no difference. The graph freezes the LayerNorm σ, the max-routing winners and the TopK sets, so it is a first-order guide only; the page says so next to every result |

Known limits (also stated in the page): the edge-site latents have no names (the hint predicates the node-site features use fit pair latents
poorly); the faithfulness of the replacement is measured at n = 16 for all algorithms and at n = 64 only for the seven tier-A ones
(bridges 0.54, articulation_points 0.89, dijkstra 0.90 at n = 64); with k = 8 the replacement changes the max-winner in 9–20% of channels when it
runs on its own (the graph uses the real winners, so it is exact for the real run regardless); cost: about 0.4 s at n = 16, 0.8 s at n = 24 and
1.5 s at n = 32 in the browser worker.


### What each latent means (`edge_tc/<algo>.latents.json`, shown in the inspector, the "Top latents at each step" panel and the dictionary)

For every latent that fires, over 60 fresh n = 16 graphs per algorithm (seeds 2000–2059 of the sampler; up to 20 evenly spaced steps per graph):

* how often it fires, with what activation, and on which kind of pair (real edge / non-edge / self pair) compared with the share of such pairs;
* its **effect**: the change of the receiver's main target logit per unit of the latent's activation (for categorical targets the logit of the predicted class), mean, standard deviation and the share of activations where it is positive. This is the graph's own edge weight into the logit divided by the activation, so it inherits the finite-difference and sum checks above;
* its best-matching **conditions** on the pair (for example "the sender is already reached", "the receiver turns 1 next step", "real edge from a reached sender to an unreached receiver") with precision, recall and F1. The conditions use the reference algorithm's true hints and the input graph. They are **chosen on the 30 even-seed graphs and scored on the 30 odd-seed graphs the choice never saw**; a latent is named only if its held-out F1 is at least 0.5 (tier B) or 0.8 (tier A);
* its three strongest activations as (seed, step, sender, receiver), which the page can load.

Result: only a small share of the latents can be named (per algorithm 3–18 of the 650–1,190 that fire; tier A 0–3), which matches the training report's warning that pair latents are poorly described by simple conditions. The rest are shown as "unnamed", with the facts above and their best held-out match, not a guessed name. Examples of named latents: in bfs, f64 fires exactly when the sender is the source node (held-out F1 1.00), f573 on real edges (0.88), f551 on edges from a reached to an unreached node, where it raises the receiver's reach logit in 99% of its activations.
