# Testing and verification

The project's rule is that every number the page shows must be real, and every claim about the method must be backed by a test that can fail. This document lists every check in the repository: what it proves, what it needs, how to run it, what it printed when last run, and which parts are *not* covered. A shorter report with the older results is `VERIFICATION.md` in the repository root.

## 1. The layers of checking

| layer | question | where |
|---|---|---|
| 1. provenance | are the checkpoints the ones the training run produced, and does each belong to its own algorithm? | `verification/verify_backend.py` |
| 2. the model in the browser | does the JavaScript forward pass equal the JAX model? | `verification/verify_js.js`, `verify_backend.py`, `integrity_test.js` |
| 3. the data | do the sampled graphs and the true hints equal the CLRS reference implementation? | `verify_backend.py`, `integrity_test.js` |
| 4. the models | does each model really execute its algorithm (scores on fresh graphs)? | `verify_scores.py` |
| 5. the transcoders | does the port equal an independent implementation, does it reconstruct this model's messages, does it work inside the network? | `verification/clrs-edge/test1`, `test2`, `test8` |
| 6. the attribution graph | are the edge weights, total effects and sums exact? | `test4_fd.js`, `fd_test.js`, `fd_targets.js` |
| 7. the meanings | do the stored precision/recall/F1 reproduce? | `test9_labels.js`, `verify_labels.py` |
| 8. the pseudo-code | is every highlighted line tied to a fact in the trace? | `verification/pseudo/*` |
| 9. the page | does everything displayed match a recomputation, in a real browser, at three widths? | `ui_truth_test.mjs`, `layout_test.mjs`, `verification/ui/*` |
| 10. the live site | do the same browser tests pass against the deployed site? | the same scripts with `URL=` |

## 2. How to get the inputs

Several tests need the model bundles, a set of graphs and the training outputs, none of which are inside the repository (the bundles are generated in the image).

| input | how to obtain it |
|---|---|
| model bundles `<algo>.json` / `.bin` | from a running site: `curl "http://98.86.163.27/models/<algo>.json?v=<build id>"` (and `.bin`); or run `python model_export.py <folder>` where the checkpoints and the CLRS library are installed |
| sampled graphs | `curl "http://98.86.163.27/api/instance?algo=<algo>&n=16&p=0.4&seed=<seed>&v=<build id>"` (gzip) into files `<algo>_<seed>.json`; `verification/clrs-edge/fetch_graphs.sh <folder>` fetches 60 per algorithm (seeds 2000 to 2059) |
| training reports | the folder `clrs-transcoder-out` of the transcoder run |
| `summary_v3.json` | the folder `results_v3` of the training run |
| the build id | `curl -s http://98.86.163.27/ | gzip -dc | grep -o "BUILD = '[0-9a-f]*'"` (the page is sent gzip-compressed to clients that accept it) |

The browser tests need Google Chrome. Against the live site, which is plain `http://`, Chrome must be started with `--disable-features=HttpsUpgrades,HttpsFirstBalancedModeAutoEnable,HttpsFirstModeV2ForEngagedSites` (the tests take it from the environment variable `CHROME_FLAGS`), otherwise Chrome blocks the address.

## 3. The edge-site tests (`verification/clrs-edge/`)

### Test 1: the port (`test1_port.js`, `test1_crosscheck.py`)
*Claim.* The JavaScript transcoder (`encodeRow`, `decodeRow`) gives the same active latents and the same values as an independent numpy float64 implementation.
*Method.* `test1_crosscheck.py` evaluates the transcoder from the `.npz` files in numpy on the 36 reference rows (3 per algorithm, 12 algorithms) and writes the result; `test1_port.js` evaluates the JavaScript port on the same rows and compares: indices must be identical and values equal to 1e-9 relative. It also compares with the stored outputs of `clrs-transcoder-reference.json` (tolerance 2e-3, see below).
*Result.* 36 rows, all active indices identical; worst relative difference to the independent numpy implementation 4.3e-13. Against the reference file's own stored outputs the worst difference is 1.0e-3: the reference stores its input rows with six significant digits but computed the outputs from the unrounded values, so agreement to better than about 1e-3 is impossible by construction, which is why the cross-check was added.
*Run.* `CROSS=<file> node verification/clrs-edge/test1_port.js` after `python3 verification/clrs-edge/test1_crosscheck.py <clrs-transcoder-out> <reference.json> <file>`.

### Test 2: reconstruction on this model (`test2_quality.js`)
*Claim.* The transcoders were trained on this very network: they reconstruct the real messages of the site's model.
*Method.* Runs the model on 4 graphs per algorithm (n = 16), computes FVU over all pair rows; gate: FVU below 0.05 (a transcoder from a different model would score about 1).
*Result.* FVU 0.002 to 0.035 on all twelve; relative to the validation FVU of the training reports the values are 0.7 to 7 times (the site's sampler produces different graphs than the CLRS validation set; the largest ratios are bridges and articulation points, at about 7).

### Tests 3, 4, 5: replay, finite differences, sums (`test4_fd.js`)
*Claims.* (E0) Replaying the replacement model (transcoder output plus error node, winners, sigma and ReLU pattern frozen) reproduces the real hidden states and target logits. (FD) Nudging one latent in the replay changes the pre-activations one step later by exactly "edge weight / activation" and every target logit by exactly the total effect. (SUM) For every latent and target logit, value = sum of incoming edges + bias.
*Method.* An independent replay of the whole replacement model, written in the test, with a large nudge (the replay is linear in the nudge, so size is irrelevant; 0.37 is used). Twenty random latents per algorithm; every edge from each into the next step is compared.
*Result (all twelve algorithms).* E0 difference at most 1.4e-14; edge weights vs finite differences at most 5.5e-14 (about 14,000 to 41,000 comparisons per algorithm); total effects at most 1.1e-13; sum test at most 1.0e-13 (about 4,000 to 6,100 nodes per run).

### Test 6: the previous graph is unchanged (`test6_regression.js`)
*Claim.* The node-site graph (now at `/demo`) is byte-identical to what it was before the edge-site work.
*Method.* Builds the node-site graph with the old and the new code (`mpnn.js` and `attrib.js` taken from an old commit via `git show <commit>:file` into a folder given as `OLD`) for bfs, bellman_ford and dijkstra (n = 8, p = 0.4, seed 42) and hashes nodes, edges, the pruned graph and the total effects.
*Result.* Identical for all three, after every change of the day.

### Test 7: interventions (`test7_intervene.js`)
*Claim.* For all twelve algorithms the clamp runs, gives finite numbers for every target, and a factor of 1 changes nothing.
*Result.* 12 of 12; for factor 1 the change is exactly 0.

### Test 8: the transcoder inside the real network (`test8_replacement.js`)
*Claim.* "Works as planned": with the message function replaced by the transcoder at every step (winners and hints recomputed), the network keeps its accuracy.
*Method.* Uses the `opts.site` hook of `mpnn.js`; compares predictions of the target with the unmodified network and with the truth on 3 graphs per algorithm (the page's sampler, n = 16).
*Result.* Accuracy kept 94% to 102% across the twelve algorithms (training reports: 97% to 100%); prediction agreement with the unmodified network 85% to 100%, lowest for dijkstra (86%) and mst_prim (85%); gate for tier-A algorithms: agreement at least 80% and accuracy kept at least 80%.

### Test 9: the label scores (`test9_labels.js`)
*Claim.* The stored held-out precision, recall and F1 of the named latents can be reproduced by a second, simple implementation.
*Method.* For the four best-named latents of an algorithm, re-runs the model on the held-out (odd-seed) graphs, applies the stored best condition with brute-force code written separately from `clrs_edge_latents.js`, and compares P, R, F1 (tolerance 2e-3) and the number of held-out rows.
*Result.* 40 latents over all twelve algorithms: all identical to three decimals, row counts equal. Two full runs of the measurement gave identical tables.

### Intervention report (`report_interventions.js`)
Not a gate. How often does the graph's predicted direction match the real model after ablating the most influential latents? Result: 223 of 325 comparisons (69%); by algorithm bfs 70%, dfs 44%, bellman_ford 77%, dijkstra 68%, dag_shortest_paths 71%, mst_prim 72%, mst_kruskal 44%, topological_sort 75%, articulation_points 75%, bridges 75%, strongly_connected_components 75%, floyd_warshall 75%. Holding the fed-back hints fixed makes no difference (so hint feedback is not the cause); the graph freezes sigma, winners and TopK sets, so it is only a first-order guide.

### The latent measurement (`clrs_edge_latents.js`, `fetch_graphs.sh`)
Produces `edge_tc/<algo>.latents.json` (see `07-data-files-and-formats.md`). 60 graphs per algorithm (seeds 2000 to 2059, n = 16, p = 0.4; the sampler sometimes answers a request with a graph of a shifted seed; all 720 files are valid and no two graphs of an algorithm are identical). Conditions are chosen on the 30 even-seed graphs and scored on the 30 odd-seed graphs. Effects are measured through the engine's own edges, and each effect computation also runs the sum test (worst difference over about 85,000 computations: 7e-14).

### Truth test of the page (`ui_truth_test.mjs`)
*Claim.* Every number the page shows is real.
*Method.* Drives Chrome on the page, pins the most influential latent of the explained step, clamps it, reads the numbers from the page (top latents, per-node predictions, model facts, dictionary, inspector with conditions, examples, total effect and edge lists, message map, clamp result) and recomputes each in Node from the same graph, model bundle and transcoder, and from the original files (`report.json` for the model facts, `summary_v3.json` for the scores, the latent tables for meanings).
*Result.* Locally 2,889 numbers over bfs, dijkstra, dfs, mst_kruskal, floyd_warshall and bridges: all equal. On the live site 1,964 numbers over bfs, dijkstra, dfs and mst_kruskal: all equal. Sensitivity: when the scores file is altered (bfs test score set to 0.5) the test fails on exactly that number.
*Run.* `URL=<page> MODELS=<bundles> REPORTS=<clrs-transcoder-out> SUMMARY=<summary_v3.json> ALGOS=bfs,dijkstra node verification/clrs-edge/ui_truth_test.mjs`.

### Layout test (`layout_test.mjs`)
Seven collapsible sections: defaults (Top latents, Inspector, Intervention open; others closed), clicking opens and closes, state survives a reload (for sections and for the inspector's sub-sections), every header shows a live summary, the inspector's sub-sections are in order, no horizontal overflow at 1440, 1024 and 390 px. Passes locally and live.

## 4. Browser tests of the page (`verification/ui/`)

| file | what it checks |
|---|---|
| `latency_test.mjs` | not a pass/fail test: cold-load timeline under an emulated slow, far-away link (env `LAT` extra round trip in ms, `KBPS` download speed, `WATERFALL=1` prints every request). Run one process per measurement (a second run in the same browser process is not throttled) against `testsrv.mjs` |
| `testsrv.mjs` | not a test: a local server that serves this repository's page, scripts and `edge_tc/` and proxies models and the API to the live site (cached on disk). Start with `PORT=8792 REPO=<repo> node verification/ui/testsrv.mjs` |
| `edge_ui_test.mjs` | the main page for each algorithm: edge-site graph appears, the old pieces (toggle, label table, node-site downloads) are gone, the inspector shows a latent, a clamp returns a result, supernodes and By meaning views render, the example loader loads a measured graph, no console error or failed request. `PANELS_ALL=1` runs the full panel checks for every algorithm |
| `demo_ui_test.mjs` | `/demo` is still the previous version: node-site graph, labelled features, label table, no edge-site elements, link to the main page |
| `pc_test.mjs` | layout and the pseudo-code pane: simulation and pseudo-code side by side with the attribution graph below at full width (and its measured width after a click), one highlighted line per step for every algorithm, the page shows exactly what the engine says (lines, context bars, ring counts), tags are short, no squeezed code lines, node click shows a graph, Play advances, tablet and phone layouts without overflow |
| `squeeze_test.mjs` | the code lines stay readable even with absurdly long tags at five widths (this was a bug found in a screenshot) |

Latest results: all pass locally and against the live site (build `588d678899`), for all twelve algorithms where the script covers them.

## 5. Pseudo-code tests (`verification/pseudo/`)

| file | what it checks |
|---|---|
| `pseudo_block.js` | not a test: the source of the engine. The same code is inlined in `viewer3.html` |
| `check_inline.js` | fails if the inlined engine differs from `pseudo_block.js` |
| `test_pseudo.js` | structure: for the 36 recorded traces (3 graphs per algorithm) every step has at least one current line, all line numbers exist, context lines do not overlap current lines, tags refer to existing lines |
| `truth_pseudo.js` | ground truth: SCC components equal the transitive-closure components, topological orders respect every edge, final distances equal true shortest paths, Kruskal's tree weight equals the minimum spanning tree weight, Prim's edges are a subset of a minimum spanning tree, bridges are found correctly |
| `test_visual.js` | 2,928 checks tying every highlighted line to a fact in the trace itself (a vertex turning gray, a value changing, a node newly reached, the head of the topological list changing, a new cut vertex, the right k for each Floyd-Warshall row, every vertex whose value changed is ringed); a deliberately broken engine fails 102 of them |
| `inst/` | the 36 recorded instances |

## 6. The earlier checks (`verification/*.py`, `*.js`, report of 2026-09-28)

| check | what it showed |
|---|---|
| provenance (`verify_backend.py`) | the repository's twelve checkpoints are byte-identical (sha256) to the originals; none equals the unrelated 30-algorithm checkpoint set; every checkpoint's parameter names equal its algorithm's CLRS spec; each fits only its own algorithm |
| bundles (`verify_backend.py`, `verify_js.js`) | the browser bundles equal a fresh export; the JavaScript forward pass equals the JAX model on 24 graphs, max difference 1.3e-4 (float32 rounding); live in the page the browser activations equal the server's JAX run (max difference at most 6.9e-4 for the longest trajectory, at most 4e-5 for the others) |
| data (`verify_backend.py`, `integrity_test.js`) | instances contain exactly the algorithm's fields; the shown true trajectory and the step-0 hints equal CLRS's reference implementation re-run on the instance's inputs (3 graphs per algorithm) |
| scores (`verify_scores.py`) | on fresh graphs with CLRS's own metric each model reproduces its recorded signature: the seven CLEAN models stay accurate at 64 nodes, the five SEVERE collapse by the recorded amounts (the largest validation deviation: mst_kruskal, +0.09) |
| exactness of the node-site graph (`fd_test.js`, `fd_targets.js`) | 720 random edges over all twelve algorithms agree with finite differences to about 1e-8; 97 leaf-to-target edges with no mismatch; interventions move targets in the predicted direction, matching to three decimals when no downstream decision flips |
| labels (`verify_labels.py`) | precision/recall/F1 of the top ten labelled node-site features, recomputed on fresh graphs: mean absolute change of F1 0.002 to 0.028, worst 0.058 (dijkstra) |
| integrity (`integrity_test.js`, 98 checks) | for every algorithm everything shown (manifest, dropdown, weights, transcoder, labels, targets, leaves, truth column, playback) belongs to it; a rapid switch dfs, bfs, dfs, floyd ends with everything on dfs |

Bugs found and fixed by those checks: stale predictions or dots of the previous algorithm shown after switching (all traced results are now tied to the exact algorithm and graph and dropped when either changes); wrong hover texts (training graph sizes, pruning rule, intervention description); transcoder-approximated predictions shown as if real (now greyed and labelled, later removed from the main page altogether).

## 7. Checks on the deployed site and the infrastructure

| check | where | what |
|---|---|---|
| `pre-demo-check` workflow (`tools/predemo_check.mjs`) | GitHub runner | loads all 12 algorithms from outside with timings |
| `cross-browser-check` workflow (`tools/browsers_check.mjs`) | GitHub runner | Chromium, Firefox, WebKit incl. phone size, five algorithms each |
| `ec2-diagnose` workflow | the instance | state, credits, memory, logs |
| smoke test inside `build-and-push-ecr` | GitHub runner | both servers answer 200 before the image is pushed |

## 8. What is not covered

- **Real devices and slow networks.** Phone browsers and long-distance connections were exercised only through emulated sizes and through this Mac's connection (which has intermittent stalls of several seconds on some requests).
- **Other browsers for the new panels.** The cross-browser workflow covers the page in Firefox and WebKit but not every new panel; the detailed tests run in Chrome.
- **Algorithms at n above 16.** Latent meanings and transcoder fidelity are measured at n = 16 (and, for tier A, 64 for the fidelity only). Graph builds at n = 24 and 32 were timed (0.8 s and 1.5 s) and the sums verified, but meanings were not measured there.
- **Latent meanings beyond simple conditions.** Latents that follow a more complex rule remain unnamed by design.
- **The accuracy of clamp predictions** is only 69% in direction; this is reported, not fixed.
- **Load.** No load test was run against the live site, by choice.
- **Documentation numbers** in the hand-written documents were checked against files when written; the generated documents (`04`, `06b`, `15`, `16`, `17`) are regenerated from the files by `tools/clrs_docs_generate.py`.

## 9. Running everything locally (a checklist)

1. Obtain the inputs (section 2).
2. `node verification/clrs-edge/test1_port.js` (with `CROSS`), `test2_quality.js`, `test4_fd.js`, `test7_intervene.js`, `test8_replacement.js`, `test9_labels.js` with `MODELS` and `INST` set.
3. `OLD=<folder with old mpnn.js, attrib.js> MODELS=... REG=<folder with w_<algo>.json, i_<algo>.json> node verification/clrs-edge/test6_regression.js`.
4. `node verification/pseudo/check_inline.js`, `test_pseudo.js`, `truth_pseudo.js`, `test_visual.js`.
5. Start `testsrv.mjs`, then `edge_ui_test.mjs`, `demo_ui_test.mjs`, `pc_test.mjs`, `squeeze_test.mjs`, `layout_test.mjs`, `ui_truth_test.mjs`.
6. Against the live site: the same with `URL=http://98.86.163.27/...` and `CHROME_FLAGS` as in section 2.
