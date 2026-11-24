# CLRS-30 Circuit Tracer

Circuit-tracing (attribution graphs, sparse transcoders, causal interventions) for MPNN models trained on all 12 CLRS-30 graph algorithms, in the spirit of Anthropic's ["On the Biology of a Large Language Model"](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

Every checkpoint studied here is a real, independently-trained 128-hidden MPNN (`hint_teacher_forcing=0.5`, `train_steps=10000`, seed 42) — not a toy example. Every number in the viewers and generated figures traces to a saved JSON file produced by an actual forward pass or intervention recompute; nothing is illustrative or fabricated.

## Algorithms covered

bfs, dfs, bellman_ford, dijkstra, dag_shortest_paths, mst_prim, mst_kruskal, topological_sort, articulation_points, bridges, strongly_connected_components, floyd_warshall.

## Method

1. **Transcoder** (`algos/<algo>/transcoder.npz`): a top-k sparse dictionary (F=192, k=8) trained to reconstruct each MPNN's node-update function from `(z, msgs)` → `hidden`.
2. **Feature labeling** (`algos/<algo>/feature_labels.json`): each of the 192 sparse features labeled against hand-derived predicates over the algorithm's own hint fields (precision/recall/F1 reported).
3. **Attribution graphs** (two kinds, switchable in the page: node-site features with labels, or the newer edge-site latents on sender → receiver pairs; see `VERIFICATION.md`): linearized, per-example causal graphs from sparse features through to a target prediction, built by freezing the top-k selection mask for that specific forward pass.
4. **Interventions**: features (individually or grouped into "supernodes") clamped to a negative multiple of their activation, with the resulting logit change measured through the actual replacement-model recompute — including letting the top-k selection cascade to a different feature set, a real causal test, not a linear estimate.
5. **Aggregation**: every claim about confirmation rate is reported over 15 independently-sampled graphs per algorithm, not a single cherry-picked example (`fig47/phase2agg_<algo>.json`).

## Interactive viewers

Two local tools, no external dependencies beyond the checkpoints:

- **`server_128.py` + `viewer3.html`** (port 8766) — pick any of the 12 algorithms, sample a fresh random graph, click any node to see its live attribution graph, click any feature for its label/stats, try `×0`/`×-1` interventions. Includes a "Supernodes" view that collapses same-label features together.

  ```bash
  python server_128.py
  # open http://localhost:8766/viewer3.html
  ```

- **`fig47/`** — a from-scratch reproduction of the paper's own Figure 4 (distilled supernode diagram) and Figure 7 (aggregate intervention confirmation grid) for every algorithm, both as an interactive page (`viewer_fig47.html`, port 8767) and as static PNGs (`fig4_<algo>.png`, `fig7_<algo>.png`).

  ```bash
  cd fig47 && python -m http.server 8767
  # open http://localhost:8767/viewer_fig47.html
  ```

## Key findings

**Accuracy is not the same as circuit confirmability.** Real OOD test accuracy (`summary_v3.json`) splits the 12 algorithms cleanly into two classes — CLEAN (gap < 0.10: bfs, dag_shortest_paths, dijkstra, bridges, articulation_points, bellman_ford, mst_prim) and SEVERE (gap > 0.40: mst_kruskal, topological_sort, dfs, floyd_warshall, strongly_connected_components) — but this split does **not** predict how confirmable each algorithm's attribution-graph-derived supernodes are under intervention (CLEAN mean 75.4% vs SEVERE mean 71.9% downstream-confirmed, well within noise). `bfs`, the single most accurate model (99.8% test F1), has the *lowest* confirmation rate (55%) of any algorithm tested.

**`floyd_warshall` never learned a coherent circuit.** Its own predicted distance matrix violates the triangle inequality even at in-distribution graph sizes (201 genuine violations, verified against a ground-truth cross-check) — a real failure of the underlying model, not a tooling artifact. It's the one algorithm with no clean baseline to run interventions against.

**A within-category recurring-feature pattern was found and then correctly downgraded.** `bfs` and `topological_sort` initially showed a striking pattern — clamping one feature reliably promotes the same handful of "sibling" features of the same predicate category. Testing this on two further CLEAN algorithms (`dag_shortest_paths`, `mst_prim`) failed to replicate it (50%/67% recurrence vs. the 80% threshold), so the final, honest verdict is **WEAK / isolated** — not a general mechanism. See `fig47/final_finding.json`.

**One real measurement bug was caught and corrected mid-project**: the `bridges` algorithm's reported "17–34% accuracy" was an artifact of comparing against CLRS's `-1` "no edge" sentinel value; the real accuracy is 100% on every valid edge at every tier. A second bug (`'is source'` vs `'is_source'` key mismatch) had silently disabled the "source" supernode across 7 algorithms in every phase until caught via a ground-truth cross-check and fixed. Both are documented in-repo rather than silently corrected.

## Directory structure

```
algos/<algo>/           transcoder.npz, feature_labels.json, meta.json  (per algorithm)
capture_net.py          spec-generic MPNN wrapper exposing internal (z, msgs, hidden)
pipeline.py             Phase 2+3+4: activation extraction, transcoder training, labeling
server_128.py           backend for viewer3.html (runs the real jax forward pass on demand)
viewer3.html            interactive attribution-graph explorer, all 12 algorithms
fig47/
  phase0.py             three-layer ground-truth harness (precondition / stage-correctness / invariant checks)
  phase1.py             supernode distillation + frozen pre-registered predictions
  phase2_aggregate.py   15-example intervention sweep per algorithm
  phase2_6.py           diagnostics for the accuracy/confirmability inversion
  phase3_maskchanged.py mask_changed / recurring-feature analysis
  gen_single_example_data.py, graph_engine.py, paper_style.py
  render_fig4_paper.py, render_fig7_paper.py   static PNG generators
  viewer_fig47.html     interactive Figure 4/7 reproduction, all 12 algorithms
  final_finding.json    the final, reconciled verdict on the recurring-feature question
```

Trained model checkpoints (`checkpoints_v3/`) are not included in this repo — this covers the circuit-tracing analysis built on top of them.
