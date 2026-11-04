# Prompt: understand the CLRS-30 Circuit Tracer, then recommend its AWS deployment

You are being handed a complete technical description of a research project, followed by a concrete deployment question at the end. Read the whole thing before answering — the deployment answer depends on details buried in the mechanism description (model size, compute pattern, memory footprint).

---

## Part 1 — What this project is

The **CLRS-30 Circuit Tracer** is a mechanistic-interpretability toolkit, built in the spirit of Anthropic's paper *"On the Biology of a Large Language Model"* (the attribution-graphs / transcoders methodology), but applied to small graph-neural-network models solving classic CS algorithms instead of to a large language model.

It reverse-engineers **how a trained neural network internally executes an algorithm** — not just whether it gets the right answer, but which internal features cause which prediction, and whether that causal story survives an actual intervention (not just a plausible-looking diagram).

**Repository (private):** https://github.com/hvts007/CLRS-circuit-tracer
**Author:** Anindya Mazumder (GitHub email: `mazumder@hvts.ai`)

## Part 2 — The 12 algorithms and the underlying model

CLRS-30 is DeepMind's benchmark suite of classic algorithms with step-by-step "hint" supervision. This project covers 12 of its graph algorithms: **BFS, DFS, Bellman-Ford, Dijkstra, DAG shortest paths, Prim's MST, Kruskal's MST, topological sort, articulation points, bridges, strongly connected components, Floyd-Warshall.**

For each one, there is a real, independently-trained **Message-Passing Neural Network (MPNN)** — 12 separate checkpoints, not one shared model, trained for 10,000 steps each with `hint_teacher_forcing=0.5`.

### What "128 hidden" actually means (a common point of confusion)

This is **not** a 128-layer network. An MPNN has **one single set of weights**, applied *repeatedly* — once per algorithm step (BFS step 1, step 2, step 3, ...), the way a `for` loop reuses one function body rather than unrolling into distinct layers. "128 hidden" means: **every node in the input graph carries a 128-number vector as its running state**, and that vector is what gets updated, over and over, by the same small neural network at every step. There is no "layer 47" to point at — bigger hidden size just means more numeric capacity per node to encode whatever the model needs to remember (e.g. "have I been reached," "what's my current best distance," "who's my parent"), smeared across those 128 numbers in an entangled way. Interpreting *what* those 128 numbers actually encode is the entire motivation for the rest of this project.

### What a "hint" is, and what `reach_h` specifically means

CLRS-30 doesn't only supply a final answer — it also supplies the **intermediate reasoning trace** a textbook implementation would produce, called **hints**. For BFS, `reach_h` is the hint "reachability so far": at algorithm step `t`, `reach_h[t][v] = 1` if node `v` has been reached from the source by step `t`, else `0`. It updates every step as the frontier expands. It is a *hint*, not the final output (BFS's actual output is `pi`, the predecessor pointer forming the BFS tree) — but it's the most legible, concrete, step-by-step signal available, which is why it was chosen as the primary explanation target for BFS. Every other algorithm has its own analogous hint used the same way (e.g. `d` = current distance estimate for Dijkstra/Bellman-Ford; `color` = white/gray/black DFS visitation state for DFS/bridges/SCC; `key` = current best edge weight for Prim's MST).

## Part 3 — The interpretability pipeline, stage by stage

1. **The real model.** As above — one 128-hidden MPNN per algorithm, actually trained, not a toy.

2. **The transcoder.** The model's raw 128-number hidden state is a black box: training smears meaning across many numbers at once (superposition), so no single number has a clean meaning. A **sparse transcoder** is trained per algorithm to reconstruct that same 128-number update using a **dictionary of 192 candidate "features"**, of which only **8 may be active (nonzero) at once** per node per step. That forced sparsity is what makes individual features interpretable — each of the 8 active ones has to carry real, specific meaning rather than being one arbitrary blend of everything. Reconstruction quality is measured as **fraction of variance unexplained (FVU)**; across the 12 algorithms this ranged from 0.0018 (bfs, excellent) to 0.077 (floyd_warshall, still under 8%).

3. **Feature labeling.** Each of the 192 features is checked against hand-written predicates over the real hint values (e.g. "does this feature fire exactly when a node just became reachable?"), scored by precision/recall/F1. This is how a raw feature index like `f176` earns a human label like `"newly 1"`.

4. **Attribution graphs.** For one specific real example — one sampled graph, one specific step, one specific node — the sparse top-8 feature selection is *frozen*, turning that one forward pass into an exact linear equation: *this specific prediction = a weighted sum of these specific active features, each of which is itself a weighted sum of its own upstream causes one step earlier*. Drawn as boxes and arrows, that equation **is** the attribution graph for that instance.

5. **Supernodes.** Individual features are numerous; features sharing the same predicate label are grouped into one named "supernode" (e.g. every feature labeled `"newly 1"` becomes one box called `frontier`) — this is what turns a 192-node tangle into a handful of human-readable boxes, following the same convention Anthropic's paper uses.

6. **Interventions — the actual causal test.** To check an attribution graph isn't just a plausible-looking guess, a supernode's real activation is clamped to `-2×` its value, and the **real replacement-model computation is re-run** — including letting the top-8 selection genuinely change to a different feature set if the perturbation is large enough (not a linear estimate of the effect). The result is scored `CONFIRMED` / `NOT_CONFIRMED` / `TRIVIAL` against what the graph predicted.

## Part 4 — Figure 4 vs. Figure 7, and how they relate

These names and their relationship are borrowed directly from the Anthropic paper:

- **Figure 4 — one example, illustrative.** Pick one real graph and one real prediction. Show the supernodes feeding into it at baseline, then show what happens to that *same* example when one supernode is clamped. This is a **qualitative hypothesis** — "here is *a* case where suppressing `unreached` dropped the reachability prediction from 100% to 98%, in the predicted direction." Useful for building intuition; dangerous to generalize from a single instance.
- **Figure 7 — the validation sweep.** The paper reruns the same single-example intervention across several feature groups to show the story survives multiple ablations of one prompt. This project went one step further: `phase2agg_*.json` aggregates real interventions across **15 independently-sampled graphs per algorithm**, not one — because a single example turned out to be badly misleading in practice (e.g. `bridges` appeared to confirm only 33% of the time on one example, but was actually 74% once measured properly across 15 graphs).

**The relationship, stated plainly: Figure 4 proposes a mechanism from one case; Figure 7 is where you discover whether that mechanism actually generalizes.** This distinction produced the project's central finding: real-world accuracy does *not* predict how confirmable a model's circuits are — `bfs`, the single most accurate model tested (99.8% OOD test accuracy), had the *lowest* Figure-7-style aggregate confirmation rate (55%) of any of the 12 algorithms, which a Figure-4-only view would never have revealed. Two real measurement bugs were also caught and fixed along the way via independent ground-truth cross-checks (documented in the repo's README), rather than being silently smoothed over.

## Part 5 — Tech stack (exact, verified versions)

- Python 3.11 (developed venv), verified compatible with 3.13 runtime too
- `jax==0.4.34`, `jaxlib==0.4.34` — **CPU-only usage throughout**; no GPU/CUDA anywhere in this project, including originally training the models
- `dm-haiku==0.0.12` (neural net layer library on top of JAX)
- `optax==0.2.2` (only used historically to train the transcoders; not needed at serving time)
- `numpy==2.4.6`, `matplotlib==3.11.2` (matplotlib only for offline PNG figure generation, not at serving time)
- DeepMind's `clrs` library (installed from source) — used only for its graph samplers (`clrs.build_sampler`) and reference algorithm implementations, to generate fresh random graphs and ground truth on demand
- **No web framework anywhere** — both backend servers are pure Python standard library (`http.server.BaseHTTPRequestHandler` / `ThreadingHTTPServer`), zero external HTTP dependencies
- Frontend: static HTML + vanilla JavaScript + inline SVG; no build step, no Node.js needed at runtime (Node.js was used only during development, for testing)

## Part 6 — Measured runtime footprint (not estimated)

- 12 trained model checkpoints (`checkpoints_v3/`): **65 MB total on disk** (~5 MB per algorithm; largest, `dag_shortest_paths`, is ~7.8 MB)
- Application code + generated transcoders/labels/figures (the GitHub repo itself): **23 MB**
- At process startup, `server_128.py` loads **all 12 models into memory simultaneously** (one JAX/Haiku model + parameters per algorithm) — this is the only meaningfully CPU-intensive moment; steady-state memory afterward is small (well under 1 GB), since these are tiny MPNNs (128-dim hidden state, a single message-passing "layer" reused recurrently, no attention, no transformer blocks of any kind)
- Per-request inference is one forward pass over a graph of at most ~32 nodes — sub-second on CPU, no batching, no benefit whatsoever from a GPU
- **Two server processes, two ports:**
  - `server_128.py` (port 8766) — dynamic: samples a fresh random graph per request, runs a real model forward pass, serves the interactive `viewer3.html` explorer
  - a plain static file server (port 8767) — serves `fig47/viewer_fig47.html` plus ~24 pre-generated PNG figures and JSON data files; zero compute at request time
- Expected traffic: low and intermittent — a single-user or small-team research/demo tool, not a public-internet production service

## Part 7 — What "deployment" means for this project

The goal is to have both servers reachable **at all times** (not spun up on demand), but at the **lowest possible steady-state AWS cost**, since actual compute demand per request is trivial, total working memory is small, and traffic is sparse. Container images should be pushed to and pulled from **Amazon ECR**.

---

## The question

Given everything above — CPU-only workload, tiny models with zero GPU benefit, well under 1 GB working memory, low/sparse traffic, two lightweight Python-stdlib HTTP servers, containerized and pulled from ECR:

1. What is the **single lowest-cost EC2 instance type** that comfortably runs this 24/7 without under-provisioning? (If a non-EC2 always-on compute option — e.g. Fargate, Lightsail, App Runner — would genuinely be cheaper or more appropriate for this exact workload, say so explicitly and justify it instead of defaulting to EC2.)
2. How should ECR fit into the deployment — one container image running both servers, or two separate images/tasks?
3. What is the realistic **expected monthly cost** at that size, running continuously in a standard US region (e.g. `us-east-1`)?
4. Any concrete gotchas specific to this exact workload — in particular, whether ARM/Graviton (`t4g`-class) instances are worth using here for the price/performance advantage, and whether the pinned JAX/jaxlib CPU wheels have any known ARM compatibility issues worth checking before committing to that path.

Give one specific, named instance type as the recommendation — not a range of options to pick between.
