# The page, element by element

This is the reference for everything on the main page `http://98.86.163.27/` (and where `/demo` differs). It goes from top to bottom, says what each control does, what each drawing means, and which file produces it. For the words themselves see `02-glossary.md`; for the exact hover texts see `06b-hover-text-catalog.md`.

## 1. A worked example (bfs, seed 42)

Open the page. It loads bfs with n = 16, p = 0.4, seed 42 and immediately explains one node, chosen automatically (the first node whose target hint changes after the first step). With the build that was live on 24 November 2026 this is what you see:

1. **Simulation.** A 16-node graph. Node 14 (orange ring) is the source. Nodes are coloured by the true `reach_h` at the selected step. The step slider says "step 1 of 5".
2. **Question.** "Step 1: why does the model predict `pi_h[2][6] = 5`, and `reach_h[2][6] = 1` (p = 0.998)? Truth: `reach_h[2][6] = 1`." Read: at step 1 the model is asked about node 6; its pointer says node 6's parent is node 5, and it says node 6 has been reached, with probability 0.998, which agrees with the truth.
3. **Quality note** (blue): the edge-site transcoder for bfs keeps 100% of the original F1 at 16 nodes and 100% at 64 nodes, with a reconstruction error (FVU) of 0.0008.
4. **The graph.** 47 nodes and 174 edges after pruning (of 2,633 nodes and 58,740 edges), keeping 80% of the influence. Columns are receiving nodes, rows are steps (step 1 on top, then step 0), each step with a row of latents and a row of inputs. The explained node 6 is marked with a star.
5. **Top latents at each step.** For step 1 the first row is latent `f936` on the pair 5 → 6, activation 9.99, influence 0.2686, "edge" (tier B: it fires on real edges), effect +0.130 (mostly raises). Next `f685` (unnamed), `f573` on 5 → 6 (tier A, "real edge"), `f412` (the sender is "on"), `f21`, and `f64` ("the sender is the source node", tier A).
6. **Inspector** for `f936`: "What it means: the pair is a real edge of the input graph, tier B: held-out precision 0.74, recall 0.47, F1 0.58"; effect "mostly raises the logit of reach_h at the receiver: +0.130 per unit of activation on average (sd 0.168, positive in 88% of 3,540 activations)"; it fires on 9.8% of pairs; 74% of its firings are on real edges (real edges are 16% of all pairs).

Everything in this list is a number the page computes or reads from a file; the test `ui_truth_test.mjs` re-derives each one independently.

## 2. The header

| control | what it does |
|---|---|
| **algo** (dropdown) | chooses one of the twelve algorithms. Each entry reads `name (type) . test score [CLEAN or SEVERE]`, for example `bfs (mask) . test 99.8% [CLEAN]`. The type is the type of the target. The score is the MPNN's F1 on 64-node graphs. Changing it loads that algorithm's model and a first graph. |
| **n** | number of nodes, 4 to 32 (default 16). The transcoders were trained at 16; above 16 the quality note turns orange. |
| **p** | edge probability of the sampler, 0.1 to 0.95 in steps of 0.05 (default 0.4). |
| **seed** | the sampler's seed. Changing n, p or seed loads the matching graph. |
| **New random graph** | picks a fresh random seed (one of the seeds already prepared in the background if available) and loads it. |
| status text (top right) | shows what the page is doing: `Loading dijkstra: 40%`, `bfs: n=16, length=6, seed=42 . in-browser model`, or a problem with a **Retry** button. |
| **Previous demo** link | goes to `/demo`. |

## 3. "About this algorithm" card

A collapsible card (open by default on screens at least 1181 px wide and 860 px high, closed on smaller ones). It shows for the selected algorithm: the best validation score, the test score on 64-node graphs, the drop between them, the number of training steps and minutes, the step at which validation first reached 90%, and the range of the validation score over the last 1,000 steps; then the class (CLEAN or SEVERE) and what it means for this algorithm; then "Why mask / scalar / ...": the reason the target has that type, and what the type means. "All 12 algorithms side by side" opens a table; clicking a row switches the algorithm. The numbers come from `summary_v3.json`, embedded in the page.

## 4. The simulation panel

- **The graph.** Nodes on a circle, edges as grey lines. Colours depend on the target type: mask and mask_one nodes are green when true and grey when false; categorical targets use one colour per class; scalar targets use a blue intensity scaled to the row. The source has an orange ring. A dashed blue ring marks the nodes the pseudo-code step is about.
- **Dots under nodes.** Up to five small dots under a node, one per latent of the pruned graph that is written to that node at the explained step; the colour is the pair kind (green real edge, grey non-edge, orange self pair). Clicking a dot pins that latent.
- **Click a node** to explain the model's prediction for that node at the current step (this is what builds the attribution graph).
- **Step slider and Play.** The slider goes over the algorithm's steps; the label says "step 1 of 5". Play moves through the steps automatically and traces the graph at each. The explained node stays the same; the step changes.
- **Legend.** source, true, false (or "colors = classes", "blue intensity = magnitude"), "active in this step".
- **Message map.** A sender-by-receiver grid for the explained graph. Shade = the summed influence of the pruned latents of that message at one step; outlined cells are real edges; hovering a cell shows the number; clicking a cell pins its most influential latent. The step buttons above it switch between the steps of the window. See `02-glossary.md`, "Message map".

## 5. The attribution graph panel

### 5.1 Question and chips

Question line, then the target chips: one for the pointer (for algorithms that have one, e.g. `pi_h[6] -> 5  1.000`) and one for the main hint (`reach_h[2][6] = 1  0.998`). A filled chip is the model's top choice; the bar is the probability. A chip for a scalar target shows the predicted number instead of a probability.

### 5.2 Quality note

The blue box (orange when the replacement fidelity of this algorithm is below 0.90, or when the graph has more than 16 nodes). It states: the transcoder, its size, what a latent is, the retention at 16 and (when measured) 64 nodes, the FVU and the sentence that the graph itself is exact for the real run.

### 5.3 Toolbar

| control | meaning |
|---|---|
| **Pruned / All** | Pruned: the pruned graph (nodes with 80% of the influence, edges with 98%). All: the strongest 6,000 edges of the full graph; choosing All raises "min |w|" to 0.5 if it was 0 so the picture stays readable. |
| **By node / Supernodes / By meaning** | the three views (see the glossary). Supernodes merge the latents of one pair at one step; By meaning merges latents found to mean the same thing, naming the rest "unnamed". |
| **min |w|** (slider 0 to 3) | hides edges whose absolute weight is below the value. |

### 5.4 The drawing

- **Columns**: `bias`, then one per receiving node (`-> 0 ... -> 15`; a star marks the explained node, "src" the source). In the supernode views the columns are the pairs or meanings.
- **Rows**: the top row is "Predictions of step t"; below it, for each step of the window from the newest to the oldest, a "Features" row (latents) and a "Leaves" row. Labels `STEP 1 . FEATURES`, `STEP 1 . LEAVES`.
- **Symbols**
  - circle: latent (size grows with the square root of its activation, up to a cap; colour = pair kind: green real edge, grey non-edge, orange self pair);
  - hatched square: error node of a pair;
  - grey square: leaf (encoded input or fed-back hint, for example `A[5,6]`, `adj[5,6]`, `pos[5]`, `reach_h[5]`, `pi_h[6,6]`); the "earlier state" leaf is shown as `h[k]`;
  - diamond: bias;
  - rounded chip: the prediction.
- **Labels** next to latents read `sender:latent` (for example `5:f936`); in crowded cells (more than four latents) the labels are not drawn and appear on hover.
- **Edges**: blue pushes up, red pushes down; thickness `0.5 + 3.5 x sqrt(|w| / largest |w|)`; edges from the bias are drawn fainter.
- **Hover** a node: its edges are highlighted and the others dimmed; a hover card shows what it is. **Click** pins it (also by keyboard: the nodes can be focused and Enter or Space pins). Hover an edge: a card shows its weight and its share of the inputs of its destination.
- **Legend**: latent on a real edge / non-edge pair / self pair, encoder leaf, error, bias, positive, negative.

### 5.5 The status line

Example: "Attribution graph (edge-site transcoder): 2,633 nodes / 58,740 edges at steps 0-1; pruned to 47 / 174, keeping 80% of the influence on the targets. Error nodes supply 0.2% of the absolute weight arriving at the predictions (the part the latents do not explain). ... Checked on this graph: 2,227 nodes, largest difference 5.0e-14. Edge weights and total effects were checked against finite differences in the test suite (verification/clrs-edge). Traced in your browser in 152 ms."

| part | meaning |
|---|---|
| nodes / edges at steps a-b | size of the whole graph before pruning and the steps of the window |
| pruned to x / y, keeping z% | nodes and edges kept, and the share of influence they carry |
| error nodes supply | share of the absolute weight arriving at the target logits that comes from error nodes |
| checked on this graph | how many nodes were checked and the largest difference between a node's value and the sum of its incoming edges plus its bias |
| traced in your browser in N ms | the time the worker needed |

When a trace is running the status reads `tracing the circuit...`, `tracing...` or `loading the model into your browser...`; if something goes wrong: "Could not trace this prediction just now. Click the node again to retry." or "Could not load what is needed for this view. Retrying...".

## 6. The pseudo-code panel

- Header: the algorithm name, then the code, one line per row with its number. The language is plain pseudo-code (`for each`, `if`, `while`, `repeat ... until`, assignments with an arrow).
- **Highlight rules.** The line being executed now has a coloured background and a coloured outline. The loops and tests around it have only a thin bar in their own colour. Colours show depth: level 1 blue, 2 green, 3 amber, 4 purple, 5 pink. Small round tags on a line show a current loop value (`u = 7`, `k = 4`, `(u, v) = (14, 5)`); long lists are not shown.
- **Step box**: the step number and a title (for example `STEP 1: EXPAND`), one sentence of what happened, and the main variables (`reached 2 of 16`, `new this step 5`).
- **Rings on the graph**: the nodes the step is about get a dashed blue ring in the simulation.
- **How a step is mapped to a line.** Comparing each recorded hint row with the previous one tells what the algorithm just did (a node turned gray, a distance decreased, a node was added to the tree...). Where the row alone does not say enough (Dijkstra's extracted node, Kruskal's edge order, the bridges' low values, articulation points) the missing detail is rebuilt from the graph and used only when it agrees with the recorded trace. 2,928 checks tie each highlighted line to a fact in the trace (`test_visual.js`). Floyd-Warshall records one step per `k`, so its inner loops (`i`, `j`) are shown as the whole pair loop, not as individual values.

## 7. The lower sections

Seven collapsible blocks. Click a title to open or close; the choice is remembered in the browser (`localStorage` key `clrs-sec-open`, one entry per section, plus one per inspector sub-section). Defaults: Top latents, Inspector and Intervention open; the rest closed. Each title carries a live summary.

### 7.1 Top latents at each step
For every step of the window the six most influential latents of the pruned graph: latent number, pair, activation, influence, meaning (name and tier, or "unnamed") and effect. Click a row to pin the latent in the graph and the inspector.

### 7.2 Latent dictionary
The named latents of the algorithm (up to 20, best F1 first) with the condition, tier, precision, recall, F1, firing rate and effect.

### 7.3 Per-node predictions
`node | reach_h(1) | pred reach_h(2) | true(2)` for every node: the true hint now, the real model's prediction for the next row (blue; shows `...` until the step has been traced), and the true next value. For categorical targets the prediction reads `class k`; for scalar targets a number; for mask targets a probability; for mask_one a softmax probability.

### 7.4 Model & edge-site transcoder
`hidden_dim 128`, the target, "val / test score (OOD)" from `summary_v3.json`, and the facts of this algorithm's transcoder: size, tier, FVU (validation, train, 64 nodes), the error by step with a small curve, dead latents, active latents per pair, replacement fidelity, winner flips, hidden-state error, training size and time. All from `edge_tc/clrs-edge-summary.json`; the test compares them with `report.json`.

### 7.5 How to read the attribution graph
A short guide (nine bullets) inside the page.

### 7.6 Inspector
Shows the pinned node, or the hovered one while nothing is pinned. For a **latent** the sections are: header (latent, step, pair, pair kind), activation, influence, where it fires in this graph; **What it means** (name and tier or "Unnamed", effect, firing rate, kinds of pair); **Best-matching conditions (held-out graphs)**; **Strongest activations** with "show" buttons (these load the graph of that example and explain that receiver at that step); **Total effect on the targets**; **Incoming edges**; **Outgoing edges** (top ten by absolute weight, bars with signed values). For an **error node** the norm of the error; for a **leaf** its value and meaning; for the **bias** its role; for a **logit** its logit and probability; for a **supernode** its members.

### 7.7 Intervention
Three buttons: "Clamp to -1x", "Ablate (x0)", "Amplify (x2)". The result table lists, for each target, the real logit and the changes "Delta actual" (real network) and "Delta graph" (prediction), then a sentence saying whether the direction was confirmed, how many frozen decisions changed, and the measured reliability of the prediction (69% of directions matched in the project's tests, 44% to 77% depending on the algorithm).

## 8. Responsive behaviour

| width | layout |
|---|---|
| at least 1181 px | three columns in the top row (simulation, graph, pseudo-code); two columns below; the right column of the lower part (inspector, intervention) stays in view while scrolling |
| 761 to 1180 px | two columns in the top row (the graph spans the full width below); one column below |
| 760 px and below | one column everywhere; code text and tables are scrollable instead of overflowing |

The layout tests check that there is no horizontal overflow at 1440, 1024 and 390 px.

## 9. What happens when the page loads

1. The server returns the page with the build id and the list of algorithms filled in. Scripts are requested with `?v=<build id>` and cached immutably.
2. The page starts early downloads (the first model bundle and the first instance) before the scripts finish, if the browser supports it.
3. `boot()` builds the algorithm dropdown and loads the default graph (bfs, n = 16, p = 0.4, seed 42): instance from `/api/instance`, model bundle from `/models/bfs.*`, which a Web Worker turns into a running model. The first graph is shown as soon as the first steps are ready (the worker streams the first two steps, then the rest).
4. The first node is chosen and the attribution request is sent to the worker; it needs the edge-site transcoder (`/edge_tc/bfs.bin`, 1.3 MB) and, after the graph is drawn, the latent table (`/edge_tc/bfs.latents.json`, about 49 KB compressed) and the summary file.
5. When the person is idle (no input for 1.2 seconds, no active transfer, tab visible) the page quietly downloads the other algorithms' model bundles and edge-site transcoders with low priority; at most four algorithms are kept ready in the workers (`RES_CAP = 4`).
6. Prepared seeds: while idle, the page computes the next few random graphs in the background so "New random graph" is instant.

Caches: instances (the last 60), computed graphs (a bounded cache), the models in the workers (4), the browser's own HTTP cache (immutable files).

## 10. Errors a visitor can see

| message | cause | what to do |
|---|---|---|
| "This page was just updated. Reload to continue." | a new build was deployed while the page was open | reload |
| "Could not reach the server. Check your connection, then press Retry." | the first request failed | Retry |
| "Could not trace this prediction just now. Click the node again to retry." | the worker could not finish a trace | click the node again |
| "This graph has no explainable step. Click New random graph." | a graph with fewer than two hint rows | New random graph |
| Chrome shows a warning instead of the page | the address is plain `http://` (Chrome's HTTPS-first behaviour) | choose to continue, or use an HTTPS address once it exists |

## 11. Where `/demo` differs

`/demo` is the earlier version: the attribution graph is built from the node-site transcoder (192 features with labels and top-activating examples), the label table is present, the per-node table shows predictions approximated by the transcoder before a step is traced, and the lower row has three panels (model and per-node table, feature labels, inspector with intervention). It has a tag "previous demo" and a link to the main page. The pseudo-code panel and the algorithm card are the same. Its graphs are byte-identical to those of the version before the edge-site work (test 6).

## 12. Which file draws what

| thing | file and function |
|---|---|
| page structure and styles | `viewer3.html` (markup, `<style>`) |
| algorithm card | `renderAlgoInfo`, `algoLabel` in `viewer3.html`; data `SUMMARY_V3` and `TRAIN_V3` in the same file |
| simulation graph | `drawGraph`, `nodePositions`, `colorForNode` in `viewer3.html` |
| pseudo-code panel | `renderPseudo`, `pseudoNow` and the inline engine (`PSEUDO`) in `viewer3.html` (source `verification/pseudo/pseudo_block.js`) |
| per-node table | `renderNodeTable` in `viewer3.html` |
| model panel | `fillHeader`, `renderEtcPanel` in `viewer3.html` |
| quality note | `renderEdgeNote` in `viewer3.html` |
| attribution graph (layout, drawing, hover, pin) | `avCurrent`, `avPlace`, `avRender`, `avHighlight` in `attrview.js` |
| question and chips | `avQuestion`, `avTargetText`, `avChipText` in `attrview.js` |
| inspector | `avInspect`, `avMeaningHtml`, `avSec` in `attrview.js` |
| top latents, dictionary, message map | `renderTopLatents`, `renderDictionary`, `renderMessageMap` in `attrview.js` |
| interventions | `avIntervene`, `avIvResult` in `attrview.js`; `intervene` in `clrs-edge-attrib.js` |
| hover cards | `explain.js` |
| the worker | `mpnn_worker.js` |
