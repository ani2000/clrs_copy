# Glossary: every keyword, in detail

This is the dictionary of the project. Each entry says what the word means, where you meet it on the site or in the repository, and what it is connected to. Words are in alphabetical order inside topic groups; the last section is a plain alphabetical index of all entries.

Conventions: `code font` is a file, function, field or value that really exists; "the page" is the main page `http://98.86.163.27/`; "`/demo`" is the previous version at `http://98.86.163.27/demo`.

## Contents

1. [Benchmark and data](#1-benchmark-and-data)
2. [The neural network](#2-the-neural-network)
3. [Training and scores](#3-training-and-scores)
4. [Interpretability: transcoders and latents](#4-interpretability-transcoders-and-latents)
5. [The attribution graph](#5-the-attribution-graph)
6. [Interventions](#6-interventions)
7. [Measuring the transcoders](#7-measuring-the-transcoders)
8. [What a latent means](#8-what-a-latent-means)
9. [The page and its parts](#9-the-page-and-its-parts)
10. [How the page loads and caches things](#10-how-the-page-loads-and-caches-things)
11. [The server and the API](#11-the-server-and-the-api)
12. [AWS and deployment](#12-aws-and-deployment)
13. [GitHub and automation](#13-github-and-automation)
14. [Testing words](#14-testing-words)
15. [Alphabetical index](#15-alphabetical-index)

---

## 1. Benchmark and data

### CLRS-30
A benchmark from DeepMind of thirty classic algorithms taken from the textbook *Introduction to Algorithms* (Cormen, Leiserson, Rivest, Stein, "CLRS"). Each algorithm comes with a generator of random inputs, a reference implementation, and the step-by-step state of that implementation (the hints). This project uses twelve of the thirty, the graph algorithms.
*Where:* the page title area ("CLRS-30 Circuit Tracer"), `README.md`, the file `requirements.txt` (the `clrs` library is installed from GitHub).

### Algorithmic reasoning
Teaching a neural network to carry out an algorithm, not only to give the final answer. The network is trained to imitate every intermediate step of a textbook implementation.

### Graph
A set of nodes with edges between some pairs. In this project a graph has `n` nodes (default 16, between 4 and 32 on the page) and each possible edge exists with probability `p` (default 0.4) in the sampler.

### Node, edge, ordered pair
A node is a vertex of the graph. An edge is a connection. An *ordered pair* (sender, receiver) is any two nodes in order, whether or not an edge exists between them. There are `n x n` ordered pairs, including pairs of a node with itself.
*Where:* the new attribution graph puts its latents on ordered pairs. The page calls the pair kinds "real edge", "non-edge" and "self pair".

### n
The number of nodes of the sampled graph. The input box "n" on the page accepts 4 to 32; the transcoders were trained on 16-node graphs, so n = 16 is the setting every guarantee refers to.

### p
The edge probability used by the CLRS graph sampler. The input box "p" on the page accepts 0.1 to 0.95 in steps of 0.05; the default is 0.4.

### seed
The random seed of the sampler. The same (algorithm, n, p, seed) always gives the same graph, so any result can be reproduced by typing the same four values. If the sampler produces a degenerate graph (fewer than 3 algorithm steps) it retries with a shifted seed (+1000 and so on); the instance then reports the seed it actually used, but the page still loads the same graph for the same requested seed.
*Where:* input box "seed", the "show" buttons of the strongest examples, the example lists in `15-latent-dictionaries.md`.

### Sampler
The CLRS component that produces random graphs and the reference trace for an algorithm. On this site it runs on the server (`model_export.build_instance`, `/api/instance`); the browser never runs it.

### Reference implementation
The textbook implementation inside CLRS that produces the true hints and the true outputs. It is the ground truth the model is compared with ("Truth" on the page, the "true" column of the per-node table).

### Hint
An intermediate quantity of the reference implementation recorded at every step, for example "which nodes have been reached so far" in BFS. Hints are the supervision that teaches the network the algorithm's steps. The network reads the previous hints as input and predicts the next hints.
*Where:* the table of fields in `04-the-12-algorithms.md` (stage "hint"); the target of the explanation is a hint.

### Trajectory
The sequence of hint rows from step 0 to the last step for one graph. `length` is the number of rows; the number of message-passing steps is `length - 1`.

### Step
One round of message passing, which turns hint row t into hint row t+1. The step slider of the page runs over `length - 1` steps. "Step 1" on the page means the transition from hint row 1 to hint row 2, which is why a question about step 1 mentions `reach_h[2][6]`.

### Input, hint, output (stage)
The three stages of a CLRS field. *Inputs* never change (the graph, the weights, the source). *Hints* change at every step. *Outputs* are the final answer. The model bundle exported for the browser contains inputs and hints only.

### Location: node, edge, graph
Where a field lives. A *node* field has one value per node, an *edge* field one value per ordered pair, a *graph* field one value for the whole graph.

### Type: mask, mask_one, categorical, scalar, pointer
The kind of value a field holds.
- **mask**: one true/false value per node; any number can be true. Predicted with a sigmoid.
- **mask_one**: exactly one node is true at a time. Predicted with a softmax over nodes.
- **categorical**: one class per node out of a small fixed set (for example the three DFS colours). Predicted with a softmax over classes.
- **scalar**: one real number (a distance, a key, a time).
- **pointer**: for every node, a link to another node (a parent). Predicted with a softmax over nodes.
*Where:* the "target type" in the algorithm dropdown, `meta.json`, the explanation card "Why mask/scalar/...". See also [the target](#target-hint).

### Target (hint)
The one hint the page explains. For each algorithm it is fixed in `algos/<algo>/meta.json`: `reach_h` for bfs, `color` for dfs, bridges and strongly connected components, `d` for the shortest-path algorithms, `key` for Prim, `u` for Kruskal, `topo_head_h` for topological sort, `is_cut_h` for articulation points and `k` for Floyd-Warshall.

### Source node (s)
The starting node of the five algorithms that have one: bfs, bellman_ford, dijkstra, dag_shortest_paths and mst_prim (`source_field` is `s` in their `meta.json`). The other seven have none. Drawn with an orange ring on the page.

### pi_h (parent pointer hint)
For algorithms that build a tree, the hint holding every node's current parent. The page adds the parent pointer to the question when the algorithm has one ("why does the model predict pi_h[2][6] = 5, and reach_h[2][6] = 1").

### Adjacency (adj)
The input matrix saying which ordered pairs are edges. The MPNN overrides it with all ones when it passes messages (see [fully connected](#fully-connected-message-passing)); the real adjacency still enters as an edge input feature and is used on the page to classify pairs.

### Instance
The compact description of one sampled graph that the browser needs: `n`, `length`, `seed`, `adj`, `source`, `target_true`, `inputs` and `hints0` (the first hint row). Served by `/api/instance`. See `07-data-files-and-formats.md`.

### Length
Number of hint rows of the instance. Degenerate graphs with length below 3 are resampled.

### Validation graphs / test graphs / OOD
The MPNNs were trained on small graphs. *Validation* graphs have 16 nodes, the training size. *Test* graphs have 64 nodes, four times larger. *OOD* means out of distribution: the 64-node test measures whether what the model learned works on sizes it never saw.

---

## 2. The neural network

### MPNN (message-passing neural network)
The kind of network studied here. Each node keeps a state vector; at every step nodes send messages to each other, each node aggregates the messages it receives, and updates its state. In CLRS the MPNN processor is deliberately simple, which makes it analysable.

### Hidden size / 128-hidden
Every node state is a vector of 128 numbers (`hidden_dim = 128` in the model panel). It does *not* mean 128 layers: there is one processor with one set of weights, applied once per algorithm step, like the body of a loop.

### Processor
The part of the network that does the message passing. Its weights are the same at every step. It consists of the linear layers `m1`, `m2`, `me`, `mg` (building the message input), a two-layer MLP (the message function), the node update layers `o1`, `o2`, and a LayerNorm.

### Encoder / decoder
Encoders turn CLRS inputs and hints into 128-number vectors the processor can read; decoders turn the processor's states into predictions for the hints and the outputs. Each field has its own encoder and, for hints, its own decoder.

### node_fts, z, e, g
The names used for the processor's inputs at one step. `node_fts` is the sum of the encodings of all node-located fields (128 numbers per node). `z = [node_fts ; h]` is the concatenation of that with the node's hidden state `h` (256 numbers). `e` is the encoding of the edge-located fields (128 numbers per ordered pair). `g` is the encoding of the graph-located fields (128 numbers).

### Hidden state (h)
The 128-number state each node carries from one step to the next. It starts at zero. After each step it is `LayerNorm(ReLU(O1 z + O2 m))`.

### Message
What a sender sends to a receiver at one step. For the ordered pair (i to j) the message is a 128-number vector `y_ij` produced by a small MLP from the sum of four linear terms (see [site input](#site-input-s_ij)).

### Sender, receiver
In an ordered pair (i, j), i is the sender and j the receiver. The new attribution graph puts a pair's latents in the column of the *receiver*, because the message is written to the receiver's state.

### Fully connected message passing
The CLRS MPNN ignores the graph's edges when sending messages: every node sends a message to every node (the adjacency is overridden with ones). The graph's real edges reach the network only as an edge input feature. This is why every ordered pair, including non-edges and self pairs, has latents on the new graph.

### Max aggregation
A receiver combines the messages it gets by taking, for each of the 128 channels separately, the largest value over all senders: `m_j[c] = max over i of y_ij[c]`. (The CLRS paper found max better than sum for these tasks.)

### Channel
One of the 128 coordinates of a message or a hidden state. Channels are the unit at which the max is taken.

### Winner (route)
For a receiver j and a channel c, the sender whose message value is the largest. Different channels can have different winners. The graph builder freezes the winners at their real values. A sender that wins no channel at a receiver has no effect there, so it has no outgoing edges to that receiver's state.

### Max-routing
The pattern of winners as a whole. The page says "max-routing winners ... frozen".

### MLP (message function)
Two linear layers with ReLUs: `y = L2 ReLU(L1 ReLU(s) + b1) + b2`. It is the only real nonlinearity before the hard max, which is why the edge-site transcoder replaces it.

### ReLU
The function `max(0, x)`. A ReLU unit is "on" when its input is positive. The ReLU pattern of the node update (which channels are on) is one of the three things frozen to make the graph linear.

### LayerNorm
Normalises a vector to zero mean and unit variance and then rescales it with learned `gamma` and `beta`. In the processor it is applied to each node's updated state; its standard deviation (`sigma`, computed with a small epsilon of 1e-5) is the second thing frozen in the graph.

### Node update
`x_j = O1 z_j + O2 m_j + bo`, `u_j = ReLU(x_j)`, `h'_j = LayerNorm(u_j)`. The node-site transcoder replaces this whole map; the edge-site transcoder leaves it in the network (frozen, hence linear).

### Hint feedback
The network's own predicted hints at step t are encoded again and become inputs at step t+1. During training the true hints are fed back with probability 0.5 (see [teacher forcing](#hint-teacher-forcing)); at test time (and on the page) the model's own predictions are fed back (in "soft" form).

### Soft hint feedback / repred
How predictions are fed back: as probabilities (sigmoid or softmax outputs), not as hard 0/1 decisions. The browser port `mpnn.js` reproduces this exactly.

### JAX, Haiku, optax
The software the models were trained and are run with on the server. JAX does the numerical computation, Haiku (dm-haiku) defines the layers, optax was used to train. On the instance only JAX and Haiku are needed (and only for `/api/graph` and `/api/contexts`).

### Checkpoint (`best.pkl`)
The saved parameters of a trained model. `checkpoints_v3/<algo>/best.pkl` holds the parameters of the best validation step of the training run of that algorithm. They are stored with Git LFS and are the only trained-model input of the project.

### Model bundle
`models/<algo>.json` (manifest) and `models/<algo>.bin` (float32 weights) written by `model_export.py` at image build time; what the browser downloads to run the model itself. About 1.2 to 2.1 MB per algorithm.

---

## 3. Training and scores

### Training step
One update of the weights on a batch of graphs. Each MPNN was trained for 10,000 steps with batch size 32, learning rate 0.001, gradient clipping 1.0 and hint teacher forcing 0.5 (see `summary_v3.json`, field `protocol`).

### Hint teacher forcing
During training, with probability 0.5 the true hint is fed back into the next step instead of the model's own prediction. This stabilises training; at test time only the model's own predictions are fed back.

### F1 / micro-F1 / score
The measure of how well a model's outputs match the truth. For mask-like outputs it is F1, the harmonic mean of precision and recall; for pointer outputs CLRS uses accuracy. The files call both "F1". `best_val_f1` is the best validation score during training, `test_f1` the final score on 64-node graphs.

### Precision and recall
For a set of predicted positives: precision = the share that are truly positive; recall = the share of truly positive items that were found. F1 = 2 * P * R / (P + R).

### Gap
`best_val_f1 - test_f1`: how much the score drops from 16-node to 64-node graphs.

### CLEAN / SEVERE
The project's classes by gap. **CLEAN**: gap below 0.10 (the model generalises to larger graphs). **SEVERE**: gap above 0.40. In this training run all twelve algorithms fall into one of the two (largest CLEAN gap 7.4 points for Prim, smallest SEVERE gap 46.0 for Kruskal).
*Where:* the algorithm dropdown (`[CLEAN]`, `[SEVERE]`), the "About this algorithm" card.

### `summary_v3.json`
The output of the training run: for each algorithm best validation score, test score, steps, minutes, validation curve, plus the protocol. The numbers in the "About this algorithm" card come from it.

### Validation curve
The validation score sampled every 100 steps during training (100 points per algorithm in `results_v3/<algo>__mpnn.json`).

### Reached 90% at
The first training step at which the validation score was at least 0.9 (card field).

---

## 4. Interpretability: transcoders and latents

### Mechanistic interpretability
Trying to understand a neural network by reverse-engineering its internal computation, not only by testing inputs and outputs. This project is modelled on Anthropic's "On the Biology of a Large Language Model" and applies the same method to a network that executes algorithms.

### Circuit
The set of internal features and the connections between them that cause one particular prediction.

### Superposition
Networks store many more concepts than they have coordinates, so single coordinates (the 128 numbers of a state) mix several meanings. Sparse dictionaries are used to pull the mixture apart.

### Sparse dictionary / sparse autoencoder
A wide layer (here 1,280 or 192 entries) of which only a few entries (8) are active at a time, trained to reproduce something from the network. Each entry is a candidate "feature".

### Transcoder
A sparse dictionary that reads the *input* of one part of a network and predicts the *output* of that part (unlike an autoencoder, which reconstructs its own input). Replacing the part by its transcoder gives a network made of sparse, more interpretable pieces.

### Site
The part of the network a transcoder replaces. Two sites exist in the MPNN: the **message function** (edge site) and the **node update** (node site).

### Edge-site transcoder (new)
The transcoder on the message function. Input: the pre-ReLU message input `s_ij` of an ordered pair (128 numbers). Output: the message `y_ij` (128 numbers). 1,280 latents, 8 active per pair. One per algorithm, in `edge_tc/`. Used by the main page.

### Node-site transcoder (previous)
The transcoder on the node update. Input: 384 numbers (the node's `z`, 256, and its aggregated message, 128). Output: the new hidden state (128). 192 features, 8 active. One per algorithm, in `algos/`. Used by `/demo`.

### Site input (s_ij)
`s_ij = M2 z_i + M1 z_j + Me e_ij + Mg g + b`: the sum of four linear terms plus biases, before the first ReLU of the message function. Because it is a sum of a sender term, a receiver term, an edge term and a graph term, every latent's pre-activation splits exactly into those four parts.

### Site output (y_ij)
The message: `y_ij = L2 ReLU(L1 ReLU(s_ij) + b1) + b2`.

### Latent
One of the entries of the edge-site dictionary (numbered 0 to 1,279). On the page it is written `f936` (latent number) and sits on one ordered pair at one step.

### Feature
The word used for the node-site dictionary entries (0 to 191) and, loosely, for latents. On `/demo` features have labels.

### TopK
The sparsity rule: of all 1,280 pre-activations keep only the k = 8 largest (ties at the threshold are all kept) and set the rest to zero. After TopK the kept values are passed through ReLU, so a latent is active only if it is among the 8 largest *and* positive.

### Encoder, decoder, b_pre, b_enc, b_dec
The transcoder's parameters. Encoder: `pre = (x - b_pre) W_enc + b_enc` with `x = (s - s_mu) / s_scale`. Decoder: `y_hat = (a W_dec + b_dec) * y_scale + y_mu`. `W_enc` is 128 x 1280, `W_dec` is 1280 x 128.

### s_mu, s_scale, y_mu, y_scale
Fixed normalisers stored inside the transcoder file so that the transcoder works in the network's raw units: `s_mu` and `y_mu` are mean vectors (128), `s_scale` and `y_scale` are single numbers.

### Pre-activation
The value of a latent before TopK and ReLU. The attribution graph explains pre-activations: each is exactly the sum of its incoming edges.

### Activation
The value of a latent after TopK and ReLU. Zero for inactive latents; for the 8 active ones it equals the pre-activation.

### L0
The number of active latents per row. With TopK it is at most 8; the measured average is about 7.9 because some rows have fewer than 8 positive pre-activations.

### Dead latent
A latent that never fires on the training rows. 44% of the bfs edge-site latents are dead. The summary reports `dead_frac`.

### Auxiliary loss
An extra training term (from the ProtoMech recipe) that asks dead latents to explain the part the live ones missed, so that fewer latents die.

### FVU / NMSE (reconstruction error)
Fraction of variance unexplained: sum of squared reconstruction errors divided by the variance of the true message around its mean. 0 means perfect, 1 means no better than the mean. The two names mean the same thing in these files.

### Error node
The part of a message the latents did not reconstruct: `eps_ij = y_ij - y_hat_ij`, computed on the real run. Adding it back gives `y_hat + eps = y` exactly, so the replacement model reproduces the real network. In the graph, error nodes are hatched squares.

### Replacement model
The network with a site replaced by its transcoder (plus the error node when exactness is wanted).

### Direct, frozen, full (replacement modes)
*Direct*: one step is replaced and everything else is the real network. *Frozen*: all steps are replaced, but winners and sigma are taken from the real run. *Full*: all steps are replaced and winners, sigma and hints are recomputed. The training report evaluates all three.

### plain, plumb
Two plumbing checks: *plain* wires the transcoder in without the error node, *plumb* adds the error node back (it must equal the unmodified network).

### Hint policy (H-true, H-orig, H-live)
Which hints are fed into the steps when the transcoder is trained or evaluated: the ground truth, the original model's own predictions, or the replaced network's own predictions.

### Retention
The share of the unmodified network's score that is kept when the transcoder is put into the network, for example 99.0%. It is the main faithfulness number.

### Winner flip
When the replacement changes which sender wins a channel. A flip needs twice the reconstruction error to exceed the gap between the best and second-best sender (the "2 delta >= gap bound"); the training report checked that the bound was never violated.

### Tier A / B / C (transcoder tiers)
By the baseline MPNN: **A** = validation F1 at least 0.95 and test F1 at least 0.90 (replacement judged at 16 and 64 nodes); **B** = validation at least 0.95 but test below 0.90 (judged at 16 nodes only); **C** = validation below 0.95 (indicative only). Not the same thing as the tiers of latent labels (see [tier A/B of a latent](#tier-a--b-of-a-latent)).

### Parity
The check that the training code reproduces the recorded score of the baseline network exactly (difference 0.0 for bfs).

### Offline parity
The check that the transcoder training code recomputes the site input, output and hidden state to within about 1e-3 of the network's own values.

---

## 5. The attribution graph

### Attribution graph
For one prediction (one node at one step), a graph whose nodes are the internal quantities that mattered and whose edges are exact linear contributions, so that every node's value is the sum of its incoming edges. It is built by freezing the network's nonlinear decisions at their real values.

### Frozen (winners, sigma, ReLU pattern, TopK set)
Fixed at the values they have on this one input. With them frozen the replacement model is linear in the latent activations, which is what makes edge weights well defined. The price: the graph describes the network on this input only, and a large change of a latent can flip the frozen decisions.

### Linearisation
Replacing a nonlinear function by the linear function that agrees with it at the current input. Here nothing is approximated at the current input; what is "linear" is the response to changes with the frozen decisions held.

### Question line
The sentence above the graph: "Step 1: why does the model predict pi_h[2][6] = 5, and reach_h[2][6] = 1 (p = 0.998)? Truth: reach_h[2][6] = 1."

### Logit
The raw number a decoder produces before it is turned into a probability. The graph explains logits. For a mask target, probability = sigmoid(logit); for a class or pointer, a softmax over logits.

### Target chip (output chip)
The rounded boxes at the top of the graph: one for the main hint of the receiver and one for the pointer prediction. Filled chips are the model's top choice; the bar shows the probability.

### Node types of the new graph
- **Latent** (circle): a latent on one sender-to-receiver pair at one step.
- **Error node** (hatched square): what the latents of a pair did not rebuild.
- **Leaf** (square): an encoded input or a fed-back hint at one step.
- **Earlier-state leaf**: the hidden state from before the three-step window.
- **Bias** (diamond): all constant terms.
- **Logit** (chip): the prediction being explained.

### Leaf
An input of the explanation: the value an encoder reads (position, edge weights, adjacency, the source mask, and the hints fed back from the previous prediction). Leaves are where the explanation stops.

### Bias node
One node per step collecting every constant term on the frozen path (encoder, message-site, node-update, hidden-state-decoder and transcoder biases and offsets), plus one for the predictions. It is computed from the constants, not chosen to make the sums work.

### Window
The number of steps shown: the last three. State older than the window enters as one "earlier state" leaf per node.

### Edge (in the graph)
A direct linear contribution from a source node to a destination node: `weight = source activation x effect of the source on the destination`. Blue edges push the destination up, red edges push it down. Edge thickness grows with the square root of |weight|.

### Skip edge
An edge that jumps a step. Hidden state reaches later steps not only through latents but also through the node update's `O1 z` term; those contributions are exact edges from a latent two steps back.

### Sum test (exactness)
For every node: value = sum of incoming edge weights + bias. The page reports the largest difference over all nodes of the graph; it is about 1e-14.

### Influence
How much of the predictions' (probability-weighted) inputs flow through a node, counting all paths. It is the quantity used to prune the graph. Not the same as total effect.

### Pruning (80% / 98%)
The Circuit Tracing recipe: keep the nodes carrying 80% of the influence (plus the target nodes), then the edges carrying 98% of the influence among them. The status line reports how much influence the pruned graph keeps.

### Pruned / All
The two edge sets of the toolbar. Pruned shows the pruned graph. All shows the strongest 6,000 edges of the full graph.

### min |w|
A slider that hides edges whose absolute weight is below the chosen value.

### By node / Supernodes / By meaning
The three views. *By node*: every latent drawn once. *Supernodes*: all latents of one sender-to-receiver pair at one step merged into one box, edges summed. *By meaning*: latents that were found to mean the same thing merged into one box ("unnamed" for the rest).

### Supernode
A group of nodes shown as one box, with the edges between groups equal to the sum of the underlying edges (edges inside a group are dropped).

### Column / row
Columns of the graph are the receiving nodes (`-> 6`, with a star for the explained node and "src" for the source); rows are steps, each with a row of latents and a row of leaves. The predictions are on top.

### Total effect
The change of a target logit per unit change of a node's activation, summed over every path (direct, skip and indirect) in the frozen-linear graph. The inspector shows it per target. Checked against finite differences.

### Error share
The share of the absolute weight arriving at the target logits that comes from error nodes, shown in the status line. Low values mean the latents explain almost all of the prediction.

### Message map
A sender-by-receiver grid under the simulation: how much influence the pruned latents of each message carry at one step. Outlined cells are real edges.

---

## 6. Interventions

### Intervention / clamp
Changing a latent's activation by a chosen factor and re-running the real network to see whether the prediction moves as the graph says. For the edge-site latents the message of the latent's pair is changed by `(factor - 1) x activation x decoder vector x y_scale`.

### Ablate (x0), invert (-1x), amplify (x2)
The three buttons: factor 0 removes the latent, factor -1 reverses it, factor 2 doubles it.

### Delta actual / Delta graph
"Delta actual" is what the real network did (change of each target logit). "Delta graph" is what the graph predicted: `(factor - 1) x activation x total effect`.

### Direction confirmed / not confirmed
The page says "Direction confirmed" when every logit that moved noticeably (more than 0.001) moved in the predicted direction. In the project's tests the direction matched in 69% of comparisons (44 to 77% depending on the algorithm); the graph is a first-order guide, not a guarantee.

### Decisions changed
"38 of 4,352 downstream max-routing winner / TopK decisions changed": how many frozen decisions differed in the intervened run. The more that changed, the further the run was from the frozen-linear picture.

### Finite difference
Estimating a derivative by making a tiny change and measuring the response. Used in the tests to check edge weights.

---

## 7. Measuring the transcoders

### E0, E1, E2
Names of the checks in the training plan. **E0**: plumbing (identity replacement reproduces the baseline). **E1**: reconstruction error, per step. **E2**: the transcoder inside the network (retention).

### Golden test / reference vectors
`clrs-transcoder-reference.json` holds, for every algorithm and site, three random input rows with the active latent indices and the reconstruction, computed independently. A port must reproduce them.

### Port
Re-implementing the transcoder maths in another language (here JavaScript) from the same weights.

### Cross-check
Comparing the port with a second independent implementation (numpy, float64) on the same inputs.

### Brute force
A deliberately simple implementation used to check a faster one.

### Held-out
Data the choice was not fitted on. For latent meanings: conditions are chosen on graphs with even seeds and scored on graphs with odd seeds.

### Evaluation set sizes
Train rows, validation rows and 64-node test rows of the training run (for bfs: 207,872, 28,928 and 32,768).

---

## 8. What a latent means

### Condition
A simple, checkable statement about an ordered pair at a step, such as "the sender is already reached" or "real edge from a reached sender to an unreached receiver". The conditions use the true hints and the input graph; they are listed per algorithm in `15-latent-dictionaries.md`.

### Base rate
The share of held-out pair rows on which a condition is true. A latent that fires on 100% of non-edge pairs is much less surprising when 78% of all pairs are non-edges.

### Named / unnamed
A latent is *named* when its best condition reaches a held-out F1 of at least 0.5. Otherwise it is *unnamed* and the page shows only the facts measured about it.

### Tier A / B of a latent
Naming strength by held-out F1: **A** at least 0.8, **B** at least 0.5. Shown next to a name.

### Effect (of a latent)
The change of the receiver's main target logit per unit of the latent's activation, averaged over the activations measured, with its standard deviation and the share of positive values. "Raises" means positive in at least 90% of activations, "mostly raises" at least 65%, "mixed" in between, "mostly lowers" at most 35%, "lowers" at most 10%.

### Where it fires
How often a latent fires and on which kind of pair (real edge / non-edge / self), compared with how common those pairs are.

### Strongest activations
The three largest activations of a latent among the measured graphs, as (seed, step, sender, receiver); the "show" button loads that graph and explains that receiver at that step.

### Dictionary
The table of the named latents of the algorithm with their evidence.

### Latent table (`edge_tc/<algo>.latents.json`)
The file behind all of the above.

---

## 9. The page and its parts

### Header
Algorithm dropdown, n, p, seed, "New random graph", a status text, and the link to the other version.

### About this algorithm
The collapsible card under the header with the training scores, the reason for the target type, and the table of all twelve algorithms (click a row to switch).

### Simulation panel
The sampled graph at the selected step, coloured by the true hint; nodes with dots show latents of the pruned graph; a step slider and a Play button; the message map.

### Attribution graph panel
Question, target chips, quality note, toolbar, the graph, legend and the status line.

### Quality note
The blue box above the graph with the measured fidelity of this algorithm's transcoder; orange when retention is below 0.90 or the graph has more than 16 nodes where the transcoder was not validated.

### Pseudo-code panel
The textbook algorithm in plain pseudo-code, with the line executed at the current step highlighted. Colours show nesting depth; thin bars show the loops and tests around the highlighted line; small tags show current loop values.

### Level colours (nesting depth)
Blue = level 1, green = level 2, amber = level 3, purple = level 4, pink = level 5.

### Top latents at each step
Table of the six most influential latents of each step of the window with pair, activation, influence, meaning and effect.

### Latent dictionary section
The named latents of the algorithm.

### Per-node predictions
The true hint now, the real model's prediction for the next step (after the step has been traced), and the true next value for every node.

### Model & edge-site transcoder section
Facts about the MPNN and the measured quality of the transcoder.

### How to read the attribution graph
A short guide inside the page.

### Inspector
Shows the pinned (or hovered) node: for a latent its pair, activation, influence, meaning, conditions, strongest activations, total effect, incoming and outgoing edges.

### Intervention section
Buttons to clamp the pinned latent and the table with the result.

### Collapsible section
A block that opens and closes by clicking its title; the state is remembered in the browser (`localStorage` key `clrs-sec-open`).

### Hover card
The explanation that appears when the pointer rests on any dotted term, number or box.

### Pin
Click a node of the graph to fix it in the inspector; click again to release.

### Play
Steps through the algorithm automatically and re-traces the graph at each step.

### Skeleton
The grey placeholder circles shown while the first graph loads.

---

## 10. How the page loads and caches things

### In-browser model
The trained network runs inside the visitor's browser (a JavaScript port, `mpnn.js`) so a new graph needs only a few kilobytes from the server.

### Web Worker
A background thread of the browser. `mpnn_worker.js` runs the model and the attribution engines there so the page stays responsive.

### Prefetch (quiet background download)
While the person is idle, the page downloads the model bundles and edge-site transcoders of the other algorithms with low priority so switching is instant.

### Speculation
Pre-computing the graphs most likely to be asked for next (the next seeds) in the background.

### Build id (`BUILD`)
A 10-character hash over the page, the scripts, all model bundles and all `edge_tc/` files. It changes whenever any of them changes.

### `?v=BUILD` (versioned URL)
Every URL of a script, model or API call carries the build id so it can be cached forever without ever serving an old file.

### Immutable cache
`Cache-Control: public, max-age=31536000, immutable`: the browser may keep the file for a year without asking. Safe because the URL changes with every new build.

### ETag / 304
A fingerprint of a file. The browser asks "has it changed?" and the server answers 304 Not Modified when not. Used for the pages, which are not cached blindly.

### 409 stale
If an old page (old build id) is still open after a deploy, the server answers 409 for its versioned requests; the page then shows "This page was just updated. Reload to continue." instead of mixing old and new files.

### gzip
Compression of text over the network. Pages, scripts and the latent tables are sent compressed (about 3 to 4 times smaller); model bundles barely compress (about 8%) and are sent as they are.

### Residency cap
The page keeps at most 4 algorithms ready in the background workers (`RES_CAP = 4` in `viewer3.html`) and unloads the least recently used; the edge-site transcoder of an unloaded algorithm is dropped from the worker too and reloaded when needed.

### localStorage
Small storage in the visitor's browser. Used for which sections are open. Nothing else is stored there.

---

## 11. The server and the API

### `server_128.py`
The web server of the project, written with the Python standard library only, serving on port 8766 inside the container.

### Port 8766 and 8767 (container) / 80 and 8080 (host)
The container runs two servers: 8766 (the application) and 8767 (a static file server for `fig47/`). Docker maps them to ports 80 and 8080 of the instance.

### `/api/instance`
Returns the compact instance of a sampled graph. No model run, cheap.

### `/api/algos`
The list of algorithms with target, type, FVU and scores (also embedded in the page).

### `/api/weights`, `/api/feature_labels`, `/api/contexts`
Node-site transcoder weights, feature labels and the search for the strongest real examples of a feature. Used by `/demo` only.

### `/api/graph`
Runs the real model on the server for one graph; a fallback when the browser cannot run it.

### `/models/<algo>.json|bin`, `/edge_tc/...`
Model bundles and edge-site transcoder files.

### `/demo`
The previous version of the page. `/demo/` redirects to `/demo`.

### Lazy registry / MODEL_CAP
The server loads a JAX model on first use and keeps at most `MODEL_CAP` (3) of them, evicting the least recently used.

### Warm-up
Loading and compiling `bfs` and `dfs` at start so the first request is fast (skipped if memory is low).

### Semaphore (GRAPH_SEM, CTX_SEM)
Limits on simultaneous heavy work: two model runs at once, one context search at once.

---

## 12. AWS and deployment

### AWS
Amazon Web Services, the cloud provider that hosts the site.

### Region / availability zone
`us-east-1` (N. Virginia); the instance runs in the zone `us-east-1a`.

### EC2 instance
A virtual machine. The site runs on one: ID `i-0877713b5583c8533`, name and project tag `clrs-circuit-tracer`.

### t3.small
The instance type: 2 virtual CPUs and 2 GiB of memory, burstable. About 1.9 GiB usable.

### Burstable / CPU credits
T3 instances earn CPU credits while idle and spend them under load. This instance is set to "standard" credits (it can never spend more than it has earned). Its balance is at the maximum, 576 credits, because the load is tiny.

### AMI (Amazon Linux 2023)
The operating system image the instance was launched from.

### EBS volume (gp3)
The instance's disk: 16 GB, gp3, encrypted.

### Swap
A 2 GiB swap file on the disk, created at launch, as a safety margin for memory spikes.

### Security group
The instance's firewall. `clrs-circuit-tracer-app-sg` allows inbound TCP 80 and 8080 from anywhere; there is no port 22.

### Public IP / Elastic IP
The instance has an automatically assigned public address (98.86.163.27). It is not an Elastic IP, so it changes if the instance is stopped and started (a reboot keeps it).

### SSM (Systems Manager) / Run Command
How administration works without SSH: the workflows send shell commands to the instance through SSM.

### IAM instance profile (`ecsInstanceProfile`)
The permissions the instance itself has: reading images from ECR and receiving SSM commands.

### IMDSv2
The protected way an instance reads its own metadata; the launch workflow requires it (`HttpTokens=required`).

### ECR (Elastic Container Registry)
AWS's store of Docker images. The repository is named `clrs-circuit-tracer`; the image the instance runs is `clrs-circuit-tracer:latest`.

### Image / tag / digest / untagged image
An *image* is the packaged application. A *tag* is a movable label (`latest`). A *digest* is the immutable fingerprint of one image. When a new image is pushed as `latest`, the old one loses the tag and becomes *untagged* but stays in the repository (and costs storage) until deleted.

### Docker / container
Docker packages the application with its dependencies; a *container* is one running copy. The container is named `tracer`.

### `--restart always`
The container restarts by itself after a crash or a reboot.

### Memory limit (`--memory 1500m`)
The container may use at most 1,500 MB of RAM (1,800 MB including swap); beyond that the kernel kills the process (an "OOM kill").

### OOM kill
"Out of memory": the kernel stopped the Python process when it exceeded the container's memory limit. The kernel log of the instance still shows three such events from the earlier memory incident that the lazy loading and cache limits fixed.

### BBR
A TCP congestion-control algorithm that is better on long, lossy paths. The `tune-network` workflow switches the instance from the default (cubic) to BBR.

### CloudFront
AWS's content delivery network. A workflow to put it in front of the site exists but has not been applied.

### HTTPS / Caddy / sslip.io
A workflow to add HTTPS exists but has not been applied: a small Caddy container would obtain a certificate for `98-86-163-27.sslip.io`, a free name that always points to the IP in its name.

### HTTPS-first (Chrome)
Recent Chrome versions warn on, or block, plain `http://` addresses of public sites. The site currently works over HTTP, so Chrome may show a warning page; this is the reason HTTPS is on the list of open items.

---

## 13. GitHub and automation

### Repository
`ani2000/clrs_copy` on GitHub, branch `main`.

### GitHub Actions / workflow
Automation defined by YAML files in `.github/workflows/`. All twelve of them are started by hand (`workflow_dispatch`).

### `workflow_dispatch`
A manual trigger: you press a button (or run `gh workflow run`) and may fill in inputs.

### Confirm gate
Workflows that change something do nothing unless the input `confirm` is exactly `confirmed`.

### Tag guard / expected-ID guard
Workflows that act on the instance first check that exactly one running instance carries both tags `Name=clrs-circuit-tracer` and `Project=clrs-circuit-tracer`, and some also that its ID is the expected one.

### Secrets / variables
The AWS access key id and secret are stored as repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). If a repository variable `AWS_ROLE_ARN` is set, workflows use an OIDC role instead of keys.

### OIDC
A way for GitHub to obtain short-lived AWS credentials without storing keys.

### Runner
The machine GitHub provides to run a workflow (`ubuntu-latest` or `ubuntu-24.04`).

### Pinned action (SHA)
Third-party actions are referenced by commit hash so they cannot change underneath.

### Buildx / build cache
Docker's builder; the workflow uses the GitHub Actions cache to speed up repeated builds.

### Git LFS
Large File Storage: the checkpoints (`*.pkl`, `*.zip`, `*.png` under `checkpoints_v3/`) are stored outside normal git history.

### Fast-forward merge
Moving `main` forward to a branch that already contains it, without a merge commit; used for all merges here.

---

## 14. Testing words

### Test number 1 to 9
The numbered checks in `verification/clrs-edge/` (see `11-testing-and-verification.md`).

### Regression test
A test that the old behaviour is unchanged: test 6 compares the previous graph before and after, byte for byte.

### Truth test
`ui_truth_test.mjs`: reads every number on the page and compares it with a recomputation and with the original training files.

### Sensitivity check
Making sure a test can fail: the truth test was run against a deliberately altered source file and failed as it should.

### Headless Chrome / DevTools protocol / Playwright
Ways to drive a real browser from a script without showing a window.

### Live check
Running a test against the deployed site (with Chrome's HTTPS-first feature switched off for the test browser only).

---

## 15. Alphabetical index

Ablate; About this algorithm; Activation; Adjacency; AMI; Algorithmic reasoning; All (edge set); Attribution graph; Auxiliary loss; AWS; Availability zone; Base rate; Bias node; Brute force; Build id; BBR; Buildx; Burstable; Caddy; Categorical; Channel; Checkpoint; CLEAN; CloudFront; CLRS-30; Collapsible section; Column; Condition; Confirm gate; Container; CPU credits; Cross-check; Decisions changed; Decoder; Dead latent; Delta actual; Delta graph; Dictionary; Digest; Direct (mode); Direction confirmed; Docker; E0; E1; E2; EBS; EC2; ECR; Edge; Edge-site transcoder; Effect; Elastic IP; Encoder; Error node; Error share; ETag; Evaluation set sizes; F1; Fast-forward; Feature; Finite difference; Frozen; Fully connected message passing; FVU; Gap; Golden test; Graph; gzip; Held-out; Hidden size; Hidden state; Hint; Hint feedback; Hint policy; Hint teacher forcing; Hover card; HTTPS-first; IAM instance profile; IMDSv2; Immutable cache; In-browser model; Influence; Input; Inspector; Instance; Intervention; JAX; L0; Latent; Latent table; Leaf; Length; LayerNorm; Linearisation; Live check; localStorage; Logit; Max aggregation; Max-routing; Mask; mask_one; Mechanistic interpretability; Memory limit; Message; Message map; min |w|; MLP; Model bundle; MPNN; n; Named; Node; Node-site transcoder; Node update; node_fts; OIDC; OOD; OOM kill; Output; p; Parity; Pin; Play; Pointer; Port; Pre-activation; Precision; Prefetch; Processor; Pruning; Quality note; Receiver; Recall; Reference implementation; Region; Regression test; Replacement model; Repository; Residency cap; Retention; ReLU; Runner; Sampler; Scalar; Security group; Seed; Sender; Sensitivity check; Shift; Sigma; Simulation; Site; Skeleton; Skip edge; Source node; Sparse dictionary; Speculation; SSM; Stage; Step; Strongest activations; Sum test; Supernode; Superposition; SEVERE; Swap; t3.small; Tag guard; Target; Test graphs; TopK; Total effect; Trajectory; Transcoder; Tier; Unnamed; Validation graphs; Versioned URL; Warm-up; Web Worker; Window; Winner; Winner flip; Workflow.
