/* Hover preview cards for the CLRS-30 circuit tracer.
 * Any element carrying data-tip="<kind>:<arg>" shows a styled card above (or below) it while hovered.
 *   g:<key>      static glossary entry (what it is / why it is here / how it is computed)
 *   node:<i>     a node of the simulated graph at the current step
 *   cell:<i>:<c> a value in the per-node table
 *   an:<name>    a node of the attribution graph (feature, leaf, bias, target, supernode)
 *   ae:<index>   an edge of the attribution graph
 *   fl:<f>       a row of the feature-label table
 *   dyn:<key>    live values (status, target, FVU, F1, step, intervention results)
 * Content is computed at hover time from the viewer's live state (G, W, CUR_ATTR, labelByFeat, ...). */
(function () {
  'use strict';
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const f3 = (x, d = 3) => (x === null || x === undefined || Number.isNaN(+x)) ? '-' : Number(x).toFixed(d);
  const sgn = w => (w > 0 ? '+' : '') + f3(w);
  const CLRS = 'Veličković et al., “The CLRS Algorithmic Reasoning Benchmark”, ICML 2022';
  const CT = 'Ameisen et al., “Circuit Tracing: Revealing Computational Graphs in Language Models”, Anthropic 2025';
  const BIO = 'Lindsey et al., “On the Biology of a Large Language Model”, Anthropic 2025';

  const TARGETS = {
    reach_h: ['reachability hint', 'reach_h[t][v] = 1 once BFS has reached node v from the source by step t, else 0. It grows by one frontier per step.'],
    color: ['DFS colour hint', 'Classic DFS colouring: 0 = white (not yet visited), 1 = grey (discovered, still on the stack), 2 = black (finished). In strongly connected components the colouring runs twice: on the graph, then on its transpose.'],
    d: ['distance-estimate hint', 'd[t][v] is the current best known distance from the source to v. Relaxations lower it until it equals the true shortest-path distance.'],
    key: ['Prim key hint', 'key[t][v] is the weight of the cheapest edge found so far that connects v to the growing spanning tree.'],
    u: ['current-edge endpoint hint', 'Kruskal scans edges in order of weight; u marks (one-hot over nodes) the lower-numbered endpoint of the edge currently being considered (node 0 before the first edge).'],
    topo_head_h: ['topological-order head hint', 'One-hot marker of the node currently at the head of the topological order being built (the most recently finished node in the DFS-based algorithm).'],
    is_cut_h: ['cut-vertex hint', 'is_cut_h[t][v] = 1 once the algorithm has flagged v as an articulation point (removing it would disconnect the graph).'],
    k: ['Floyd–Warshall pivot hint', 'One-hot marker of the current intermediate vertex k: in round k every pair (i,j) is relaxed through k.'],
  };
  const TYPES = {
    mask: 'mask: one independent yes/no per node, decoded with a sigmoid (probability shown).',
    mask_one: 'mask_one: exactly one node is “on”; decoded with a softmax across nodes.',
    categorical: 'categorical: one class per node, decoded with a softmax across classes.',
    scalar: 'scalar: a real number per node, decoded linearly (raw value shown).',
    pointer: 'pointer: each node points at another node, decoded as a softmax over nodes.',
  };

  // ---------------- glossary ----------------
  const GL = {
    algo: ['Algorithm', 'Which of the 12 CLRS-30 graph algorithms to trace. Each has its own separately trained 128-hidden MPNN checkpoint and its own transcoder.',
      'Different algorithms need different internal state (reachability, colours, distances…), so comparing their circuits shows what the same architecture learns for each.', 'Tag after the name: target type and whether the model generalises (CLEAN) or breaks (SEVERE) on bigger graphs.'],
    n: ['n — number of nodes', 'Size of the random graph sampled by CLRS’s own sampler (4–32 here). The checkpoints were trained on CLRS’s default mix of 4, 7, 11, 13 and 16-node graphs, validated on 16 and tested on 64.',
      'Going above 16 probes out-of-distribution size generalisation, which is exactly what the CLRS benchmark tests.', 'Changes apply immediately; neighbouring values are pre-computed in the background.'],
    p: ['p — edge probability', 'Each possible edge is included independently with probability p: an Erdős–Rényi random graph, directed/acyclic/weighted as the algorithm requires (CLRS samplers). Training drew p from 0.1–0.9 (halved for articulation points, bridges and Kruskal).',
      'Sparse graphs give long, chain-like executions; dense graphs finish in few steps. The circuits used can differ between the two.', 'Changes apply immediately.'],
    seed: ['seed', 'Random seed for the graph sampler. Same (algorithm, n, p, seed) always gives the same graph, so any finding can be reproduced.',
      'If a sampled graph is degenerate (fewer than 3 algorithm steps), the seed is shifted automatically and the used seed is shown.', 'Use the spinner arrows to step through graphs; ±1 is pre-computed.'],
    new: ['New random graph', 'Samples a fresh graph and runs the real trained model on it.',
      'A circuit seen on one graph is only a hypothesis; checking it on many graphs is how you find out whether it generalises (Figure 4 vs Figure 7 in this project).', 'Two or three next graphs are always pre-computed, so this is usually instant.'],
    hidden: ['hidden_dim = 128', 'Every node carries a 128-number state vector. It is NOT 128 layers: the MPNN has one processor whose weights are reused at every algorithm step, like the body of a loop.',
      'This vector is where the model stores what it has worked out so far (reached? distance? colour?), mixed together across many dimensions. Decoding it is the point of the transcoder.', 'CLRS “MPNN” processor: m_ij = MLP(relu(W₁z_i + W₂z_j + W_e e_ij + W_g g)), aggregated by a max over ALL nodes (the processor is fully connected; the graph’s edges enter only as edge features), then h′ = LayerNorm(relu(O₁z + O₂m)).'],
    simulation: ['Simulation', 'The sampled graph at the selected algorithm step, coloured by the ground-truth hint for that step (from CLRS’s reference implementation).',
      'This is the “input and context” side: you pick a node and a step here, and the attribution graph explains the model’s prediction for that node at the next step.', 'Orange ring = source node (for algorithms that have one).'],
    play: ['Play / Pause', 'Advances one algorithm step every 1.6 s and re-traces the attribution graph for the selected node at each step.',
      'Watching the same node over time shows which features switch on as the algorithm progresses (e.g. a “frontier” feature right before a node becomes reached).', 'Changing the graph, n, p, seed or algorithm stops playback immediately.'],
    step: ['Algorithm step', 'CLRS records the textbook algorithm’s state after every step (the hints). The model runs one message-passing step per hint step, predicting step t+1 from step t.',
      'Attributions are computed per step because the same processor is reused, so “which step” plays the role that “which layer” plays in a transformer.', ''],
    attrgraph: ['Attribution graph', 'A graph of direct linear effects that explains one prediction. Columns are the graph’s nodes, rows are algorithm steps (newest at the top). Each step has a row of transcoder features (circles, one set per graph node) and a row of encoder leaves (squares: the inputs and fed-back hints read at that step). Edges cross columns because every node receives messages from every other node. The window covers the last 3 steps; older state enters as one leaf per node.',
      'This is the method of ' + CT + ', used in ' + BIO + ' to trace multi-step reasoning. Here it is applied to a graph neural network executing a textbook algorithm.', 'To make the model linear for one input, the top-k feature choice is frozen at its actual value (the paper freezes attention patterns and normalisation denominators for the same reason). Edge = source activation × (source’s decoder vector · the gradient of the destination).'],
    pruned: ['Pruned', 'Keeps the nodes carrying 80% of the influence on the predictions (influence = probability-weighted share of all paths into the targets), then the edges carrying 98% of the influence among them — the pruning rule of the Circuit Tracing paper.',
      'Full graphs have thousands of edges; pruning leaves the few paths that carry most of the effect.', 'The statistic below the graph reports how much influence the pruned graph keeps.'],
    all: ['All', 'Shows the strongest edges of the full linearised computation (up to 6,000, by |weight|), unpruned.',
      'Use it to check that pruning did not hide something important.', ''],
    bynode: ['By node', 'Every feature appears once per graph node and step where it is active: column = graph node, row = step. Its label (e.g. f37) is the transcoder feature index.',
      'Keeps full detail: you see exactly when and at which node each feature fired.', ''],
    supernodes: ['Supernodes', 'Features that share the same human label are merged into one box, and their edges are summed.',
      'The paper groups related features into “supernodes” by hand to make a graph readable. Here the grouping is automatic, by the predicate that best describes each feature.', 'Aggregation only: the numbers are exact sums of the underlying edges.'],
    minw: ['min |w|', 'Hides edges whose |weight| is below this absolute value.',
      'A quick way to look at only the dominant paths.', ''],
    featlabels: ['Feature labels', 'Each transcoder feature is scored against hand-written predicates on the true hints (e.g. “newly 1” = this node became reached this step).',
      'Labels turn feature indices into readable meanings. The score tells you how much to trust a label; a low-F1 label is a guess.', 'P = precision, R = recall, F1 = their harmonic mean, over a fixed labelling dataset.'],
    precision: ['Precision (P)', 'Of all the sites (node, step) where the feature fires, the fraction where the predicate is true.', 'High P means that when the feature is on, the label is almost always correct.', ''],
    recall: ['Recall (R)', 'Of all the sites where the predicate is true, the fraction where the feature fires.', 'High R means the feature catches almost every case of the concept. Features can be precise but partial (one of several features covering a concept).', ''],
    f1: ['F1', 'Harmonic mean of precision and recall: 2PR / (P + R).', 'A single number for how well the label describes the feature.', ''],
    firesOn: ['Fires on', 'How many (node, step) sites in the labelling dataset activate this feature, out of all sites.', 'Very rare features get the label “(rare)” because there is too little data to name them reliably.', ''],
    featdetail: ['Feature detail', 'Everything known about the clicked feature: its label and scores, its edges in this graph, and real dataset examples where it fires most strongly.',
      'Following the paper, a feature is interpreted from its top-activating examples as well as its label.', ''],
    intervention: ['Intervention', 'Clamps the pinned feature (−1× reverses it, ×0 removes it, ×2 amplifies it) and re-runs the real model from that step on, with every nonlinearity and the model’s own hint feedback recomputed.',
      'An attribution graph is a hypothesis about causes. The paper validates it by perturbing features and checking the prediction moves as the graph predicts; this does the same.', '“Δ graph” = (factor − 1) × activation × total effect; “Δ actual” = what the real network did. The count of flipped ReLU / max-routing decisions tells you how far outside the frozen-linear regime the perturbation went.'],
    ivZero: ['×0 — ablate', 'Sets the feature’s activation to zero and re-runs the real model.',
      'Tests necessity: does the prediction depend on this feature?', ''],
    ivNeg: ['−1× — invert', 'Reverses the feature’s activation and re-runs the real model.',
      'A stronger perturbation than ablation; the paper uses negative multiples to make effects easier to see.', ''],
    fvu: ['Transcoder FVU', 'Fraction of variance unexplained: how much of the real 128-number update the transcoder fails to reconstruct (0 = perfect).',
      'The attribution graph explains the transcoder’s version of the model (the “replacement model”). The lower the FVU, the more faithful that explanation is to the real network.', 'FVU = ‖h − ĥ‖² / ‖h − mean(h)‖² on held-out data.'],
    transcoder: ['Transcoder', 'A sparse dictionary (F = 192 features, at most K = 8 active) trained to read the node-update input (256 numbers: this node’s encoded inputs + previous state; plus 128 numbers of incoming messages) and reproduce the 128-number new state.',
      'The raw state is dense and entangled. Forcing it through a few active features makes each feature carry a specific, nameable meaning.', 'The paper uses cross-layer transcoders across a deep transformer. This MPNN has one processor reused each step, so one transcoder per algorithm plays the same role.'],
    cleanSevere: ['CLEAN / SEVERE', 'Based on the gap between the validation score (16-node graphs) and the out-of-distribution test score (64-node graphs). CLEAN models keep most of their accuracy (gap ≤ 0.074); SEVERE ones lose a lot (gap ≥ 0.46).',
      'CLRS’s central question is whether a model learned the algorithm or a size-specific shortcut. A circuit from a SEVERE model may explain a shortcut, not the algorithm.', 'Source: Veličković et al., “The CLRS Algorithmic Reasoning Benchmark”, ICML 2022; numbers from this project’s results_v3/summary_v3.json.'],
    valtest: ['val / test score (OOD)', 'CLRS’s output score for this checkpoint (accuracy for pointer and categorical outputs, F1 for mask outputs): validation on 16-node graphs, test on 64-node graphs (the standard CLRS out-of-distribution split), as recorded by the training run.',
      'The gap is the most informative number here: it says whether the circuits below are an algorithm or a memorised pattern.', 'Re-scoring the checkpoints on fresh graphs reproduces these numbers (see the verification report in the repository).'],
    target: ['Target', 'The hint this tool explains for the selected algorithm, with its type.', 'The paper explains one output token at a time; here the “token” is one node’s hint value at the next step.', ''],
    leafNodeFts: ['node_fts@t:j — input leaf', 'The encoded inputs and hints of node j at step t (position, source flag, the previous hint values…), after CLRS’s linear encoders.', 'Leaves are where the explanation stops: they are observed data, not computed features.', ''],
    leafMsg: ['msg@t:j — message leaf', 'The aggregated message node j received at step t: a max over all other nodes of the message MLP.', 'Bundling it as a leaf keeps the graph per-node. It means influence from neighbours is shown as one input rather than traced node by node.', 'A limitation of this tool versus the paper, which traces through attention to other token positions.'],
    bias: ['bias', 'The decoder’s constant term: what the prediction would be with every feature and leaf at zero.', 'Included so the edges into the target add up exactly to the logit being explained.', ''],
    legendReach: ['Reach/state feature', 'A feature whose best label is a state predicate of the target hint (e.g. newly reached, stays reached, distance decreases).', '', ''],
    legendSource: ['Source feature', 'A feature labelled as tracking whether a node is the source.', 'Many algorithms start from the source, so the model needs a stable “I am the source” signal.', ''],
    legendCat: ['Categorical-class feature', 'A feature labelled by a class transition of a categorical hint (e.g. becomes class 1 = turns grey in DFS). Tinted with that class’s colour.', '', ''],
    legendUnlabeled: ['Unlabeled / rare', 'No predicate describes this feature well, or it fires too rarely to name. It can still carry real influence.', 'The paper also finds many features that resist a clean label; they are shown, not hidden.', ''],
    legendLeaf: ['Leaf / bias', 'Grey boxes are inputs to the explanation (node_fts, msg) or the constant bias.', '', ''],
    legendPos: ['Positive edge', 'Blue: the source pushes the destination up (toward the predicted value / class).', '', ''],
    legendNeg: ['Negative edge', 'Red: the source pushes the destination down (away from the predicted value / class).', '', ''],
    inBrowser: ['In-browser model', 'The trained 128-hidden network runs inside your browser (a JavaScript port of the CLRS encoders, processor and hint decoders, checked against the original JAX model on all 12 algorithms to ~1e-4). The server only samples the graph.',
      'Removes the network round trip for the heavy part, so changes to n, p and seed appear almost instantly.', 'Until the weight bundle (1–2 MB, cached) has downloaded, the server runs the model instead.'],
    howItWorks: ['How this tool works', '1) CLRS samples a graph and runs the textbook algorithm to get hints. 2) The trained MPNN executes the same algorithm step by step. 3) A transcoder rewrites each step’s node update as a few interpretable features. 4) Freezing which features are active makes one prediction a linear sum, shown as an attribution graph. 5) Clamping features checks that the graph is causally right.',
      'Accuracy says whether the model is right; this says how it gets there, and interventions say whether that story is true.', 'Method: ' + CT + ' / ' + BIO + '. Benchmark: ' + CLRS + '.'],
  };

  Object.assign(GL, {
    avRowO: ['Predictions of this step', 'The target logits being explained: the algorithm’s main hint for the selected node at the next step, and (for algorithms with a parent pointer) its pointer prediction. Filled chips are the model’s top choice.', 'Every path in the graph ends at one of these.', ''],
    avRowF: ['Features at this step', 'For every graph node, the transcoder features active at its node update in this step (≤8 each), coloured by the predicate that best describes them. The hatched square beside them is that node’s error node.', 'Features are the interpretable units the explanation is written in.', 'Size ∝ √activation.'],
    avRowL: ['Leaves at this step', 'The values the encoders read at this step, per node: inputs (pos, A, adj, s…) and hints the model fed back from its previous prediction; edge-located values (A[i,j]) sit under the first node of the pair.', 'Leaves are where the explanation stops.', ''],
    avg_state: ['Target-state feature', 'Best described by a predicate on the target hint itself: newly/still reached, distance increases/decreases/stays, etc.', '', ''],
    avg_class: ['Class-change feature', 'Best described by a categorical transition of the target hint (e.g. “becomes class 1” = turns grey in DFS).', '', ''],
    avg_source: ['Source feature', 'Fires on (or tracks) the start node.', 'Algorithms that start from a source need a stable “I am the source” signal.', ''],
    avg_neg: ['Negated-state feature', 'Best described by the absence of a state (“not stays 0” fires on every node that is not staying unreached, …).', '', ''],
    avg_none: ['Unlabelled feature', 'No predicate describes it well, or it is too rare to name. It can still carry real influence.', '', ''],
    avError: ['Error node', 'The part of a node’s new hidden state that the transcoder did not reconstruct. Adding it back makes the replacement model reproduce the real network exactly (Circuit Tracing).', 'Its outgoing weight shows how much of the mechanism lies outside the features.', ''],
    avAct: ['Activation', 'The feature’s value at this node and step (after ReLU and top-k).', '', ''],
    avInfluence: ['Influence on targets', 'How much of the targets’ (probability-weighted) inputs flow through this node, counting all paths: the quantity used to prune the graph (Circuit Tracing’s influence).', 'The pinned feature at load time is the most influential one.', ''],
    avTotalEffect: ['Total effect', 'Change in each target logit per unit change of this feature’s activation, summed over every path in the frozen-linear graph (direct and indirect).', 'This is what an intervention on the feature should do, if the linear picture is right.', ''],
    ivDbl: ['×2 — amplify', 'Doubles the feature’s activation and re-runs the real model.', 'The opposite test to ablation: the prediction should move the other way.', ''],
  });

  // wording that differs for the edge-site graph (latents on sender -> receiver pairs); everything else is shared
  const GL_EDGE = {
    avRowF: ['Latents at this step', 'For every receiver node (column), the transcoder latents active on the pairs sender → receiver in this step (8 per pair, all senders), coloured by what the pair is in this graph: a real edge, a non-edge or a self pair. The hatched square is the error node of a pair. Label: sender : latent.', 'Latents are the units this explanation is written in. They have no names yet.', 'Size ∝ √activation.'],
    avRowL: ['Leaves at this step', 'The values the encoders read at this step: inputs (pos, A, adj, s…) and hints the model fed back from its previous prediction, per node; edge-located values (A[i,j]) sit under the sender. The earlier-state leaf is the hidden state from before the window.', 'Leaves are where the explanation stops.', ''],
    avError: ['Error node', 'The part of a pair’s message y that the transcoder did not reconstruct. Adding it back makes the replacement model reproduce the real network exactly (Circuit Tracing).', 'Its outgoing weight shows how much of the mechanism lies outside the latents.', ''],
    avAct: ['Activation', 'The latent’s value on this pair and step (after ReLU and top-k).', '', ''],
    avTotalEffect: ['Total effect', 'Change in each target logit per unit change of this latent’s activation, summed over every path in the frozen-linear graph (direct, skip and indirect). Checked against finite differences in the test suite.', 'The intervention panel compares this prediction with the real model.', ''],
    attrgraph: ['Attribution graph', 'A graph of direct linear effects that explains one prediction. Columns are the receiving nodes, rows are algorithm steps (newest at the top). Each step has a row of transcoder latents (circles, 8 per sender → receiver pair) and a row of encoder leaves (squares: the inputs and fed-back hints read at that step). The window covers the last 3 steps; older state enters as one leaf per node.',
      'This is the method of ' + CT + ', used in ' + BIO + ' to trace multi-step reasoning. Here it is applied to a graph neural network executing a textbook algorithm.', 'To make the model linear for one input, the max-routing winners, the LayerNorm σ and the ReLU pattern of the node update are frozen at their real values, and every pair has an error node. Edge = source activation × (its decoder vector · the gradient of the destination).'],
    bynode: ['By node', 'Every latent appears once per pair and step where it is active: column = receiving node, row = step. Its label is sender : latent index.', 'Keeps full detail: you see exactly when and on which pair each latent fired.', ''],
    supernodes: ['Supernodes', 'All latents of one sender → receiver pair at one step are merged into one box, and their edges are summed.', 'Shows which messages (which sender to which receiver) carry the computation.', 'Aggregation only: the numbers are exact sums of the underlying edges; edges between latents of the same box are dropped.'],
    featdetail: ['Latent detail', 'What is known about the clicked latent: its pair, activation, influence, where else it fires in this graph, its total effect on the targets and its edges.', 'The latents have no names yet, so only facts measured on this graph are shown.', ''],
    intervention: ['Intervention', 'Clamps the pinned latent (−1× reverses it, ×0 removes it, ×2 amplifies it): the message of its pair changes by (factor − 1) × activation × decoder vector, and the real model is re-run from that step on with every nonlinearity and its own hint feedback recomputed.',
      'An attribution graph is a hypothesis about causes; clamping tests it.', '“Δ graph” = (factor − 1) × activation × total effect; “Δ actual” = what the real network did. The graph freezes σ, winners and TopK sets, so in our tests its predicted direction matched the real model 69% of the time (44–77% depending on the algorithm).'],
    ivZero: ['×0 — ablate', 'Sets the latent’s activation to zero and re-runs the real model.', 'Tests necessity: does the prediction depend on this latent?', ''],
    ivNeg: ['−1× — invert', 'Reverses the latent’s activation and re-runs the real model.', 'A stronger perturbation than ablation.', ''],
    ivDbl: ['×2 — amplify', 'Doubles the latent’s activation and re-runs the real model.', 'The opposite test to ablation.', ''],
    transcoder: ['Edge-site transcoder', 'A sparse dictionary (D = 1,280 latents, at most k = 8 active) trained to read the message input of one sender → receiver pair (128 numbers: the summed, pre-ReLU message) and reproduce the 128-number message the message MLP outputs.',
      'The raw message is dense and entangled; forcing it through a few active latents is what lets a circuit be read.', 'The MPNN has one processor reused at every step, so one transcoder per algorithm covers all steps. It was trained on the real activations of that algorithm’s MPNN.'],
    howItWorks: ['How this tool works', '1) CLRS samples a graph and runs the textbook algorithm to get hints. 2) The trained MPNN executes the same algorithm step by step; every ordered pair of nodes sends a message. 3) A transcoder rewrites each message as a few sparse latents. 4) Freezing the max-routing winners, the LayerNorm σ and the ReLU pattern makes one prediction a linear sum, shown as an attribution graph. 5) Clamping latents tests whether the graph is causally right.',
      'Accuracy says whether the model is right; this says how it gets there, and interventions say whether that story is true.', 'Method: ' + CT + ' / ' + BIO + '. Benchmark: ' + CLRS + '.'],
    bias: ['bias', 'All constant terms on the frozen path (encoder, message-site, node-update, hidden-state-decoder and transcoder biases and offsets).', 'Included so the edges into a node add up exactly to its value; the page checks that for every node.', ''],
    etcPanel: ['Model & edge-site transcoder', 'The trained MPNN for this algorithm and the measured quality of its edge-site transcoder, copied from the training run’s report.', '', ''],
    etcSite: ['Transcoder site', 'It reads the message input s of every sender → receiver pair and rebuilds the message y. D latents, of which k are active for each pair.', '', ''],
    etcTier: ['Tier', 'A: the baseline MPNN generalises to 64 nodes (OOD F1 ≥ 0.90), so the replacement is judged at 16 and 64 nodes. B: it generalises badly, so the replacement is judged at 16 nodes only. C: the baseline itself is weak (validation F1 below 0.95): results are indicative only.', '', ''],
    etcFvu: ['Reconstruction error (FVU)', 'Fraction of the message variance the latents do not rebuild: on validation graphs (16 nodes), on the training rows, and on 64-node test graphs (measured for tier A only). 0 is perfect.', 'The error node carries the rest, so the graph is exact for the real run whatever the FVU.', ''],
    etcStep: ['Error by step', 'Reconstruction error of the messages at each algorithm step on validation graphs. Later steps are usually harder.', '', ''],
    etcDead: ['Dead latents and sparsity', 'Latents that never fire on the training rows, and the average number of active latents per pair (at most k).', 'Dead latents are simply unused dictionary entries.', ''],
    etcRet: ['Replacement fidelity', 'The share of the original network’s score that is kept when the transcoder replaces the message MLP at every step and everything downstream is recomputed (winners included). “x of y” is the replaced score of the original score; a share above 100% can happen by chance.', 'This measures whether the sparse code is a faithful stand-in for the real computation.', ''],
    etcFlips: ['Winner flips', 'How often the replacement changes which sender wins a channel of the max when it runs on its own. A flip needs twice the reconstruction error to exceed the gap between the best and second-best sender; that bound was checked with zero violations.', 'The attribution graph uses the real winners, so it is not affected.', ''],
    etcHid: ['One-step hidden-state error', 'Relative error of the next hidden state when one step’s messages are replaced, starting from the true input; mean over steps.', '', ''],
    etcTrain: ['Training', 'Optimisation steps, number of pair rows used, and minutes of training for this algorithm’s edge-site transcoder.', '', ''],
    demoLink: ['Previous demo', 'The earlier version of this tool, kept as it was: attribution graphs built from the node-site transcoder, with labelled features, top-activating examples and the node table.', '', ''],
    bymeaning: ['By meaning', 'All latents that were found to mean the same thing (for example “frontier”) at one step are merged into one box, and their edges are summed. Latents without a name go into “unnamed”.', 'Shows which kinds of message carry the computation. Names are only given to latents whose match holds up on graphs they were not fitted on.', 'Aggregation only: the numbers are exact sums; edges between latents of the same box are dropped.'],
    topLatents: ['Top latents at each step', 'For every step in the window, the six latents of the pruned graph with the most influence on the predictions: the pair they sit on, their activation, what they were found to mean and what they do to the receiver. Click a row to pin the latent.', 'Step by step, this is the short list of what mattered.', ''],
    latDict: ['Latent dictionary', 'The latents of this algorithm that can be named, with the evidence: precision, recall and F1 of the condition on graphs that were not used to pick it (tier A: F1 ≥ 0.8, tier B: F1 ≥ 0.5), how often the latent fires, and its average effect on the receiver’s prediction.', 'Most latents have no simple description; those are left unnamed on purpose rather than guessed.', 'Measured on 60 fresh graphs per algorithm, 30 to choose the condition and 30 to score it.'],
    avg_pe: ['Latent on a real edge', 'The pair sender → receiver is an edge of the input graph (adj > 0). Colour says what the pair is in this graph, not what the latent means.', '', ''],
    avg_pn: ['Latent on a non-edge pair', 'The pair is not an edge of the input graph. The MPNN still computes a message for it (it uses an all-ones adjacency), so every pair has latents.', '', ''],
    avg_ps: ['Latent on a self pair', 'Sender and receiver are the same node.', '', '']
  };
  function card(title, what, why, how, extra) {
    let h = `<div class="hc-title">${title}</div>`;
    if (what) h += `<div class="hc-body">${what}</div>`;
    if (extra) h += extra;
    if (why) h += `<div class="hc-why"><b>Why:</b> ${why}</div>`;
    if (how) h += `<div class="hc-how">${how}</div>`;
    return h;
  }
  const rows = kv => `<table class="hc-kv">${kv.filter(Boolean).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>`;

  // ---------------- live state helpers (all globals from viewer3.html) ----------------
  const has = n => { try { return eval('typeof ' + n) !== 'undefined'; } catch (e) { return false; } };
  const S = () => ({ G: has('G') ? G : null, W: has('W') ? W : null, A: has('CUR_ATTR') ? CUR_ATTR : null,
                     L: has('labelByFeat') ? labelByFeat : {}, step: has('curStep') ? curStep : 0, algo: has('curAlgo') ? curAlgo : '' });
  function labelOf(f) { const r = S().L[f]; return r && r.label ? r : null; }
  function featsAt(t, j) {
    const { G, W } = S(); if (!G || !G.z || !W || !W.w_enc || t < 0 || t >= G.z.length) return null;
    return encode([...G.z[t][j], ...G.msgs[t][j]]);
  }
  function topFeatures(t, j, k = 3) {
    const fa = featsAt(t, j); if (!fa) return [];
    return fa.map((v, i) => [v, i]).filter(x => x[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, k);
  }
  function hintVal(t, i) { const { G } = S(); if (!G || !G.target_true) return null; const r = G.target_true[Math.min(t, G.target_true.length - 1)]; return r ? r[i] : null; }
  function fmtHint(v) { const { W } = S(); if (v === null || v === undefined) return '-'; return W && (W.target_type === 'categorical' || W.target_type.startsWith('mask')) ? String(Math.round(v)) : f3(v, 2); }
  function describeTarget() {
    const { W } = S(); if (!W) return '';
    const t = TARGETS[W.target]; return t ? `<b>${esc(W.target)}</b> — ${t[1]}<br><span class="hc-muted">${TYPES[W.target_type] || ''}</span>` : esc(W.target);
  }

  // ---------------- dynamic renderers ----------------
  const R = {
    g(key) { const e = (window.AV_EDGE_ON && window.AV_EDGE_ON() && GL_EDGE[key]) || GL[key]; return e ? card(e[0], e[1], e[2], e[3]) : null; },

    node(arg) {
      const { G, W, step } = S(); if (!G || !W) return null;
      const i = +arg, st = Math.min(step, G.length - 2);
      let deg = 0; for (let j = 0; j < G.n; j++) if (j !== i && (G.adj[i][j] > 0 || G.adj[j][i] > 0)) deg++;
      const pred = (has('NODE_PRED') && NODE_PRED[i] !== undefined) ? NODE_PRED[i] : '-';
      const tops = st >= 0 ? topFeatures(st, i) : [];
      const tf = tops.length ? tops.map(([v, f]) => `f${f} <span class="hc-muted">${esc(labelOf(f) ? labelOf(f).label : '(unlabeled)')}</span> ${f3(v, 2)}`).join('<br>') : '-';
      return card(`Node ${i}${i === G.source ? ' · source' : ''}`, null,
        'Click the node to build the attribution graph for the model’s prediction here, at this step.', null,
        rows([['degree', deg], [`${esc(W.target)} at step ${step}`, fmtHint(hintVal(step, i))],
              st >= 0 ? [`true ${esc(W.target)} at step ${st + 1}`, fmtHint(hintVal(st + 1, i))] : null,
              st >= 0 ? ['model’s prediction', esc(pred)] : null,
              (window.AV_EDGE_ON && window.AV_EDGE_ON()) ? null : ['strongest active features', tf]]));
    },

    cell(arg) {
      const { G, W, step } = S(); if (!G || !W) return null;
      const [i, col] = arg.split(':'); const st = Math.min(step, G.length - 2);
      const m = {
        node: [`Node ${i}`, `Row for node ${i}. Hover the other cells for their meaning.`],
        prev: [`${esc(W.target)}(${st}) — current state`, `The true hint value of node ${i} at step ${st}, from CLRS’s reference implementation. This is part of what the model is given.`],
        pred: [`pred ${esc(W.target)}(${st + 1}) — model’s prediction`, `The real model’s prediction for step ${st + 1}, from the in-browser run of the checkpoint ${window.AV_EDGE_ON && window.AV_EDGE_ON() ? '(shown once the step has been traced)' : '(grey italic until the step has been traced: a transcoder approximation)'}. ${TYPES[W.target_type] || ''}`],
        true: [`true(${st + 1}) — correct answer`, `The reference algorithm’s value at step ${st + 1}. Compare with the prediction to see where the model is right or wrong.`],
      }[col];
      return m ? card(m[0], m[1], col === 'pred' ? 'Explaining the replacement model is what makes attribution possible; its agreement with the true value says whether the explanation concerns correct behaviour.' : null) : null;
    },

    an(name) {
      if (window.avCard) { const h = window.avCard(name); if (h) return h; }
      const { A, W } = S(); if (!A || !W) return null;
      if (name === A.targetName || name === 'target') {
        return card(`Target · ${esc(A.targetName)}`, describeTarget(), 'This is the single prediction the whole graph explains. Every edge ultimately feeds into this box.', null,
          rows([['value explained', esc(interpretLogit(A))], ['edges in full graph', A.eAll], ['coverage of pruned graph', f3(A.coverage * 100, 1) + '% of |influence|']]));
      }
      if (name === 'bias') { const e = A.edges.find(x => x[0] === 'bias'); return card('bias', GL.bias[1], GL.bias[2], null, rows([['contribution', e ? sgn(e[2]) : '-']])); }
      const toT = A.edges.filter(e => e[0] === name && e[1] === A.targetName).reduce((s, e) => s + e[2], 0);
      const m = name.match(/^f(\d+)@(\d+):(\d+)$/);
      if (m) {
        const f = +m[1], t = +m[2], j = +m[3], lab = labelOf(f) || {};
        const fa = featsAt(t, j), act = fa ? fa[f] : null;
        const inE = A.edges.filter(e => e[1] === name), outE = A.edges.filter(e => e[0] === name);
        return card(`Feature ${f} · step ${t} · node ${j}`,
          `<b>${esc(lab.label || '(unlabeled)')}</b>${lab.f1 !== undefined ? ` <span class="hc-muted">P ${f3(lab.precision, 2)} · R ${f3(lab.recall, 2)} · F1 ${f3(lab.f1, 2)}</span>` : ''}`,
          'An active transcoder feature: one of the ≤8 directions the update at this node and step was decomposed into. Click it for full detail and interventions.',
          t === A.step ? 'Active at the step being predicted: writes directly into the new state the decoder reads.' : 'Active one step earlier: acts through the previous hidden state.',
          rows([['activation', f3(act, 3)], ['direct effect on target', sgn(toT)], ['incoming / outgoing edges', `${inE.length} / ${outE.length}`]]));
      }
      if (name.startsWith('node_fts')) return card(esc(name), GL.leafNodeFts[1], GL.leafNodeFts[2], null, rows([['direct effect on target', sgn(toT)]]));
      if (name.startsWith('msg')) return card(esc(name), GL.leafMsg[1], GL.leafMsg[2], GL.leafMsg[3]);
      // supernode
      const inAgg = has('SHOWN_EDGES') ? SHOWN_EDGES.filter(e => e[0] === name).reduce((s, e) => s + e[2], 0) : 0;
      const members = new Set(A.edges.flatMap(e => [e[0], e[1]]).filter(n => supernodeKey(n, A) === name));
      return card(`Supernode · ${esc(name)}`, `Merges ${members.size} feature/leaf instance(s) sharing this label across steps.`, GL.supernodes[2], null,
        rows([['summed weight to next node', sgn(inAgg)]]));
    },

    ae(idx) {
      if (window.avEdgeCard) { const h = window.avEdgeCard(idx); if (h) return h; }
      const { A } = S(); if (!A || !has('SHOWN_EDGES')) return null;
      const e = SHOWN_EDGES[+idx]; if (!e) return null;
      const [s, d, w] = e, total = A.edges.reduce((a, x) => a + Math.abs(x[2]), 0) || 1;
      let how = 'Edge = activation of the source × (source’s decoder vector · the gradient of the destination), with the top-k choice frozen.';
      if (/^node_fts|^msg/.test(s)) how = 'Edge = the leaf’s values · the destination feature’s encoder direction (how strongly this input drives the feature).';
      if (s === 'bias') how = 'The decoder’s constant term.';
      return card(`${esc(s)} → ${esc(d)}`, `${w > 0 ? 'Pushes the destination <b>up</b>' : 'Pushes the destination <b>down</b>'} (toward ${w > 0 ? 'more' : 'less'} of what is predicted).`,
        'Edges are direct linear effects, so they add up: all edges into a node sum to its value (minus nothing hidden).', how,
        rows([['weight', sgn(w)], ['share of total |influence|', f3(100 * Math.abs(w) / total, 1) + '%']]));
    },

    fl(arg) {
      const lab = labelOf(+arg); if (!lab) return null;
      const { W } = S();
      return card(`Feature ${arg} · ${esc(lab.label)}`, `Best-matching predicate on the true <b>${esc(W ? W.target : '')}</b> hint over the labelling dataset.`,
        lab.f1 > 0.8 ? 'High F1: this label is a reliable reading of what the feature does.' : lab.f1 > 0.5 ? 'Moderate F1: the label captures part of the feature’s behaviour.' : 'Low F1: treat this label as a weak hint, not a definition.', null,
        rows([['precision', f3(lab.precision)], ['recall', f3(lab.recall)], ['F1', f3(lab.f1)], ['fires on', `${lab.n_fire}${has('FL') && FL ? ' / ' + FL.total_sites : ''} sites`]]));
    },

    dyn(key) {
      const { G, W, A, algo } = S();
      if (key === 'status') {
        if (!G) return card('Status', 'Waiting for the model.', null, null);
        return card('Current run', G._local ? GL.inBrowser[1] : 'The model ran on the server (the in-browser copy is still loading, or unavailable).', GL.inBrowser[2], null,
          rows([['algorithm', esc(algo)], ['nodes', G.n], ['hint steps', G.length], ['seed', G.seed !== undefined ? G.seed : '-'], ['model ran', G._local ? 'in your browser' : 'on the server']]));
      }
      if (key === 'target') return W ? card('Target hint', describeTarget(), GL.target[2], null) : null;
      if (key === 'fvu') return W ? card(GL.fvu[0], GL.fvu[1], GL.fvu[2], GL.fvu[3], rows([['this algorithm', f3(W.fvu, 4)], ['in words', W.fvu < 0.01 ? 'near-perfect reconstruction' : W.fvu < 0.05 ? 'good reconstruction' : 'noticeable reconstruction error']])) : null;
      if (key === 'f1') {
        const gt = has('SUMMARY_V3') ? SUMMARY_V3[algo] : null;
        return card(GL.valtest[0], GL.valtest[1], GL.valtest[2], GL.cleanSevere[3],
          gt ? rows([['val F1', f3(gt.val)], ['test F1 (OOD)', f3(gt.test)], ['gap', f3(gt.gap)], ['class', gt.cls]]) : '');
      }
      if (key === 'step') {
        if (!G) return null;
        return card(GL.step[0], GL.step[1], GL.step[2], null, rows([['current step', has('curStep') ? curStep : 0], ['last step', G.length - 1]]));
      }
      if (key === 'iv') return has('LAST_IV') && LAST_IV ? card('Intervention result', `Clamped <b>${esc(LAST_IV.name)}</b> ×${LAST_IV.factor}.`,
        'If the change has the sign the edge predicted, the attribution is confirmed for this input; if not, the linear picture missed something.', null,
        rows([['baseline', f3(LAST_IV.base)], ['after clamp', f3(LAST_IV.after)], ['change', sgn(LAST_IV.after - LAST_IV.base)], ['predicted direction', LAST_IV.predicted || '-']])) : null;
      return null;
    },
  };

  // ---------------- card element & behaviour ----------------
  let cardEl = null, current = null, showTimer = null;
  function ensureCard() {
    if (cardEl) return cardEl;
    cardEl = document.createElement('div');
    cardEl.className = 'hc'; cardEl.setAttribute('role', 'tooltip'); cardEl.hidden = true;
    document.body.appendChild(cardEl);
    return cardEl;
  }
  function place(target) {
    const c = ensureCard(), r = target.getBoundingClientRect(), vw = window.innerWidth, vh = window.innerHeight, gap = 8;
    c.style.left = '0px'; c.style.top = '0px';
    const cw = c.offsetWidth, ch = c.offsetHeight;
    let top = r.top - ch - gap, below = false;
    if (top < 6) { top = r.bottom + gap; below = true; }
    if (top + ch > vh - 6) top = Math.max(6, vh - ch - 6);
    let left = r.left + r.width / 2 - cw / 2;
    left = Math.max(6, Math.min(left, vw - cw - 6));
    c.style.left = left + 'px'; c.style.top = top + 'px';
    c.classList.toggle('below', below);
    c.style.setProperty('--arrow-x', Math.max(12, Math.min(cw - 12, r.left + r.width / 2 - left)) + 'px');
  }
  function render(el) {
    const spec = el.getAttribute('data-tip') || ''; const k = spec.indexOf(':');
    const kind = spec.slice(0, k), arg = spec.slice(k + 1);
    try { return R[kind] ? R[kind](arg) : null; } catch (e) { return null; }
  }
  function show(el) {
    const html = render(el); if (!html) return hide();
    const c = ensureCard(); c.innerHTML = html; c.hidden = false; place(el);
    current = el; el.classList.add('hc-active');
    if (el.hasAttribute('data-an')) focusEdges(el.getAttribute('data-an'));
  }
  function hide() {
    clearTimeout(showTimer);
    if (cardEl) cardEl.hidden = true;
    if (current) current.classList.remove('hc-active');
    current = null; focusEdges(null);
  }
  function focusEdges(name) {
    const svg = document.getElementById('attribSvg'); if (!svg) return;
    svg.querySelectorAll('line[data-s]').forEach(l => {
      const on = !name || l.getAttribute('data-s') === name || l.getAttribute('data-d') === name;
      l.classList.toggle('hc-dim', !on); l.classList.toggle('hc-hi', !!name && on);
    });
  }
  const tipOf = t => (t && t.closest) ? t.closest('[data-tip]') : null;
  document.addEventListener('mouseover', e => {
    const el = tipOf(e.target);
    if (el === current) return;
    clearTimeout(showTimer);
    if (!el) { hide(); return; }
    showTimer = setTimeout(() => { if (current) current.classList.remove('hc-active'); show(el); }, current ? 0 : 90);
  });
  document.addEventListener('mouseout', e => {
    const to = tipOf(e.relatedTarget);
    if (!to) { clearTimeout(showTimer); hide(); }
  });
  document.addEventListener('focusin', e => { const el = tipOf(e.target); if (el) show(el); });
  document.addEventListener('focusout', hide);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
  window.addEventListener('scroll', () => { if (current) place(current); }, true);
  document.addEventListener('pointerdown', e => {           // touch: tap shows, tap elsewhere hides
    if (e.pointerType !== 'touch') { hide(); return; }
    const el = tipOf(e.target); if (el && el !== current) show(el); else hide();
  });
  // keep an open card current when the viewer re-renders (step changes, play)
  window.HoverCard = {
    refresh() {
      if (!current) return;
      if (document.body.contains(current)) { show(current); return; }
      const spec = current.getAttribute('data-tip');           // element was re-rendered: follow its replacement
      const twin = [...document.querySelectorAll('[data-tip]')].find(e => e.getAttribute('data-tip') === spec);
      if (twin) show(twin); else hide();
    }, hide, GL, GL_EDGE };
})();
