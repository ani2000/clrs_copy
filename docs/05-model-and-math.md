# The model and the maths behind the attribution graph

This document derives, step by step, everything the page computes: the network's forward pass, the two kinds of transcoder, the replacement model with error nodes, the attribution graph (edge weights, the bias, skip edges), pruning, total effects and interventions. Each section ends with the place in the code where the formula lives. The sources are the project's own planning documents (the transcoder plan and the toy walkthrough written for the project) and the code.

Notation. `H = 128` is the width. Time steps are `t`. Nodes are `i, j, k`; `i` is a sender and `j` a receiver. Channels are `c` (1 to H). Latents are `f` (1 to D, with D = 1280). Vectors are rows; a matrix `W` of a layer with `din` inputs and `dout` outputs is `din x dout`, and the layer computes `x W + b`.

## 1. One step of the network

### 1.1 Inputs of a step

For every node k the processor reads

```text
node_fts_k  = sum over node-located fields (inputs and fed-back hints) of enc_field(value_k)      [128]
h_k         = hidden state of node k (zero at step 0)                                           [128]
z_k         = [ node_fts_k ; h_k ]                                                              [256]
e_ij        = sum over edge-located fields of enc_field(value_ij)                               [128]   per ordered pair
g           = sum over graph-located fields of enc_field(value)                                 [128]
```

Each field has its own linear encoder (`x W + b`; for categorical fields `x` is the one-hot class vector, for pointer fields a one-hot of the target node). Hints come from the previous step's own predictions (soft: sigmoid or softmax outputs), see section 1.5.

Code: `encodeInto` in `mpnn.js`.

### 1.2 The message from i to j

```text
s_ij = M2 z_i + M1 z_j + Me e_ij + Mg g + b                    (site input, pre-ReLU)           [128]
y_ij = L2 ReLU( L1 ReLU(s_ij) + b1 ) + b2                       (site output, "the message")     [128]
```

`M1, M2, Me, Mg` are the layers `m1, m2, me, mg` of the manifest; `b` is the sum of their four biases; `L1, L2` are `mlp0, mlp1`. The CLRS MPNN overrides the adjacency with all ones, so **every ordered pair (i, j), including i = j and non-edges, sends a message**. The real adjacency enters only through `e_ij`.

Because `s_ij` is a *sum* of a sender term (`M2 z_i`), a receiver term (`M1 z_j`), an edge term and a graph term, any linear read-out of `s_ij` splits exactly into those four parts. The edge-site transcoder's encoder is such a read-out.

Inside one pattern of the ReLUs the message function is affine: with `D1 = diag(1[s > 0])` and `D2 = diag(1[L1 D1 s + b1 > 0])`, `F(s) = L2 D2 L1 D1 s + (L2 D2 b1 + b2)`. The matrix `L2 D2 L1 D1` changes only when a ReLU flips; this is what the transcoder has to imitate with a sparse code.

Code: the loop over pairs in `runGen` of `mpnn.js` (`m1`, `m2`, `me`, `mg`, `mlp0`, `mlp1`).

### 1.3 Max aggregation

```text
m_j[c]  = max over senders i of y_ij[c]
w_j[c]  = argmax over senders i of y_ij[c]                       (the winner of channel c at receiver j)
```

Ties are broken in favour of the first sender with the largest value (strict `>` in the code, `argmax` in the reference). The hard max is why gradients flow only through winners.

### 1.4 Node update and LayerNorm

```text
x_j  = O1 z_j + O2 m_j + bo                                      [128]
u_j  = ReLU(x_j)
mu_j = mean over channels of u_j
sigma_j = sqrt( mean over channels of (u_j - mu_j)^2 + 1e-5 )
h'_j = gamma * (u_j - mu_j) / sigma_j + beta                      new hidden state
```

`O1, O2` are `o1, o2`; `bo` is the sum of their biases; `gamma, beta` are the LayerNorm scale and offset.

### 1.5 Feedback and decoders

After the update, each hint field has a decoder reading `[node_fts_j ; h_j ; h'_j]` (3H = 384 numbers) and producing the hint for the next row: a sigmoid for masks, a softmax for categorical and mask_one fields, a plain number for scalars, and for pointers a max over a pair of linear read-outs followed by a softmax over nodes. These predictions are encoded and become the hint inputs of the next step. This is the "hint loop": the network's own predictions are its next inputs.

During training the true hint replaces the prediction with probability 0.5 (teacher forcing); the page always feeds back the model's own predictions.

Code: `decodeHint` in `mpnn.js`.

### 1.6 What is hard about explaining it

- *No residual stream.* The state is overwritten by `h' = LayerNorm(ReLU(...))`. A latent cannot "write forward" into later layers as in a transformer; its effect on the next step goes through the max and the node update.
- *Hard routing.* A latent on pair (i, j) affects only the channels that i wins at j.
- *n^2 sites.* With n = 16 there are 256 message rows per step.
- *Tied weights.* The same processor runs at every step.
- *Hint loop.* Inputs at step t+1 depend on predictions at step t.

## 2. The transcoders

### 2.1 Edge-site transcoder (the main page)

Trained on rows `(s_ij, y_ij)` from the trained MPNN's own runs.

```text
x     = (s - s_mu) / s_scale                         s_mu [128], s_scale scalar
pre   = (x - b_pre) W_enc + b_enc                    W_enc [128 x D], D = 1280
a     = ReLU( TopK_k(pre) )                          k = 8: keep the 8 largest entries (ties at the 8th value all kept), then clip negatives to 0
y_hat = (a W_dec + b_dec) * y_scale + y_mu           W_dec [D x 128]
eps   = y - y_hat                                    the error node (computed on the real run)
```

`a` has at most 8 non-zero entries (the average is about 7.9). Training loss: `NMSE(y_hat, y) + (1/32) * L_aux` with `L_aux` an auxiliary term asking the dead latents to explain the residual; 50,000 steps; batch of rows. The quality of the result per algorithm is in `16-transcoder-training-reports.md`.

Code: `encodeRow` and `decodeRow` in `clrs-edge-attrib.js`; the training code is in the notebook `clrs-transcoder.ipynb` (functions `tc_encode`, `tc_decode`).

### 2.2 Node-site transcoder (`/demo`)

Reads the node-update input and predicts the new hidden state directly:

```text
x  = [ z_i ; max-aggregated message m_i ]  (384 numbers, standardised with mu_x, sd_x)
pre = x W_enc + b_enc,   W_enc [384 x 192]
a  = ReLU(TopK_8(pre))   (192 features)
h'_hat = a W_dec + b_dec                    [128]
```

Because it replaces the *whole* node update (including the LayerNorm), the old graph needs no frozen LayerNorm. It still has to follow how the aggregated message depends on the senders, so it freezes the ReLU masks of the message function and the max routing (`backprop` in `attrib.js` walks back through them). It has one error node per node (the part of `h'` not reconstructed), not one per pair.

Code: `transcode`, `prune`, `build` in `attrib.js`.

### 2.3 Why two sites, and why only the edge site is used now

The training plan chose the message function as the first site because it is "the only real nonlinearity before the hard max". After training, the measured *retention* (share of the baseline score kept when the transcoder replaces the site inside the network) was 0.97 to 1.00 at n = 16 for the edge site on all twelve algorithms, while the node-site transcoder alone kept only 0.13 to 0.49 on dfs, strongly_connected_components and floyd_warshall and collapsed at n = 64 for several algorithms. Both sites together collapsed on long runs. So the page uses the edge site only.

## 3. The replacement model and the error node

Replacing the message function by `y_hat` everywhere changes the network. To keep the *real* computation, the error is added back on the real run:

```text
y_used = y_hat + eps = y        (exactly the original message)
```

so the replacement model with error nodes reproduces the real network, and the attribution graph is exact for the real run whatever the quality of the transcoder; the quality decides only *how much* of the computation the latents (rather than the error nodes) explain. The check "E0" of the plan is that this identity holds to rounding error; test 4 replays it in float64 (difference about 1e-14).

Three modes are evaluated in the training report: *direct* (one step replaced, true inputs), *frozen* (all steps replaced, winners and sigma taken from the real run), *full* (all steps replaced, winners, sigma and hints recomputed). In full mode the winner can flip: with reconstruction error at most delta in every entry, the winner of a channel can change only when `2 delta >= gap`, the gap being the difference between the best and second-best sender; the value `max y_hat` is always within `delta` of `max y`. Winner flips happened in 9% to 20% of channels when the replacement ran alone; the bound was checked and never violated. The attribution graph uses the *real* winners, so flips do not affect it.

## 4. What is frozen to make the graph linear

On one input, three families of decisions are fixed at their real values:

1. **the winners** `w_j[c]` of the max (so `m_j[c] = y_{w_j[c], j}[c]`);
2. **the ReLU pattern of the node update**, `Du_j = diag(1[x_j > 0])`;
3. **the LayerNorm sigma** `sigma_j` (treated as a constant).

The TopK set of each transcoder is also fixed (the latents that are active are the nodes of the graph). With these fixed the node update is linear in `(z_j, m_j)`:

```text
u_j   = Du_j x_j
h'_j  = diag(gamma / sigma_j) P u_j + beta,       P = I - (1/H) 1 1^T        (centring over the channels)
      = G_j x_j + beta,                            G_j = diag(gamma/sigma_j) P Du_j
```

and `m_j` is linear in the latents and error nodes at the pairs (winner, j):

```text
m_j[c] = y_hat_{w,j}[c] + eps_{w,j}[c] = ( sum_f a_f(w,j) W_dec[f][c] + b_dec[c] ) * y_scale + y_mu[c] + eps_{w,j}[c]       with w = w_j[c]
```

This is exactly what the notation "(M2 1[j = i'] + M1 1[j = j']) ..." of the toy walkthrough expresses, spelled out below.

## 5. The attribution graph

### 5.1 Nodes

For the target (receiver `r` at step `t`) and a window of three steps `tLo..t`:

| node | meaning | count |
|---|---|---|
| latent `F{step}:{pair}:{f}` | latent `f` active on the pair `(sender, receiver)` at `step`; activation `a` | up to 8 per pair; all pairs at steps before `t`; at step `t` only pairs whose receiver is a target node |
| error `E{step}:{pair}` | the error `eps` of the pair's message | created when used |
| leaf `L...` | an encoded input or hint value at a step (one per field and node, edge or the graph) | created when used |
| earlier-state leaf `S{step}:{node}` | the hidden state entering the window | created when used |
| bias `B{step}` and `Bout` | all constant terms | one per step and one for the predictions |
| logit `T:...`, `P:...` | the target logits being explained: the main hint of the receiver and, for pointer algorithms, the pointer logits | 1 to a few |

At step `t` only latents written *to a target node* can reach the logits (the hidden state of the receiver is the only thing the decoder reads), which is why the engine encodes only those pairs at the last step.

### 5.2 Edges into a latent (the key formula)

Take latent `f2` at pair `(i2, j2)` at step `s`. Its pre-activation is

```text
pre_f2 = b_enc[f2] + sum_c ( (s_c - s_mu_c)/s_scale - b_pre_c ) W_enc[c][f2]
       = const_f2 + q . s,            q = W_enc[:, f2] / s_scale
       = const_f2 + q . ( M2 z_{i2} + M1 z_{j2} + Me e_{i2 j2} + Mg g + b )
```

Define the cotangents on the inputs of the step: `gz_{i2} += M2 q`, `gz_{j2} += M1 q`, `ge_{i2 j2} = Me q`, `gg = Mg q` (each a layer applied transposed to `q`). Then every term of `pre_f2` is a product of a cotangent and a *source quantity*:

- the first half of `gz_k` multiplies `node_fts_k`, which is a sum of encoder outputs: edges from **leaves** (section 5.4);
- `ge` multiplies the edge encoding at `(i2, j2)`: leaf edges for edge-located and node-pointer fields;
- `gg` multiplies the graph encoding: leaf edges for graph-located fields;
- the second half of `gz_k` multiplies `h_k`, the hidden state entering the step: section 5.3.

### 5.3 Where the hidden state comes from (including skip edges)

Let `gh` be the cotangent on the hidden state `h'_k` produced by step `s-1` (it is the second half of some `gz_k`). Using section 4:

```text
h'_k = G_k x_k + beta,    x_k = O1 z_k^{s-1} + O2 m_k^{s-1} + bo
```

so with `gx = G_k^T gh = Du_k * P( (gamma/sigma_k) * gh )` (elementwise products; `P` subtracts the mean):

```text
gh . h'_k  =  gx . ( O1 z_k + O2 m_k + bo ) + beta . gh
```

and three kinds of terms appear:

1. **Messages.** `gm = O2 gx` (a cotangent on `m_k`). For each sender `i` that wins at least one channel `C_i` at receiver `k`:
   - for every active latent `f` on the pair `(i, k)` at step `s-1`: edge weight `a_f * y_scale * sum over c in C_i of gm[c] W_dec[f][c]`;
   - the error node of that pair: weight `sum over c in C_i of gm[c] eps_{ik}[c]`;
   - constants `sum over c in C_i of gm[c] (b_dec[c] y_scale + y_mu[c])` go into the bias.
2. **Own previous state (skip edges).** `gz' = O1 gx` is a cotangent on `z_k^{s-1} = [node_fts_k ; h_k^{s-1}]`: its first half gives leaf edges at step `s-1`, its second half is a hidden state again, and the same expansion is applied one step further back (or, at the start of the window, becomes an earlier-state leaf with weight `h . gh`). The latents of step `s-2` reached this way are connected *directly* to the latent at step `s`: a **skip edge**. They are exact: hidden state can bypass the latents through `O1 z`.
3. **Constants.** `beta . gh + gx . (bo-part)` goes into the bias.

Because only the node `k` itself appears at every level of the expansion (the cotangent never spreads to other nodes), the recursion is cheap: it costs about 50 thousand multiplications per level.

A sender that wins no channel at receiver `k` contributes nothing at all: all its outgoing edges to `k`'s state are zero.

### 5.4 Edges from leaves

For a node-located field the weight of the leaf "value of field F at node k, step s" is `(value) x (encoder direction . gz_k[first half])`; for categorical fields the value is the one-hot vector, so the weight is the cotangent entry of the active class; edge-located and graph-located fields work the same with `ge` and `gg`; edge-located pointer fields use the two means of the pointer tensor. Encoder biases enter the bias node.

### 5.5 The bias is explicit, not forced

The bias of each destination is the sum of every constant term collected on the way: the transcoder's `b_enc` and the offsets `s_mu, b_pre`, the four layer biases `b` of `M1, M2, Me, Mg` (times `q`), the encoder biases of the fields, `beta`, the biases `bo` of `O1, O2`, and the decoder biases through the winners. It is *computed*, not defined as "value minus the sum of the other edges". Therefore the statement

```text
value of a node  =  sum of its incoming edge weights  +  its bias
```

is a real test: the engine evaluates the left side from the actual forward pass and the right side from the formulas, and the page reports the largest difference over every node (about 1e-14). Sections 4 and 5 are what make that identity true.

### 5.6 Edges into a target logit

For a node-type target (mask, scalar, categorical, mask_one) the decoder is `logit = d0.b[c] + sum_k d0.w[k][c] * [node_fts ; h ; h'][k]` over the three blocks of 3H = 384 inputs of the receiver `i`; so the cotangents are: the first two blocks on `z_i` (leaf edges and the hidden-state expansion of section 5.3) and the third block on `h'_i` (the expansion of section 5.3 started at the last step). For pointer targets the logit is `l4.b + sum_c l4.w[c] * (sel[c] ? p1[c] : p2[c] + p3[c])` with `p1` from the sender's block, `p2` from the receiver's block and `p3` from the edge feature; the selection `sel` is frozen as well. The biases of those layers go into the bias node `Bout`.

### 5.7 Everything linear gives exact total effects

Put `W[s, d] = weight(s -> d) / activation(s)` for latent sources and `weight` for others. The **total effect** of one unit of activation of node `v` on logit `l` is `r[v]` where `r[l] = 1` and `r[s] += W[s, d] r[d]` for every edge, processed from the latest step backwards (every edge goes from an earlier step to a later one, so one pass in that order is enough). For the last-step latents this is just `weight / activation` of their edge into the logit; earlier latents add the paths through later latents and skip edges.

## 6. Pruning (the Circuit Tracing recipe)

Let `A[e] = |weight(e)|` and, for a destination `d`, `colsum[d] = sum of A over edges into d`.

1. **Influence.** Start with `x[logit] = probability of the logit` (for a mask target the probability of the chosen side, for a class or pointer the softmax probability, with at least 0.02 for mask_one) and zero elsewhere. Moving backwards over the edges: `x[source] += A[e] * x[dest] / colsum[dest]`. The *influence* of a node is its `x` (without the logit's own probability).
2. **Nodes.** Keep the logits that matter (probability above 0.01 or the always-shown top choice), the most influential node, and then nodes in order of decreasing influence until they carry 80% of the total influence.
3. **Edges.** Among kept nodes, score every edge `A[e] / col2[dest] * (influence[dest] + probability if the destination is a logit)` and keep edges in decreasing score until they carry 98% of the total score.
4. A node is shown if it is a logit or touches a kept edge.

The status line reports the kept share of the influence ("pruned to 47 / 174, keeping 80% of the influence on the targets").

Code: `prune` in `clrs-edge-attrib.js` (and `prune` in `attrib.js` for the node-site graph).

## 7. Interventions

To clamp the latent `f` at pair `(i, j)` at step `t0` to `factor` times its activation `a`, add

```text
dy = (factor - 1) * a * W_dec[f] * y_scale
```

to the message `y_ij` of that pair at step `t0`, and run the **real network** forward from there (winners, ReLUs, sigma, TopK sets and the hint feedback all recomputed). Compare, for every target logit, the real change `actual - base` with the graph's prediction `(factor - 1) * a * total_effect`. The page also counts the downstream decisions that changed (winners of every channel at the steps `t0..t`, and the TopK sets of all pairs at the steps after `t0`).

Why the prediction is only approximate: it holds the winners, sigma and TopK sets fixed, and it keeps the errors `eps` fixed, while the real network recomputes the message function (and thus the errors) from the changed input; changing a latent by a large factor also changes who wins channels. In the project's tests the direction matched in 223 of 325 comparisons (69%); the share is 44% to 77% depending on the algorithm.

Code: `intervene` in `clrs-edge-attrib.js`; `patchY` in `mpnn.js`.

## 8. How the maths was checked

| claim | test |
|---|---|
| the JavaScript transcoder equals an independent numpy float64 implementation | test 1 (4e-13) |
| the transcoder reconstructs this site's real messages | test 2 (FVU 0.002 to 0.035) |
| replacing the message function by `y_hat + eps` reproduces the real hidden states and logits | test 4, "E0 replay" (about 1e-14) |
| every edge weight equals the finite difference of the replacement model | test 4 (about 1e-13, 15 to 40 thousand comparisons per algorithm) |
| every total effect equals the finite difference of the logit | test 4 |
| value = sum of incoming edges + explicit bias for every node | the sum test inside every graph build (about 1e-14) and test 4 |
| the transcoder works inside the network | test 8 |
| the old graph is unchanged | test 6 |

Details and results are in `11-testing-and-verification.md`.

## 9. Where each formula lives

| formula | function | file |
|---|---|---|
| message input and output, max, node update, hint decoders | `runGen`, `encodeInto`, `decodeHint` | `mpnn.js` |
| transcoder encode / decode | `encodeRow`, `decodeRow` | `clrs-edge-attrib.js` |
| edges into latents, hidden-state expansion, leaves, bias | `explain` (with `leafEdges`, `fromHidden`, `expandHidden`) | `clrs-edge-attrib.js` |
| edges into logits | the target loop at the end of `explain` | `clrs-edge-attrib.js` |
| influence, pruning, total effects | `prune` | `clrs-edge-attrib.js` |
| packing the result for the page | `pack` | `clrs-edge-attrib.js` |
| clamping | `intervene` | `clrs-edge-attrib.js` |
| the target definitions (which logits are explained) | `targetsFor` | `attrib.js` |
| the node-site graph | `build`, `backprop`, `prune`, `intervene` | `attrib.js` |
