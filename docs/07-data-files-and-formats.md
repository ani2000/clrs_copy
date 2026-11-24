# Data files and formats

Every number on the site comes from one of the files described here. For each file this document says who writes it, who reads it, the exact layout, and a real example taken from the repository.

## Map of the data

```text
training run (GPU, Sept 2026)        transcoder run (SageMaker, Nov 2026)
  results_v3/summary_v3.json            clrs-transcoder-out/<algo>/report.json, transcoder.npz, transcoder_node.npz
  results_v3/<algo>__mpnn.json              |
  checkpoints_v3/<algo>/best.pkl            |  clrs_export_edge_tc.py
        |                                   v
        |  model_export.py (image build)   edge_tc/<algo>.bin, .json, clrs-edge-summary.json
        v                                   |  clrs_edge_latents.js (60 graphs per algorithm from /api/instance)
   models/<algo>.json + .bin                v
   (inside the image only)             edge_tc/<algo>.latents.json

 pipeline.py (earlier work)  -->  algos/<algo>/transcoder.npz, feature_labels.json, meta.json   (node-site, for /demo)
 fig47/*.py                  -->  fig47/*.json, *.png                                           (earlier Figure 4 / 7 study)
 /api/instance               -->  the instance JSON of one sampled graph (never stored)
```

## 1. The instance (`/api/instance`)

Written by `model_export.build_instance`; read by the browser. Not stored anywhere.

Real example (`algo=bfs&n=16&p=0.4&seed=101`):

| field | type and size | example |
|---|---|---|
| `n` | integer | 16 |
| `length` | integer | 5 (hint rows 0 to 4, so 4 message-passing steps) |
| `seed` | integer | 101 (the seed actually used) |
| `adj` | 16 x 16 numbers | adjacency matrix |
| `source` | integer or null | 5 |
| `target_true` | `length` rows x 16 numbers | row 1: `[1, 0, 1, 0, 1, 1, ...]` (the true `reach_h` after the first step) |
| `inputs` | dict | keys `pos`, `s`, `A`, `adj` for bfs |
| `hints0` | dict | keys `reach_h`, `pi_h`: the first hint row |

For categorical targets `target_true` holds the class index (the arg-max of the one-hot hint). The set of keys in `inputs` and `hints0` depends on the algorithm (the model manifest lists them).

## 2. The model bundle (`models/<algo>.json` and `.bin`)

Written by `model_export.py` during the image build (not in git). Read by `mpnn.js`.

`.bin` is a flat array of little-endian float32 numbers. `.json` is the manifest that tells where each array starts (all offsets are in floats, not bytes):

```json
{
  "algo": "bfs", "H": 128, "nfloats": 298626,
  "proc": {
    "m1":  {"w": 0,     "b": 32768,  "din": 256, "dout": 128},
    "m2":  {"w": 32896, "b": 65664,  "din": 256, "dout": 128},
    "me":  {"w": 65792, "b": 82176,  "din": 128, "dout": 128},
    "mg":  {"w": 82304, "b": 98688,  "din": 128, "dout": 128},
    "o1":  {"w": 98816, "b": 131584, "din": 256, "dout": 128},
    "o2":  {"w": 131712,"b": 148096, "din": 128, "dout": 128},
    "mlp0":{"w": 148224,"b": 164608, "din": 128, "dout": 128},
    "mlp1":{"w": 164736,"b": 181120, "din": 128, "dout": 128},
    "ln":  {"scale": 181248, "offset": 181376}
  },
  "fields": [
    {"name": "pos", "stage": "input", "loc": "node", "type": "scalar",
     "enc": [{"w": 181504, "b": 181632, "din": 1, "dout": 128}], "K": 1},
    ...
  ]
}
```

| key | meaning |
|---|---|
| `proc.m1`, `m2` | linear layers on `z` giving the receiver term and the sender term of the message input (`din` 256 = `z`, `dout` 128) |
| `proc.me`, `mg` | linear layers on the edge encoding and the graph encoding |
| `proc.o1`, `o2` | node-update layers: `O1` on `z`, `O2` on the aggregated message |
| `proc.mlp0`, `mlp1` | the two layers of the message MLP (`L1`, `L2`) |
| `proc.ln` | LayerNorm scale (gamma) and offset (beta) |
| `fields[i].enc` | the encoder layer(s) of an input or hint field (two for edge-located pointer fields) |
| `fields[i].dec` | the decoder layers of a hint field (three or more for pointer fields) |
| `fields[i].K` | width of the encoder input (1 for scalar-like fields, the number of classes for categorical ones) |
| `nfloats` | total number of floats in the `.bin` |

A weight matrix `w` of a layer with `din` inputs and `dout` outputs is stored row-major as `din x dout` (`out = x @ w + b`). The sizes of the bundles range from about 1.19 MB (bfs) to 2.14 MB (dag_shortest_paths).

## 3. The edge-site transcoder (`edge_tc/<algo>.bin` and `.json`)

Written by `clrs_export_edge_tc.py` from `clrs-transcoder-out/<algo>/transcoder.npz`. Read by `clrs-edge-attrib.js` (`prepTC`).

Manifest example (bfs):

```json
{"algo":"bfs","H":128,"D":1280,"k":8,
 "offsets":{"s_mu":[0,128],"y_mu":[128,128],"b_pre":[256,128],"b_dec":[384,128],"b_enc":[512,1280],
            "scales":[1792,2],"W_enc":[1794,163840],"W_dec":[165634,163840]},
 "sha256":"bd2ffdd7007c0ac4e7538e829a9db7d3209826bc1ffab729e44ab35f89f7324d","floats":329474}
```

Each `offsets` entry is `[start, length]` in floats. The arrays in order:

| array | shape | role |
|---|---|---|
| `s_mu` | 128 | mean of the site input, subtracted before encoding |
| `y_mu` | 128 | mean of the message, added after decoding |
| `b_pre` | 128 | pre-encoder bias |
| `b_dec` | 128 | decoder bias |
| `b_enc` | 1280 | encoder bias |
| `scales` | 2 | `[s_scale, y_scale]` |
| `W_enc` | 128 x 1280 (row-major) | encoder weights |
| `W_dec` | 1280 x 128 (row-major) | decoder weights |

Formulas: `x = (s - s_mu) / s_scale`, `pre = (x - b_pre) @ W_enc + b_enc`, `a = relu(topk_8(pre))` (ties at the threshold are kept), `y_hat = (a @ W_dec + b_dec) * y_scale + y_mu`. The `.bin` file has 329,474 floats, that is 1,317,896 bytes; `sha256` in the manifest is the checksum of the `.bin`. The transcoder arrays are stored as float32 and converted to float64 in the browser.

## 4. The edge-site summary (`edge_tc/clrs-edge-summary.json`)

Written by `clrs_export_edge_tc.py`, field by field from `report.json`; read by the page for the model panel and the quality note. Top-level keys: `source`, `site` (`edge: s_ij -> y_ij`) and `summary` (a dict by algorithm).

| field | meaning |
|---|---|
| `tier` | A, B or C (see the glossary) |
| `baseline_val_f1`, `baseline_test_f1` | score of the unmodified MPNN on 16-node validation graphs and 64-node test graphs |
| `D`, `k` | number of latents and active latents per pair |
| `nmse_train`, `nmse_val`, `nmse_n64` | reconstruction error (FVU) of the message; `nmse_n64` is null when not measured |
| `dead_frac_train` | share of latents that never fire on the training rows |
| `l0_val` | mean number of active latents per row on validation rows |
| `retention_val`, `retention_n64` | share of the baseline score kept with the transcoder in the network (n64 null when not measured) |
| `f1_full_val`, `f1_off_val`, `f1_full_n64`, `f1_off_n64` | the two scores behind the retention |
| `winner_flip_rate`, `winner_flip_violations` | share of channels whose max-winner changes when the replacement runs alone, and violations of the `2 x error >= gap` bound |
| `hidden_rel_error_direct_mean` | mean over steps of the relative hidden-state error of a single replaced step |
| `hidden_rel_error_direct` | the same, per step |
| `nmse_per_step_val` | reconstruction error per algorithm step on validation rows |
| `dead_frac_val`, `train_rows`, `train_steps` | dead fraction on validation rows, rows and steps used to train |
| `parity_abs_diff`, `minutes` | difference to the recorded baseline score, minutes of the whole run |
| `node_site_retention_val`, `both_sites_retention_val` | retention when the node-site transcoder replaces the node update alone, and both sites together (not used on the site) |

## 5. The latent tables (`edge_tc/<algo>.latents.json`)

Written by `verification/clrs-edge/clrs_edge_latents.js`; read by `attrview.js`. Top-level keys:

| key | meaning |
|---|---|
| `algo`, `target`, `target_type` | which algorithm and target |
| `graphs`, `graphs_fit` | graphs measured (60) and how many were used to choose conditions (30, even seeds) |
| `rows_fit`, `rows_held` | pair rows in the two halves (for bfs 34,560 and 35,328) |
| `split` | text: conditions are chosen on even seeds and scored on odd seeds |
| `sampler` | `n=16, p=0.4` |
| `max_steps_per_graph` | at most 20 evenly spaced steps per graph are used |
| `effect_of` | what "effect" is measured on (for example `logit of reach_h at the receiver`) |
| `max_sum_check`, `effect_calls` | the worst sum-test difference over all effect computations, and how many were made |
| `conditions` | list of `{id, text, base_rate_held}`: the conditions tested and how often each is true on held-out rows |
| `latents` | dict by latent number (only latents that fired) |
| `fired`, `named`, `tier_a`, `never_fired` | counts |

One latent entry (real example, latent 0 of bfs):

```json
{"n": 518, "rate": 0.00741, "act": 0.651, "max": 1.75, "kind": [0, 1, 0],
 "eff": [-0.00311, 0.0111, 238, 0.445],
 "best": [[9, 1, 0.0189, 0.0371, 0.0351], [12, 1, 0.0189, 0.0371, 0.0351], [8, 1, 0.0128, 0.0253, 0.0237]],
 "tier": "-", "label": null,
 "ex": [[2044, 5, 0, 14, 1.75], [2011, 6, 0, 14, 1.67], [2011, 6, 1, 14, 1.61]]}
```

| field | meaning |
|---|---|
| `n` | number of firings among all measured pair rows |
| `rate` | `n` divided by all measured rows |
| `act`, `max` | mean and maximum activation |
| `kind` | share of the firings on real edges, non-edges, self pairs |
| `eff` | `[mean, standard deviation, count, share positive]` of the effect on the receiver's target logit per unit activation; `null` if never measured |
| `best` | up to three best conditions, each `[condition index, held-out precision, held-out recall, held-out F1, F1 on the fitting half]`, best first (ordered by the fitting half's F1) |
| `tier` | `A` if the held-out F1 of the best condition is at least 0.8, `B` if at least 0.5, else `-` |
| `label` | the text of the best condition when named, else null |
| `ex` | the three strongest activations: `[seed, step, sender, receiver, activation]` |

Latent 0 of bfs fires only on non-edge pairs (`kind [0, 1, 0]`) but rarely (0.7% of the rows), and matches no condition well (best held-out F1 0.04), so it is unnamed.

## 6. The node-site files (`algos/<algo>/`, used by `/demo`)

### `transcoder.npz`

NumPy archive written by `pipeline.py`:

| key | shape | meaning |
|---|---|---|
| `w_enc` | 384 x 192 | encoder; input is `[z (256), aggregated message (128)]` |
| `b_enc` | 192 | encoder bias |
| `w_dec` | 192 x 128 | decoder to the new hidden state |
| `b_dec` | 128 | decoder bias |
| `mu_x`, `sd_x` | 384 | mean and standard deviation used to standardise the input |
| `F`, `K` | scalars | 192 features, 8 active |

`/api/weights` serves these (as lists) together with `W_dec`/`b_dec_target` (the decoder of the target, from `meta.json`), the target description and the scores.

### `meta.json`

Keys: `algo`, `target`, `target_stage`, `target_type`, `is_output`, `source_field`, `hidden` (128), `F`, `K`, `fvu` (reconstruction error of the node-site transcoder), `val_f1`, `test_f1`, `n_rows` (rows used to train and label), `W_dec`, `b_dec` (decoder of the target hint), `elapsed_s`.

### `feature_labels.json`

`{"rows": [...], "total_sites": N}` where each row is `{feature, label, precision, recall, f1, n_fire}` (example bfs: feature 9, label "stays 0", P 0.999, R 1.0, F1 1.0, fires on 17,535 of 51,728 sites). A *site* is a (node, step) pair of the labelling dataset. Labels come from hand-written predicates on the true hints (for example "newly 1", "stays 0", "is_source", "not is_source").

## 7. The training outputs (outside the repository; copied into the documents)

### `results_v3/summary_v3.json`

Keys: `version`, `timestamp`, `instance` (the rented GPU), `elapsed_hrs` (6.8 hours in total), `gpu_vram_gb`, `processor` (`mpnn`), `seed` (42), `protocol`, `jobs_done`, `jobs_ok`, `results` (a list of twelve). `protocol` = `{train_steps: 10000, eval_every: 100, batch_size: 32, hidden_size: 128, learning_rate: 0.001, grad_clip: 1.0, hint_mode: encoded_decoded, hint_teacher_forcing: 0.5}`. Each result has `algo`, `proc`, `version`, `steps_run`, `elapsed_min`, `best_val_f1`, `test_f1`, `val_curve` (100 `[step, score]` pairs), `status`, `returncode`, `oom_details`, `gpu`, `vram_gb`, `stdout_tail`.

### `results_v3/<algo>__mpnn.json`

The same record for one algorithm. `CLRS30_Graphs_MPNN_Curve_Analysis.md` in that folder is a written analysis of the validation curves.

### `clrs-transcoder-out/<algo>/report.json`

Written by the transcoder notebook. Blocks (all copied into `16-transcoder-training-reports.md`):

| block | content |
|---|---|
| `algo`, `tier`, `baseline` | name, tier, baseline val/test score |
| `config` | `act` (`topk`), `D`, `k`, `steps`, `node_site`, `D_node`, `k_node`, `node_steps` |
| `hint_policy` | which hints were used for the rows, for the direct/frozen modes and for the full mode |
| `parity`, `offline_parity`, `offline_parity_rel` | agreement of the training code with the recorded baseline and with the network's own site inputs and outputs |
| `rows`, `rows_node` | number of train / validation / 64-node test rows |
| `train_history`, `train_history_node` | every 5,000 steps: batch NMSE, auxiliary loss, L0, dead latents now, seconds |
| `E1`, `E1_node` | reconstruction on train, validation and 64-node test rows, overall and per step, with L0 and dead fraction |
| `direct` | winner flips (counts and rate, channels at risk, violations of the bound) and hidden-state error per step for the direct and frozen modes |
| `E2` | scores for `off`, `plain`, `plumb`, `full`, `node_full`, `both_full` on validation and test, the retentions and the plumbing differences |
| `train_seconds`, `train_seconds_node`, `minutes` | time |

### `clrs-transcoder-out/bfs/lag.json`

Only for bfs: results of an experiment with "lagged" decoders (a latent also writes into later steps), for K = 0, 1, 2. It gave no shortening of the circuit and is not used on the site.

### `clrs-transcoder-reference.json`

`{readme, metrics, golden}`. `metrics` is the quality table of all algorithms; `golden` holds, for every algorithm and site, three random input rows (`s`) with the active latent indices (`active_idx`), their values (`active_val`), the first eight pre-activations (`pre_first8`) and the reconstruction (`y_hat`). The inputs are stored with six significant digits while the outputs were computed from the unrounded inputs, so a port agrees with the outputs to about 1e-3 relative, not to rounding error (see test 1).

## 8. The checkpoints

`checkpoints_v3/<algo>/best.pkl` (Git LFS) is a Python pickle `{"params": {...}}` with the Haiku parameter dictionary of the best validation step: keys such as `net/mpnn_aggr_clrs_processor/linear` (the layers in creation order `m1, m2, me, mg, o1, o2`), `net/mpnn_aggr_clrs_processor/mlp/~/linear_0` and `_1`, `net/mpnn_aggr_clrs_processor/layer_norm`, and `net/~_construct_encoders_decoders/algo_0_<field>_enc_linear` / `..._dec_linear` for every field. About 5 MB per algorithm (the largest, dag_shortest_paths, about 7.8 MB). `config.py` checks that these exist.

## 9. The earlier study (`fig47/`)

See `17-fig47-study-results.md` for every file. In short: `phase*.json` hold the intervention results, `fig4_*` and `fig7_*` the figure data and images, `single_example_<algo>.json` and `predictions_<algo>.json` the examples shown by `viewer_fig47.html`.

## 10. What is *not* stored anywhere

Visitors' choices (apart from which sections they opened, which stays in their own browser), sampled graphs, traced attribution graphs and interventions. Everything is recomputed on demand, and the server writes nothing to disk.
