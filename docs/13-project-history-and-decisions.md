# Project history and decisions

How the site got to its present form, in the order the commits record it, plus the decisions that shaped it and the reasons for them. Dates are the commit dates in the repository (all in November 2026). Commit hashes are short ones from `git log`.

## 1. Timeline

### 1.1 5 November: import, deployment, loading speed

| commit | what it did |
|---|---|
| 44d7dc1 | Initial import of the CLRS circuit tracer: the model code, the checkpoints, the page, the server |
| 9fc23f4 | Hardened deployment workflows: lookups scoped by tags, SSM instead of SSH, an OIDC option, pinned actions, a smoke test |
| 2e6e81f | The deploy workflow creates its own tagged security group (`clrs-circuit-tracer-app-sg`) |
| 3cfbe54 | Removed tool-provenance metadata and a local path; tidied a citation string |
| 85550e4 | A manual cleanup workflow for two leftover resources (exact IDs, dry run first) |
| 52a4387 | Faster loading: background model warm-up, ETag and immutable caching with build-versioned URLs, bundle download retry |
| 8d93d22 | Fixed truncated model downloads on slow links: chunked writes with per-chunk timeout, connection dropped on write errors |
| 5cdcb70 | No speculative server graphs until the in-browser model is ready (frees bandwidth on slow links) |
| 7cdc776 | Manual workflow to switch TCP congestion control to BBR (apply or revert) |
| 2b298fd | Manual workflow to soft-reboot the instance (tag and ID guarded) |
| 7dd6225 | Switching algorithm cancels the old algorithm's network work (abortable downloads, aborted graph requests) |
| d073252 | Server memory bounded: some endpoints no longer load a model, warm-up only bfs and dfs, at most 3 models and a few compiled programs, container memory capped |
| cb8b8a3 | The first graph comes from the in-browser model and switching is instant |
| dac55ca | Manual workflows: CloudFront setup, pre-demo check, cross-browser check |
| 1cd2b50 | Check workflows pinned to the Ubuntu 24.04 runner image |
| a9ae888 | CloudFront setup creates the reversible pieces first and changes the instance address last |

The first day was about making a working research tool into something that could be shown from a small machine: memory limits, caching, behaviour on slow links, and workflows that touch only this project's AWS resources.

### 1.2 23 November: the pseudo-code pane and the first changes to the selector

| commit | what it did |
|---|---|
| 4a96f44 | A pseudo-code pane that follows the simulation steps; simulation, attribution graph and pseudo-code side by side |
| 7a36d4a | CloudFront setup: no Elastic IP and no alarm; origin is the instance's public DNS name (no extra cost) |
| f242ffd | Manual workflow: HTTPS on the instance with Caddy and a free sslip.io name (check, apply, disable) |
| 30b8f7f | Training scores and the reason for each output type shown in the algorithm selector |
| adc37f9 | Pseudo-code pane: detailed lines and nesting-aware highlighting |
| 11b7370 | Page and scripts sent gzip-compressed |

### 1.3 24 November: the edge-site transcoder, the split into two pages, meanings

| commit | what it did |
|---|---|
| e5e0480 | Pseudo-code pane: path strip and long tags dropped; code readable at every width |
| 95979ac | Edge-site transcoder attribution engine, verified on all 12 algorithms (tests 1 to 9) |
| 179bede | Edge-site attribution graphs in the page, with a toggle to the previous graphs |
| 74563a0 | `clrs-edge-attrib.js` served (the worker loads it) and part of the build id |
| 8f9ad58 | The main page shows only the edge-site version; the earlier version moves to `/demo` |
| 05bcb55 | Edge-site latents: what each one means, top latents per step, dictionary, message map |
| 27b76b5 | Lower part of the page: collapsible sections, regrouped |

After these commits came the documentation set in `docs/` (this folder).

## 2. Decisions and their reasons

### 2.1 Edge site instead of node site for the main page

The first attribution graphs were built on a transcoder at the *node* site: 384 inputs to 192 features. The later one works at the *edge* site (`s_ij` to `y_ij`, D = 1280 features, TopK 8). The edge site is where the message passing actually takes place: the model's computation is a sum over edges, so features on edges can be tied to what a message carries from one node to another. The node-site version stayed available at `/demo` because it was already shown and works; the request was to keep it separately. Details: `05-model-and-math.md`, `16-transcoder-training-reports.md`.

### 2.2 Everything the graph shows must be computed, not drawn

The rule that came out of repeated requests ("no demo, no false data"): each number on the page comes from the model or the transcoder in the browser, or from a stored measurement (latent meanings, training reports). The truth test (`verification/clrs-edge/ui_truth_test.mjs`) compares what the page shows with an independent recomputation, and the layout test checks that nothing overflows. When a statement could not be verified it was removed or rephrased (for example claims in the docs about instance sizes and workflow run times).

### 2.3 Latent meanings: fit on even seeds, scored on odd seeds

A latent gets a name only if a simple condition on the graph (for example "the source node is in the frontier") predicts its activation on graphs the condition was *not* fitted on. Conditions are fitted on even-seed graphs and scored on odd-seed graphs; tier A needs F1 of at least 0.8, tier B at least 0.5. Most latents therefore remain unnamed, by design: between 3 and 18 of roughly 650 to 1,190 firing latents per algorithm are named. The alternative, naming everything by its top activations, would have produced plausible but unchecked labels.

### 2.4 Static page, work in the browser

The model (1.2 to 2.1 MB) and the transcoder (1.3 MB) are sent to the browser, which runs them in a Web Worker. The server has one small instance; per-click server work would be the bottleneck and would add network latency to every click. After the first load, clicks cost milliseconds to a few hundred milliseconds. The cost is a one-time download and a heavier first load on slow links, handled by gzip, immutable caching, prefetching, retry and a request cap (`RES_CAP = 4`).

### 2.5 Build id and versioned URLs

The build id is a SHA-1 over the static files, the models and the edge transcoder files. Every request for a script or data file carries `?v=<build id>`, files are cached immutably, and a request with an old id gets 409, which the page turns into "This page was just updated. Reload to continue." This removes the old problem of a half-updated cache mixing old scripts with new data.

### 2.6 No resize, no live load tests, latency targets

The instance is t3.small and stays so; no load test is run against the live site. Latency targets: at most 2 seconds in any case, normally below 500 ms. The budget was stated as at most 30 dollars a month, with 10 more allowed if needed; nothing in the project required using the extra amount.

### 2.7 HTTPS postponed

Two ways were prepared and *not applied*: CloudFront (reversible pieces first, no Elastic IP, origin is the instance's DNS name) and Caddy with an sslip.io name on the instance. The decision for now: leave plain HTTP. Consequence: recent Chrome versions may warn about plain HTTP pages; other browsers do not. See `12-operations-runbook.md`, section 4.

### 2.8 Administration without SSH

All remote commands go through SSM Run Command; port 22 is closed. Workflows with a side effect have a confirmation input and tag/ID guards; third-party actions are pinned to full commit hashes. Details: `10-github-actions.md`.

### 2.9 Only this project's AWS resources

Every workflow looks up resources by the project's tags and refuses to continue if the lookup is ambiguous. No other project's resources are touched. This is also why the ECR lifecycle policy is not applied: it deletes, and the decision belongs to the owner.

### 2.10 Collapsible lower sections

The lower part of the page grew long; the sections were regrouped and made collapsible with the same arrow control as "About this algorithm". Collapsed by default where the content is secondary. The layout test checks that open sections do not overflow.

## 3. Things that were tried and changed

| item | what happened |
|---|---|
| pseudo-code losing its layout at some widths | traced to the strip of path chips and long tags; both removed, code lines wrap at every width |
| edge-weights comparison | the figure of the comparison range was corrected to "about 14,000 to 41,000" after re-checking the data |
| directory listing on port 8080 | first documented as "no listing"; testing with `curl` showed the static server does list directories; documentation corrected |
| unverified statements in the AWS and Actions documents | removed (instance type alternatives, runner minutes, first-run details, reboot time) |
| server memory | a first version loaded models on every endpoint and could grow without bound; replaced by caps (`MODEL_CAP`, `VARIANT_CAP`) and a container memory limit |

## 4. Open items (as of the last commit)

1. HTTPS is not set up (the owner's decision for now).
2. Credentials that appeared in conversations or pasted material should be rotated.
3. ECR holds many large untagged images; a lifecycle policy would reduce storage but deletes data, so it needs approval.
4. Port 8080 lists the `fig47/` directory and its sources.
5. The attribution graph's clamp prediction (what happens to the output if a node's value is set by hand) points in the right direction for about 69 percent of tests; the page says so and does not claim more.
6. The accuracy audit of the node-site version (`/demo`) was not completed; it concerns only that page.
7. Most latents are unnamed. This is a property of the method, not a bug.
8. CloudFront could help visitors far from the instance's region; not applied.
