# Frequently asked questions

Short answers with a pointer to the document that explains in full.

## About the site

**What is the site?** An interactive viewer for neural networks trained on CLRS-30 algorithms. You pick one of 12 algorithms, step through a simulation of it on a small graph, and click a node at a step to see an *attribution graph*: which internal features, on which edges, produced the model's prediction there. (`01-overview.md`)

**Which address does what?** `http://98.86.163.27/` is the edge-site transcoder version. `http://98.86.163.27/demo` is the previous node-site version. `http://98.86.163.27:8080/viewer_fig47.html` is the Figure 4 / Figure 7 viewer. (`06-page-reference.md`, `08-server-and-api.md`)

**Which 12 algorithms?** The 12 that the project trained models for; see `04-the-12-algorithms.md` for each one, its inputs, outputs, hints, training scores and reasons.

**What is the model?** A message-passing neural network with 128 hidden units, fully connected message passing, max aggregation, layer normalisation and hint feedback. (`05-model-and-math.md`)

**What is a transcoder?** A sparse replacement for a part of the model. Here it takes the edge input `s_ij` and predicts the edge message `y_ij` through 1,280 features of which only the 8 strongest are kept per edge (TopK, k = 8). Because few features are active, each one can be studied. (`02-glossary.md`)

**What is an attribution graph?** A graph whose nodes are features (and error terms) and whose links say how much one contributes to another or to the output logit, for one prediction. It is computed for the clicked prediction only. (`06-page-reference.md`)

**What are "error nodes"?** The transcoder does not reproduce the model exactly. The part it misses is shown as error nodes, so the graph accounts for the full computation. The share of error is shown; for the worked example in the page reference it is 0.2 percent. (`05-model-and-math.md`)

**What does "sum test" mean?** The contributions into a node, plus the error and bias terms, add up to what the node actually received. It is the check that the graph is exact rather than approximate. (`11-testing-and-verification.md`)

**What does a latent's "meaning" mean?** A short condition on the graph state (for example about the source node or the frontier) that predicts when the latent is active. A meaning is shown only if it passed the held-out test (F1 at least 0.8 for tier A, at least 0.5 for tier B). (`02-glossary.md`, `15-latent-dictionaries.md`)

**Why are most latents unnamed?** Because naming one without evidence would be a guess. Between 3 and 18 of roughly 650 to 1,190 firing latents per algorithm have a validated meaning.

**Are the numbers real?** Yes: they come from the checkpoint and the transcoder running in the browser, or from stored measurements. Nothing is drawn by hand or randomly generated. The truth test checks the page against an independent recomputation. (`11-testing-and-verification.md`)

**Can I trust the clamp / intervention prediction?** Partly. The direction of the predicted change matched in about 69 percent of tests, so treat it as a hint and use the measured result of the intervention. (`13-project-history-and-decisions.md`, section 4)

**What does "retention" mean, and what are tiers A, B, C?** Retention is how much of the model's score survives when the transcoder replaces the original part. The transcoder tiers are: A = validation F1 at least 0.95 and test F1 at least 0.90 (judged at 16 and 64 nodes); B = validation at least 0.95 but test below 0.90 (judged at 16 nodes only); C = validation below 0.95 (indicative only). Latent labels have their own tiers A and B (F1 at least 0.8 and at least 0.5). (`16-transcoder-training-reports.md`, `02-glossary.md`)

**Why do some algorithms score so much lower on 64 nodes?** The models were trained on 16-node graphs and tested on 64-node graphs they never saw. The gap (validation F1 minus test F1) is classed CLEAN when below 10 percentage points and SEVERE when above 40. Five algorithms are SEVERE: dfs (100.0 to 48.1), topological_sort (100.0 to 51.7), mst_kruskal (90.5 to 44.5), floyd_warshall (93.1 to 38.5) and strongly_connected_components (100.0 to 27.3). The other seven are CLEAN, for example bfs 100.0 to 99.8. Why the large ones fail is not established by the project; `04-the-12-algorithms.md` gives a plausible reading for each, marked as such. (`04-the-12-algorithms.md`)

## Using the site

**It says "This page was just updated. Reload to continue."** A new version was deployed while the page was open. Reload.

**Chrome shows a warning page.** The address is plain HTTP; Chrome's HTTPS-first protection warns. Choose to continue. Other browsers do not warn. (`12-operations-runbook.md`)

**The first graph is slow.** The first visit downloads the model and the transcoder (a few megabytes). Later visits come from the browser cache.

**Which graph size should I use?** n = 16 (the default): the meanings and the fidelity numbers were measured there. Larger graphs work but are slower and the meanings may not carry over.

**Can I use my own graph?** Not a hand-drawn one. The page has inputs for the number of nodes (n), the edge probability (p) and the seed, and a **New random graph** button; each combination loads the matching sampler graph. (`06-page-reference.md`)

## Repository and build

**Where is each file explained?** `03-repository-file-map.md` lists every tracked file with a description and whether it goes into the Docker image.

**What goes into the image?** Everything not excluded by `.dockerignore`. The `docs/` folder is excluded, so documentation changes need no deployment.

**What is the build id?** A SHA-1 over the static files, the models and the edge transcoder files. It appears in every versioned URL. (`08-server-and-api.md`)

**How do I regenerate the tables in the docs?** `python3 tools/clrs_docs_generate.py --models <bundles> --reports <clrs-transcoder-out> --results <results_v3>`. (`12-operations-runbook.md`, section 6)

## AWS and workflows

**Which instance runs the site?** One t3.small in us-east-1 with a 16 GB gp3 volume and 2 GiB swap, reached through SSM only. (`09-aws-infrastructure.md`)

**Why is there no HTTPS?** It is prepared but not applied, by decision. Two workflows exist: CloudFront, and Caddy with a free sslip.io name.

**How do I deploy?** Run *build-and-push-ecr*, then *update-container* with `confirm=confirmed`. (`12-operations-runbook.md`, section 2)

**How do I go back to an older version?** Revert in git and deploy, or retag an older ECR image as `latest` and run *update-container*. (`12-operations-runbook.md`, section 3)

**Why is ECR so large?** Every build pushes an image of about 1 GB (compressed), and older ones become untagged but stay. There were 95 records, 63 of them large images, when last counted. A lifecycle policy would remove old ones; it has not been applied because it deletes.

**What does the monthly cost consist of?** The instance, its disk, ECR storage and data transfer. The target is at most 30 dollars a month; ECR storage is the item that grows over time.

**Can I run the workflows from my phone?** They are manual (`workflow_dispatch`), so they can be started from the GitHub web interface or the mobile app.

## Limits

**What the site does not show:** other models, algorithms beyond the 12, hidden-state meanings beyond the validated latents, anything for graphs far larger than 64 nodes.

**What it should not be used to claim:** that the network "understands" the algorithm in general; the five SEVERE algorithms show it does not carry over to 64-node graphs. That the unnamed latents mean nothing; they are unexplained. That the clamp prediction is reliable.
