# Operations runbook

Step-by-step instructions for running the site: deploying a change, checking health, preparing for a demo, recovering from problems, and keeping cost and security in order. Commands use the GitHub CLI (`gh`) for the workflows and the AWS CLI for the few things done outside them. Replace `<...>` placeholders. Keep tokens and keys out of shell history, files and chats (pass them through environment variables only).

## 1. The 60-second status check

| question | command or action | healthy answer |
|---|---|---|
| is the application up? | `curl -s -o /dev/null -w '%{http_code}\n' http://98.86.163.27/api/algos` | `200` |
| is the page served? | open `http://98.86.163.27/` | the page loads; "bfs ... in-browser model" appears top right |
| which build? | `curl -s -H 'Accept-Encoding: gzip' http://98.86.163.27/ \| gzip -dc \| grep -o "BUILD = '[0-9a-f]*'"` | the id of the last deployment |
| previous version up? | open `http://98.86.163.27/demo` | the page with the "previous demo" tag |
| figures page up? | open `http://98.86.163.27:8080/viewer_fig47.html` | the Figure 4 / 7 viewer |
| instance, memory, credits, logs | run the `ec2-diagnose` workflow | see `09-aws-infrastructure.md` section 8 |

## 2. Deploying a change

### 2.1 Preconditions

- The change is merged into `main` (the workflows build the branch they are started on; use `--ref main`).
- The local checks pass (`11-testing-and-verification.md`, section 9). For changes to the page: `check_inline.js`, `test_visual.js`, the browser tests; for changes to the engine: tests 4, 6, 7, 9 and the truth test.
- If a data file changed (`edge_tc/`, `algos/`), the build id changes automatically; nothing else needs touching.
- Git LFS: the checkpoints must exist as real files in the GitHub repository (the build fails with an explicit message if only pointers were fetched).

### 2.2 The two workflows

```sh
export GH_TOKEN=<a token that may run Actions>          # do not paste tokens into files
gh workflow run build-and-push-ecr.yml -R ani2000/clrs_copy --ref main
gh run list -R ani2000/clrs_copy --workflow build-and-push-ecr.yml -L 1        # note the run id
gh run view <run id> -R ani2000/clrs_copy                                       # wait for "completed success" (about 5 minutes)
gh workflow run update-container.yml -R ani2000/clrs_copy --ref main -f confirm=confirmed
gh run list -R ani2000/clrs_copy --workflow update-container.yml -L 1
```

The update takes well under a minute. The site is unreachable for a few seconds while the container is replaced, and pages that were open show "This page was just updated. Reload to continue." when they next request a versioned file.

### 2.3 Verify

1. `curl .../api/algos` answers 200 (retry for up to 30 seconds).
2. The build id in the page source changed.
3. Open the page, check the first graph and one click; open `/demo`.
4. Optional but recommended before a demo: `pre-demo-check` (see 4) and `ec2-diagnose`.
5. For a data change, spot-check a file: `curl -s -o /dev/null -D - "http://98.86.163.27/edge_tc/bfs.latents.json?v=<build id>"` answers 200 with `Content-Encoding: gzip` and the immutable cache header.

### 2.4 What not to do

- Do not run *deploy-ec2* to deploy: it creates machines. Use *update-container*.
- Do not skip the build: *update-container* only pulls what is in ECR.
- Do not set the build id by hand: it is derived from the files.

## 3. Rollback

### 3.1 Roll back the page or the data (the usual case)

The simplest and safest: revert the commit on `main` (`git revert <commit>`; push), then run the build and the update workflows as in 2.2. It produces a new image from the old files. Takes about 6 minutes.

### 3.2 Roll back to a previous *image* without rebuilding

Every push leaves the previous image in ECR (untagged). Find it and give it the tag `latest` again, then run *update-container*:

```sh
aws ecr describe-images --repository-name clrs-circuit-tracer \
  --query 'sort_by(imageDetails,&imagePushedAt)[-6:].[imageDigest,imagePushedAt,imageSizeInBytes]' --output text
# choose the digest of the image you want, then:
MANIFEST=$(aws ecr batch-get-image --repository-name clrs-circuit-tracer \
  --image-ids imageDigest=<digest> --query 'images[0].imageManifest' --output text)
aws ecr put-image --repository-name clrs-circuit-tracer --image-tag latest --image-manifest "$MANIFEST"
```

(If `put-image` asks for it, add `--image-manifest-media-type` with the value `batch-get-image` shows for that image.) Then `gh workflow run update-container.yml ... -f confirm=confirmed`. This only works while the old image still exists; if an ECR lifecycle policy is ever added, keep the most recent several images.

### 3.3 Roll back only `/demo` or only the main page

The two pages are separate files (`demo.html`, `viewer3.html`) served by the same server; they share the scripts and data. Reverting one file in git and redeploying changes only that page.

## 4. Before a demo (checklist)

1. Run `ec2-diagnose`: the instance is running, CPU credit balance is high, memory has headroom (used well below total), the container is `Up`, no recent out-of-memory lines.
2. Open `http://98.86.163.27/` yourself on a different machine and network if possible; click through two or three algorithms.
3. Run `pre-demo-check` with `url=https://...` if an HTTPS address exists; with the plain `http://` address Chrome's HTTPS-first protection can report "blocked by client" although the site is healthy (the check passes in Chrome with the HTTPS upgrade features switched off, which is how this project tested the live site).
4. Know which browsers the audience uses: Safari and Firefox open plain `http://` pages without a warning page; Chrome 157 and later may show a warning page or block the address until the visitor chooses to continue. This is the main reason an HTTPS address is on the open list.
5. Decide the algorithms to show: bfs (CLEAN, 99.8% at 64 nodes), dijkstra, bellman_ford are the cleanest stories; dfs and strongly_connected_components show the SEVERE case (validation F1 100.0% at 16 nodes, test 48.1% and 27.3% at 64 nodes).
6. Do not deploy within about two hours before the demo (the project's own rule); deploying changes the build id and restarts the container.
7. Keep the default graph size (n = 16): meanings and fidelity were measured there.

## 5. Troubleshooting

| symptom | likely cause | what to do |
|---|---|---|
| page does not load at all | container down or instance stopped | `ec2-diagnose`; `update-container` restarts the container; `reboot-instance` as a last resort |
| browser shows a warning page about an insecure site | plain `http://` and Chrome's HTTPS-first | choose "continue"; or set up HTTPS (`enable-https.yml`) |
| "This page was just updated. Reload to continue." | a deployment happened while the page was open | reload |
| first graph is slow on a far-away connection | cold cache: 1.2 to 2.1 MB model plus 1.3 MB transcoder to download | wait once; later visits use the browser cache; consider BBR (`tune-network`) or CloudFront |
| a graph is slow to appear for n = 32 | the new graph at n = 32 takes about 1.5 s of browser compute | keep n at 16 for demos |
| "Could not trace this prediction just now. Click the node again to retry." | the worker failed or was evicted | click the node again; reload if it persists |
| `/api/graph` or `/api/contexts` very slow | the JAX model is loading on the server (first request per algorithm) | wait; the page does not depend on these for the main version |
| container keeps restarting | out-of-memory kills (kernel log shows `Memory cgroup out of memory`) | `ec2-diagnose`; check `MODEL_CAP`, `VARIANT_CAP`, the container limit; reduce simultaneous server-side runs |
| workflow fails at "Verify checkpoints are real files" | LFS pointers instead of files | make sure LFS objects are in the GitHub repository (`git lfs ls-files`, push LFS objects) |
| workflow fails with "Expected exactly one running tagged instance" | no instance, or two with the tags | check the AWS console; never resolve by removing tags of other resources |
| workflow fails with an AWS authorization error | the key lacks a permission | read the denied action in the log; widen the policy only for that action, or use the console |
| `update-container` succeeds but the page is old | the build did not finish or pushed a different branch | check the build run and `--ref main`; compare `BUILD` |
| the figures page (port 8080) shows nothing | the static server is down | `ec2-diagnose`: the container log must show `serving fig47/ static files` |

## 6. Routine maintenance

| task | how often | how |
|---|---|---|
| rotate AWS keys and GitHub tokens | after any exposure; otherwise regularly | create new keys, update the repository secrets, delete the old keys |
| prune old ECR images | when storage grows | lifecycle policy (keep the newest few, expire untagged after a few days); needs a decision because it deletes |
| check the instance's CPU credits and memory | before demos | `ec2-diagnose` |
| security group review | when something changes | only 80 and 8080 should be open to the world |
| update base images and libraries | rarely | the pins in `requirements.txt` and the base image tag in the `Dockerfile` are exact; test locally, then deploy |
| regenerate the documentation tables | after any data change | `python3 tools/clrs_docs_generate.py --models <bundles> --reports <clrs-transcoder-out> --results <results_v3>` |

## 7. Changing the data

### 7.1 A new edge-site transcoder (re-trained)

1. Put the new `report.json` and `transcoder.npz` files in the `clrs-transcoder-out` layout.
2. `python3 clrs_export_edge_tc.py <clrs-transcoder-out> edge_tc` (rewrites `.bin`, `.json` and the summary; prints the retention per algorithm).
3. Check with tests 1, 2, 4, 8.
4. Re-measure the meanings: `verification/clrs-edge/fetch_graphs.sh <folder>`, then `node verification/clrs-edge/clrs_edge_latents.js <algo>` for each algorithm (about 30 to 170 seconds each; they can run in parallel), writing into `edge_tc/`; check with test 9.
5. Regenerate the documents (section 6).
6. Merge, build, update.

### 7.2 A new checkpoint

`checkpoints_v3/<algo>/best.pkl` is the model all the rest depends on. A different checkpoint invalidates the transcoders (they are valid only for the exact weights they were trained on; test 2 would show an FVU near 1), the node-site files and the latent tables. Do not replace a checkpoint without redoing the whole chain.

### 7.3 Adding a thirteenth algorithm

It needs: the checkpoint; entries in `config.py` (`ALGOS`), `pipeline.py` targets and `algos/<algo>/*` (for `/demo`); a transcoder in the training notebook and the export; the pseudo-code lines and narration in `pseudo_block.js` (and re-inlining into `viewer3.html` and `demo.html`); an entry in `SUMMARY_V3` and `TRAIN_V3` in the pages and the reasons in the algorithm card; the tests. It is a project, not a switch.

## 8. Cost control

- Keep one instance of type t3.small (the "no resize" rule).
- Do not attach unneeded resources (Elastic IPs, load balancers, NAT gateways).
- ECR images accumulate: apply a lifecycle policy when allowed.
- CloudFront, if ever applied, adds a small usage-based charge.
- Stop nothing: stopping and starting changes the public address.

## 9. Security hygiene

- Credentials never go into the repository, the documents or the chat. The workflows use repository secrets.
- Rotate any key or token that has been exposed; the GitHub tokens used for pushing and the AWS access key should be rotated after the project's current phase.
- The only open ports are 80 and 8080 (plain HTTP). There is no user data on the site.
- All administration is through SSM; do not open port 22.
- Port 8080 shows a directory listing of `fig47/`; if that is unwanted, remove the second server from `start.sh` or serve only the viewer files.

## 10. Where to look when something is unclear

| topic | document |
|---|---|
| a word | `02-glossary.md` |
| a file | `03-repository-file-map.md` |
| an algorithm | `04-the-12-algorithms.md` |
| the maths | `05-model-and-math.md` |
| a part of the page | `06-page-reference.md`, `06b-hover-text-catalog.md` |
| a data format | `07-data-files-and-formats.md` |
| the server | `08-server-and-api.md` |
| AWS | `09-aws-infrastructure.md` |
| a workflow | `10-github-actions.md` |
| a test | `11-testing-and-verification.md` |
| history and decisions | `13-project-history-and-decisions.md` |
| a quick question | `14-faq.md` |
