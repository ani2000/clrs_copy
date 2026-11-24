# GitHub Actions: the twelve workflows, one by one

All automation of the project lives in `.github/workflows/`. There are twelve workflow files. **Every one of them is started by hand** (`on: workflow_dispatch`); nothing runs on push, on a schedule or on a pull request. That is on purpose: a push to `main` must never change the live site or the AWS account by itself.

## 1. Overview

| file | name in the Actions tab | changes anything? | needs `confirm=confirmed`? | purpose in one line |
|---|---|---|---|---|
| `build-and-push-ecr.yml` | Build and push image to ECR | pushes an image to ECR | no | build the container, smoke-test it, push `latest` |
| `update-container.yml` | Update running container (pull latest image) | restarts the container on the instance | yes | deploy: pull the newest image and restart `tracer` |
| `deploy-ec2.yml` | Deploy to EC2 (one instance) | creates a security group and an instance | yes | first-time creation of the machine (and optional replacement) |
| `reboot-instance.yml` | Reboot the instance (soft reboot, same IP) | reboots the instance | yes | soft reboot of the one tagged instance |
| `tune-network.yml` | Tune instance network (BBR congestion control) | two kernel settings on the instance | yes | switch TCP to BBR, or revert to cubic |
| `enable-https.yml` | HTTPS on the instance itself (check / apply / disable) | in `apply`: opens port 443 and starts Caddy | yes (apply, disable) | give the instance an `https://` name (not applied) |
| `setup-cloudfront.yml` | CloudFront in front of the tracer (check / apply / disable) | in `apply`: creates a cache policy and a distribution | yes (apply, disable) | CDN in front of the site (not applied) |
| `ec2-diagnose.yml` | EC2 diagnose (read-only) | no | no | status, ECR images, CPU credits, memory, container logs |
| `aws-discover.yml` | AWS discovery (read-only) | no | no | what exists in the account that this project could clash with |
| `cleanup-old-resources.yml` | Clean up old clrs-circuit-tracer resources | only in `delete` mode | yes (delete) | inspect or delete exactly two leftover resources |
| `pre-demo-check.yml` | Pre-demo check | no | no | open the site in a headless browser and load all 12 algorithms |
| `cross-browser-check.yml` | Cross-browser check | no | no | Chromium, Firefox and WebKit (Safari engine) |

The usual deployment is the first two: build, then update.

## 2. Things all workflows share

### 2.1 Trigger and permissions

```yaml
on:
  workflow_dispatch:
permissions:
  contents: read
  id-token: write      # only used when the repository variable AWS_ROLE_ARN is set (OIDC)
```

`contents: read` lets the workflow read the repository; `id-token: write` lets it request an OIDC token, which is used only when an AWS role is configured. The two browser-check workflows ask for `contents: read` only because they never touch AWS.

### 2.2 AWS credentials: two alternatives, one is chosen

Every AWS workflow starts with two steps, exactly one of which runs:

| step | runs when | what it does |
|---|---|---|
| AWS credentials (OIDC role) | the repository variable `AWS_ROLE_ARN` is not empty | assumes that role with a short-lived token; no stored key |
| AWS credentials (access keys) | `AWS_ROLE_ARN` is empty | uses the repository secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` |

At the moment the second form is in use (the diagnose log shows "Proceeding with IAM user credentials"). The values of the secrets are never printed (GitHub masks them as `***`); this documentation does not contain them.

### 2.3 Pinned actions

Third-party actions are referenced by full commit hash with the version in a comment, so they cannot be changed underneath the workflow:

| action | version | hash used |
|---|---|---|
| `actions/checkout` | v4 | `11d5960a326750d5838078e36cf38b85af677262` |
| `aws-actions/configure-aws-credentials` | v4 | `7474bc4690e29a8392af63c5b98e7449536d5c3a` |
| `aws-actions/amazon-ecr-login` | v2 | `03f1aad4c6c7ffd436567f42f9384779290529bd` |
| `docker/setup-buildx-action` | v3 | `8d2750c68a42422c14e847fe6c8ac0403b4cbd6f` |
| `docker/build-push-action` | v6 | `10e90e3645eae34f1e60eeb005ba3a3d33f178e8` |

### 2.4 The guards

Workflows that act on the instance never accept a free-text target. They look the instance up themselves and stop unless the answer is exactly right:

1. **Tag guard.** `aws ec2 describe-instances --filters Name=tag:Name,Values=clrs-circuit-tracer Name=tag:Project,Values=clrs-circuit-tracer Name=instance-state-name,Values=running`. The result must contain exactly one instance ("Expected exactly one running tagged instance, found: ...").
2. **Expected-ID guard** (reboot, tune-network, enable-https, setup-cloudfront): the found ID must equal `i-0877713b5583c8533`.
3. **Confirm gate:** the `confirm` input must be exactly `confirmed`; otherwise the job is skipped (`if: inputs.confirm == 'confirmed'`) or the step refuses.
4. **Both-tags rule** in `deploy-ec2`: only resources that carry *both* tags are ever reused or terminated; a resource with the right name but no `Project` tag makes the workflow stop and ask.

### 2.5 Environment constants

| name | value | used by |
|---|---|---|
| `AWS_DEFAULT_REGION` | `us-east-1` | all AWS workflows |
| `NAME` / `PROJECT` | `clrs-circuit-tracer` | all instance workflows |
| `EXPECTED_ID` | `i-0877713b5583c8533` | reboot, tune-network, enable-https, setup-cloudfront |
| `ECR_REPO` | `clrs-circuit-tracer` | build-and-push-ecr, deploy-ec2 |
| `AZ` | `us-east-1a` | deploy-ec2 |
| `SG_NAME` | `clrs-circuit-tracer-app-sg` | deploy-ec2 |
| `PROFILE` | `ecsInstanceProfile` | deploy-ec2 |
| `CADDY_IMAGE` | `caddy:2.8.4-alpine` | enable-https |

### 2.6 How to run a workflow

From the website: repository, tab Actions, pick the workflow in the left list, "Run workflow", fill the inputs, run. From a terminal with the GitHub CLI (`gh`) and a token that may use Actions:

```sh
gh workflow run build-and-push-ecr.yml -R ani2000/clrs_copy --ref main
gh run list   -R ani2000/clrs_copy --workflow build-and-push-ecr.yml -L 1
gh run view   <run id> -R ani2000/clrs_copy --log
gh workflow run update-container.yml -R ani2000/clrs_copy --ref main -f confirm=confirmed
```

Inputs are passed with `-f name=value`. Booleans are passed as strings (`-f create_repo_if_missing=true`).

---

## 3. `build-and-push-ecr.yml`: Build and push image to ECR

**Purpose.** Build the container from the repository, prove that it starts and answers, and push it to ECR as `clrs-circuit-tracer:latest`.

**Input.** `create_repo_if_missing` (boolean, default false): create the ECR repository (with scan-on-push and project tags) if it does not exist.

**Runner.** `ubuntu-latest`. Typical duration: about 5 minutes (earlier runs took 4 min 53 s and 4 min 59 s).

**Steps in order**

1. `actions/checkout` with `lfs: true`, so the large checkpoint files are real files and not pointers.
2. **Verify checkpoints are real files, not LFS pointers:** `test "$(stat -c%s checkpoints_v3/bfs/best.pkl)" -gt 100000`. A pointer file is a few hundred bytes; a real checkpoint is several megabytes. If this fails the build stops here (a build with pointer files would produce an image that cannot load any model).
3. AWS credentials (one of the two forms of section 2.2).
4. **Create ECR repository (only if requested):** `aws ecr describe-repositories ... || aws ecr create-repository --repository-name clrs-circuit-tracer --image-scanning-configuration scanOnPush=true --tags Key=Name,Value=... Key=Project,Value=...`.
5. **Login to ECR** (`amazon-ecr-login`), which also provides the registry host name.
6. **Set up Docker Buildx.**
7. **Build image for smoke test (not pushed):** `docker/build-push-action` with `platforms: linux/amd64`, `load: true`, `push: false`, tag `clrs-smoke:test`, using the GitHub Actions cache (`cache-from: type=gha`, `cache-to: type=gha,mode=max`).
8. **Smoke test:** runs the image locally on the runner with ports 8766 and 8767 and, for up to 60 attempts five seconds apart, requests `http://127.0.0.1:8766/viewer3.html` and `http://127.0.0.1:8767/viewer_fig47.html`. Both must answer 200. On failure it prints the container logs and fails.
9. **Push to ECR (linux/amd64):** builds again (from the cache) and pushes `<registry>/clrs-circuit-tracer:latest`.
10. **Print image URI** into the run summary.

**What it does not do.** It does not touch the instance. The new image only becomes live when the *update* workflow runs.

**Why a smoke test.** The Dockerfile generates `models/` at build time with `python model_export.py models`; if that step or the server start were broken, the smoke test catches it before anything is pushed.

**Cost.** Free minutes; ECR storage grows with every push (see `09-aws-infrastructure.md`, section 5.2).

---

## 4. `update-container.yml`: Update running container

**Purpose.** Deploy. Pull the newest `latest` image on the instance and restart the container.

**Input.** `confirm` (required): must be `confirmed`.

**Guards.** Confirm gate; tag guard (exactly one running tagged instance). It does not check the expected ID.

**What it does.** Reads the account number to build the registry name, then sends one SSM Run Command (`AWS-RunShellScript`) to the instance:

```sh
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <registry>
docker pull <registry>/clrs-circuit-tracer:latest
docker rm -f tracer
docker run -d --restart always --memory 1500m --memory-swap 1800m --name tracer \
  --log-opt max-size=10m --log-opt max-file=3 -p 80:8766 -p 8080:8767 <registry>/clrs-circuit-tracer:latest
docker image prune -f
docker ps
```

then waits for the command to finish (`aws ssm wait command-executed`) and prints its status, output and errors.

**Effect on visitors.** The container is removed and recreated, so the site is unreachable for a short time (the new process needs a few seconds to start listening; the first graph of an algorithm after that needs the models in the browser cache or a re-download). Pages already open in a browser show "This page was just updated. Reload to continue." the next time they request a versioned file, because the build id changed (see `08-server-and-api.md`).

**After running.** Check `http://98.86.163.27/api/algos` answers 200 and the page source shows the new `BUILD`.

---

## 5. `deploy-ec2.yml`: Deploy to EC2 (one instance)

**Purpose.** Create the machine from nothing (or replace it). It is the only workflow that creates compute.

**Inputs**

| input | meaning |
|---|---|
| `confirm` | must be `confirmed` |
| `instance_type` | choice `t3.small` (default) or `t3.micro` |
| `replace_existing` | boolean: terminate the existing *tagged* instance (by its ID, only it) before launching |

**What it creates.** Exactly one security group and one instance, both tagged `Name` and `Project`; no key pair, no port 22, no IAM, VPC, subnet, gateway, route table or Elastic IP.

**Steps**

1. Credentials.
2. **Find network (read-only; fails safely):** exactly one default VPC; a default subnet in `us-east-1a` that is available and assigns public IPs; the instance type is offered in that zone.
3. **Pre-checks (read-only):** finds instances carrying both tags (the only ones it may ever terminate) and instances with the name but without the project tag (it stops, never touches them); refuses to continue when a tagged instance exists and `replace_existing` is false; reuses the security group only if it has the project tag; requires the ECR image `latest` to exist; reads the instance profile's role and checks that it has an ECR read policy, and warns when it lacks `AmazonSSMManagedInstanceCore`.
4. **Terminate previous tagged instance** (only if requested; by ID; waits until terminated).
5. **Create security group if missing (tagged):** ingress TCP 80 and TCP 8080 from 0.0.0.0/0.
6. **Launch instance:** newest Amazon Linux 2023 x86_64 AMI; credit specification `standard`; IMDSv2 required (hop limit 2); a 16 GB encrypted gp3 root volume deleted on termination; the user-data script of `09-aws-infrastructure.md` section 4; tags on the instance and the volume; waits until running and writes the address into the run summary ("allow about 5 minutes for boot, image pull and model load").

**Run it only** when the instance is gone or must be rebuilt. For ordinary deployments use *update-container*.

---

## 6. `reboot-instance.yml`: Reboot the instance (soft reboot, same IP)

**Purpose.** Soft-reboot the instance. A reboot keeps the public IP, the disk and the settings in `/etc/sysctl.d`; the container comes back by itself (`--restart always`).

**Input.** `confirm` = `confirmed`.

**Guards.** Confirm gate; tag guard; expected ID `i-0877713b5583c8533`.

**Action.** `aws ec2 reboot-instances --instance-ids <id>`. Creates, deletes and resizes nothing. Expect the site to be unreachable while the machine restarts and Docker brings the container back (typically a minute or two).

---

## 7. `tune-network.yml`: Tune instance network (BBR)

**Purpose.** Change two kernel network settings on the instance so that downloads on long or lossy paths (a visitor far from Virginia) are faster: TCP congestion control to BBR and the queueing discipline to `fq`.

**Inputs.** `confirm` = `confirmed`; `mode` = `apply` (default) or `revert`.

**Guards.** Confirm gate; tag guard; expected ID.

**Apply (through SSM).** Prints the current settings, loads the `tcp_bbr` module, checks that `bbr` is available, writes

```text
net.core.default_qdisc=fq
net.ipv4.tcp_congestion_control=bbr
```

to `/etc/sysctl.d/99-bbr.conf`, runs `sysctl --system`, prints the new settings.

**Revert.** Removes that file and sets cubic and `fq_codel` again.

**Persistence.** Survives reboots (file in `/etc/sysctl.d`); lost if the instance is replaced.

---

## 8. `enable-https.yml`: HTTPS on the instance itself

**Purpose.** An `https://` address without a new AWS resource and without extra cost. **Not applied** to the live instance at the time of writing.

**Inputs.** `mode` = `check` (default), `apply` or `disable`; `confirm`.

**Guards.** Tag guard; expected ID; exactly one security group on the instance, and that group must carry the project tag (otherwise "refusing to touch it").

**Check (read-only).** Prints the inbound rules of the group, does a dry run of opening port 443 to see whether the key is allowed to, and checks that `<ip-with-dashes>.sslip.io` resolves.

**Apply (`confirm=confirmed`).**

1. Opens TCP 443 from anywhere on the project-tagged group (only if not already open).
2. Through SSM: creates `/opt/caddy`, writes the Caddyfile

   ```text
   {
     auto_https disable_redirects
   }
   https://98-86-163-27.sslip.io {
     tls {
       issuer acme {
         disable_http_challenge
       }
     }
     encode zstd gzip
     reverse_proxy 127.0.0.1:80
   }
   ```

   pulls `caddy:2.8.4-alpine`, removes any old `caddy` container and runs `docker run -d --restart always --name caddy --network host --memory 200m --log-opt max-size=10m --log-opt max-file=3 -v /opt/caddy/Caddyfile:/etc/caddy/Caddyfile:ro -v caddy_data:/data -v caddy_config:/config caddy:2.8.4-alpine`.
3. Polls `https://98-86-163-27.sslip.io/api/algos` up to 24 times, ten seconds apart, until it answers 200, then prints `URL=https://98-86-163-27.sslip.io/viewer3.html`. If it never does, it says the plain http address is unchanged and still works.

**Disable.** Removes only the `caddy` container (the port 443 rule stays).

**Notes.** The tracer's own container and port 80 are untouched (the certificate challenge uses port 443 only). The name embeds the instance's public IP; after a stop and start the workflow must be run again.

---

## 9. `setup-cloudfront.yml`: CloudFront in front of the tracer

**Purpose.** A content delivery network in front of the site: HTTPS on a `*.cloudfront.net` name, nearer caches for far-away visitors. **Not applied.**

**Inputs.** `mode` = `check`, `apply` or `disable`; `confirm`.

**Guards.** Tag guard; expected ID.

**Check (read-only).** Lists any existing tagged Elastic IPs, distributions with the comment `clrs-circuit-tracer`, the custom cache policy `clrs-circuit-tracer-versioned` and CloudWatch alarms with the project prefix; probes whether the key may list distributions and cache policies.

**Apply.** Creates, only if missing:

- a **cache policy** `clrs-circuit-tracer-versioned` (default TTL 0, maximum one year; cache key includes the whole query string, so `?v=BUILD` separates builds; cookies and headers are not forwarded; gzip and brotli accepted), which means "cache for as long as the origin's headers say, and store nothing without them";
- a **distribution** with the instance's current public DNS name as the origin (HTTP, port 80), HTTP/2 and HTTP/3, IPv6, price class "all", viewer protocol policy "redirect to https", and the cache behaviour for `/models/*`, `/api/weights`, `/api/feature_labels`, `/api/instance`, `/api/algos` and `*.js`; the default behaviour uses the managed "CachingDisabled" policy.

It waits for the distribution to deploy (up to about 15 minutes) and prints its address. Nothing about the instance changes, no Elastic IP and no alarm are created.

**Disable.** Sets `Enabled=false` on the distribution (nothing is deleted).

**Caveat.** The origin is the instance's public DNS name, which changes if the instance is stopped and started.

---

## 10. `ec2-diagnose.yml`: EC2 diagnose (read-only)

**Purpose.** A health report without changing anything.

**Output sections**

1. ECR image info: for every image in the repository the tags, the push time and the size.
2. The tagged instance's ID, launch time and public IP; its CPU credit specification.
3. CloudWatch statistics for `CPUCreditBalance` and `CPUUtilization` (five-minute averages over the last hour).
4. Through SSM (waits 20 seconds, then reads the result): `docker ps -a`, the last 60 lines of the container log, `free -m`, `swapon -s`, the first lines of `top`, and any kernel lines about out-of-memory kills.

Use it before a demo and after a deployment. A sample of its output is summarised in `09-aws-infrastructure.md`, section 8.

---

## 11. `aws-discover.yml`: AWS discovery (read-only)

**Purpose.** Look at what exists, narrowly: identity, the ECR repository's `latest` image, the default VPC and the default subnet of `us-east-1a`, instances / security groups / key pairs whose name starts with `clrs-circuit-tracer`, and the instance profile `ecsInstanceProfile` (its role and attached policies). It never lists other projects' resources.  Its purpose is to learn the state of the account before anything is created.

---

## 12. `cleanup-old-resources.yml`: Clean up old resources

**Purpose.** Remove two leftovers of a previous deployment, one per run, only with explicit confirmation.

**Inputs.** `target` (a fixed choice of two IDs: the old security group `sg-0dc604c9b3c0acfc5` and the old key pair `key-0862ad30aa3ca38a4`), `mode` (`dry-run` default, or `delete`), `confirm`.

**Safety.** The script re-checks the ID *and* the resource's name (`clrs-circuit-tracer-sg`, `clrs-circuit-tracer-key`) and refuses on any mismatch. The dry run prints the evidence: for the security group the network interfaces, instances and other groups that use or reference it; for the key pair the instances that use it. Deletion needs `mode=delete`, `confirm=confirmed` and the evidence "safe to delete: yes". The default is read-only.

---

## 13. `pre-demo-check.yml`: Pre-demo check

**Purpose.** Before showing the site, load all twelve algorithms in a real browser from outside and see whether each step works and how long it takes.

**Inputs.** `url` (required; for example `http://98.86.163.27` or an `https://` address), `max_seconds` (longest allowed time per step, default 30).

**What it does.** On an `ubuntu-24.04` runner it runs `node --experimental-websocket tools/predemo_check.mjs "$URL" "$MAXS"`: headless Chrome, for each algorithm waits for the first graph, for the attribution graph and for a node click, with timings. Touches nothing in AWS.

**Important.** Recent Chrome versions refuse plain `http://` addresses of public sites (HTTPS-first), so the check against `http://98.86.163.27` can report "blocked by client" although the site is healthy; the same check passes on an `https://` address, or in a Chrome with the HTTPS upgrade features switched off (that is how the live tests of this project were run).

---

## 14. `cross-browser-check.yml`: Cross-browser check

**Purpose.** Check the page in the three browser engines: Chromium, Firefox and WebKit (the Safari engine, including a phone-sized window).

**Input.** `url` (required).

**What it does.** Installs Playwright 1.49.1 and the three browsers on the runner (for this run only), copies `tools/browsers_check.mjs` and runs it: five algorithms in each browser, pass/fail with timings. Timeout 30 minutes. Touches nothing in AWS.

---

## 15. Typical sequences

| situation | sequence |
|---|---|
| ship a change to the page or the data | merge to `main` -> *build-and-push-ecr* -> *update-container* (`confirm=confirmed`) -> check `/api/algos` and the page source's `BUILD` |
| before a demo | *ec2-diagnose* (is the instance healthy?) -> *pre-demo-check* with the address -> open the page yourself |
| the site is slow far away | *tune-network* `apply` (BBR); consider *setup-cloudfront* |
| Chrome shows a warning page | *enable-https* `check`, then `apply` (opens 443, starts Caddy) and use the `https://98-86-163-27.sslip.io` address |
| the container misbehaves | *ec2-diagnose* to read the log; *update-container* to restart it; *reboot-instance* as a last resort |
| the machine is lost | *deploy-ec2* (`confirm=confirmed`; `replace_existing` only if a broken tagged instance still exists) |

## 16. Limits and gaps

- There is no workflow that rolls back to an older image; the procedure is described in `12-operations-runbook.md`.
- There is no scheduled health check or alarm; the diagnose and pre-demo workflows are run by hand.
- There is no lifecycle policy on the ECR repository; old images accumulate.
- The checks of the page itself (`verification/`) are run from a terminal during development; they are not part of the workflows (the two browser workflows run the lighter scripts in `tools/`).
