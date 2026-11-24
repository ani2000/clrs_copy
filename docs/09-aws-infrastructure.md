# AWS infrastructure: the EC2 instance, the ECR images and everything around them

This document describes what the site runs on. Everything here was read from the repository's workflows and from read-only inspections of the real account (the `ec2-diagnose` workflow, run on 24 November 2026). Account numbers, keys and tokens are deliberately not written down anywhere in these documents.

## 1. The big picture in one diagram

```text
 you (browser)                         GitHub                              AWS (us-east-1)
 -------------                         ------                              ---------------
                                    repo ani2000/clrs_copy
                                      |  push to main
                                      v
                          workflow "Build and push image to ECR"  --push--> ECR repository clrs-circuit-tracer
                                      |                                      tag :latest (+ many untagged older images)
                          workflow "Update running container"
                                      |  SSM Run Command (no SSH)
                                      v
 http://98.86.163.27  <--- port 80 --- EC2 instance i-0877713b5583c8533 (t3.small, Amazon Linux 2023)
 http://98.86.163.27:8080 <- port 8080 -  docker container "tracer" (image clrs-circuit-tracer:latest)
                                          |- python server_128.py          (container port 8766 -> host 80)
                                          '- python fig47/static_server.py (container port 8767 -> host 8080)
```

There is exactly one machine. There is no load balancer, no auto scaling, no database, no queue, no CDN and no HTTPS in front of it today. That is a deliberate choice for a low-traffic demo: the cost is a few dollars a month for the instance plus small amounts for the disk, the public address and the image repository.

## 2. Inventory of AWS resources used

| resource | name or id | notes |
|---|---|---|
| Region | `us-east-1` (N. Virginia) | all workflows set `AWS_DEFAULT_REGION=us-east-1` |
| Availability zone | `us-east-1a` | `deploy-ec2.yml` pins `AZ=us-east-1a` |
| EC2 instance | `i-0877713b5583c8533` | tags `Name=clrs-circuit-tracer`, `Project=clrs-circuit-tracer`; launched 2026-11-04 19:23:23 UTC |
| Instance type | `t3.small` | 2 vCPU, 2 GiB (1,909 MiB usable), burstable |
| Operating system | Amazon Linux 2023 (x86_64) | AMI chosen at launch as the newest `al2023-ami-2023*-x86_64` |
| Disk | 1 x EBS `gp3`, 16 GB, encrypted, deleted on termination | device `/dev/xvda`; tagged like the instance |
| Swap | 2 GiB swap file `/swapfile` | created by the launch script; also in `/etc/fstab` |
| Public IPv4 | 98.86.163.27 | automatically assigned, **not** an Elastic IP |
| Public DNS name | `ec2-98-86-163-27.compute-1.amazonaws.com` | derived from the IP; changes with it |
| IPv6 | assigned by the subnet | the site is not served over it in any special way |
| Security group | `clrs-circuit-tracer-app-sg` (`sg-0d5fcca95d559bf57`) | inbound TCP 80 and 8080 from 0.0.0.0/0; tagged `Project=clrs-circuit-tracer`; no port 22 |
| VPC / subnet | the account's default VPC, default subnet of us-east-1a | the launch workflow refuses to proceed unless the subnet assigns public IPs |
| IAM instance profile | `ecsInstanceProfile` | gives the instance permission to pull images from ECR and (if attached) to take SSM commands |
| SSM | Systems Manager Run Command | the only way the workflows administer the instance |
| ECR repository | `clrs-circuit-tracer` | image scanning on push; one tag, `latest` |
| CloudWatch | default EC2 metrics | the diagnose workflow reads `CPUCreditBalance` and `CPUUtilization` |
| GitHub secrets | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | an IAM user's key pair used by the workflows (names only; values are never written down) |

Two older resources from a previous deployment are named in `cleanup-old-resources.yml` (a security group `clrs-circuit-tracer-sg` and a key pair `clrs-circuit-tracer-key`). The workflow can inspect and, only on explicit request, delete exactly those two; nothing has been deleted by default.

### What is deliberately *not* used

- No Elastic IP (it would cost an extra charge for a second public IPv4 address if unattached, and the demo does not need a fixed address).
- No SSH access: no key pair is attached to the instance and port 22 is closed. Administration goes through SSM.
- No Application Load Balancer, no NAT gateway, no RDS, no S3, no Lambda.
- No CloudFront and no HTTPS yet. Workflows for both exist (`setup-cloudfront.yml`, `enable-https.yml`) and have been checked in read-only mode but never applied (see section 9).

## 3. The EC2 instance in detail

### 3.1 Instance type and why

`t3.small` was chosen after the question in `DEPLOYMENT_PROMPT.md`: a CPU-only workload, tiny models, well under 1 GB of working memory in steady state, low and sparse traffic. The launch workflow also offers a `t3.micro` (1 GiB); the project settled on `t3.small`, and the earlier memory incident (section 3.3) showed that even 2 GiB needs care.

Burstable instances run at a *baseline* share of the CPU and spend *credits* to go above it. This instance uses **standard** credit mode (the launch call sets `CpuCredits=standard`), which means it can never go into debt and never incurs a surplus-credit charge. The measured balance on 24 November was **576 credits for the whole last hour**, which is the maximum a t3.small can hold (24 credits earned per hour over 24 hours), and the CPU utilisation averaged **0.43% to 0.48%**. The machine is almost idle; the credit system plays no role at the current load.

### 3.2 What is on the disk

The operating system, Docker and the image layers. No application data is stored outside the container: the site is stateless (every graph is sampled on request, nothing is saved from visitors). Because of that there is nothing to back up except the repository itself and the checkpoints (also in Git LFS).

### 3.3 Memory budget

| item | value |
|---|---|
| total RAM | 1,909 MiB |
| used at the time of the check | 837 MiB (about 44%) |
| buff/cache | 1,002 MiB |
| available | 913 MiB |
| swap file | 2,047 MiB, 31 MiB used |
| container memory limit | 1,500 MB (`--memory 1500m`), 1,800 MB with swap (`--memory-swap 1800m`) |

The process inside the container is a Python program with JAX loaded. Its peak is controlled by four settings (section 7 of `08-server-and-api.md`): at most `MODEL_CAP=3` models resident, `VARIANT_CAP=3` compiled program variants, at most two simultaneous model runs and one context search at a time. Almost all of the interactive work happens in the visitor's browser, so a visitor does not add load beyond a few small requests.

The kernel log of the instance still contains three lines `Memory cgroup out of memory: Killed process ... (python3) ... anon-rss:1525148kB` (and two similar) from the earlier memory incident, before the lazy loading and cache limits existed: the process had grown to the container's limit and was killed and restarted by Docker. Since the fix none has been recorded in the part of the log that was read.

### 3.4 Network and ports

| host port | container port | process | what is served |
|---|---|---|---|
| 80 | 8766 | `python3 server_128.py` | the pages `/` and `/demo`, scripts, `/models/*`, `/edge_tc/*`, `/api/*` |
| 8080 | 8767 | `python3 fig47/static_server.py` | the static files of `fig47/` (the Figure 4 / Figure 7 viewer, figures, JSON, scripts) |

The security group opens exactly these two ports to the whole internet. The access log of the container shows what an open port on the public internet attracts: automated scanners requesting paths such as `/boaform/admin/login.asp`, `/mcp`, `/sse`, `/zc?action=getInfo`, `/favicon.ico`, and requests for source files such as `/graph_engine.py` on port 8080. The app answers 404 to unknown paths on port 80. The static server on port 8080 serves *everything inside `fig47/`*, including its Python scripts and JSON files, and shows a directory listing of the folder; these are the same files that are in the repository, nothing secret is in that folder, but it is worth knowing it is browsable.

TCP congestion control can be switched to BBR with the `tune-network` workflow (section 4 of `10-github-actions.md`); the settings are written to `/etc/sysctl.d/99-bbr.conf` and survive reboots.

### 3.5 Metadata service and access

The instance requires IMDSv2 (`HttpTokens=required`, hop limit 2 so that containers can still reach it). No SSH. All management is by SSM Run Command, which needs the instance profile to include the managed policy `AmazonSSMManagedInstanceCore`; the launch workflow prints a warning when it does not.

### 3.6 Uptime and restarts

The instance was running for 19 days and 3 hours at the time of the check (no stop since launch). The container had been up 6 hours (it is restarted by each deployment). A *soft reboot* (`reboot-instance` workflow) keeps the disk, the public IP and the sysctl settings, and Docker restarts the container by itself because of `--restart always`. A *stop and start* would give the instance a new public IP and DNS name; the workflows that depend on the address (CloudFront, HTTPS names) would then have to be run again.

## 4. How the container is started

`deploy-ec2.yml` writes a launch script (`userdata`) that runs once at first boot:

```sh
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
dnf install -y docker
systemctl enable --now docker
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <registry>
docker pull <registry>/clrs-circuit-tracer:latest
docker run -d --restart always --name tracer --log-opt max-size=10m --log-opt max-file=3 -p 80:8766 -p 8080:8767 <registry>/clrs-circuit-tracer:latest
```

`update-container.yml` (every later deployment) runs, on the instance and through SSM:

```sh
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <registry>
docker pull <registry>/clrs-circuit-tracer:latest
docker rm -f tracer
docker run -d --restart always --memory 1500m --memory-swap 1800m --name tracer \
    --log-opt max-size=10m --log-opt max-file=3 -p 80:8766 -p 8080:8767 <registry>/clrs-circuit-tracer:latest
docker image prune -f
docker ps
```

Differences between the two: the update adds the memory limits (`--memory 1500m --memory-swap 1800m`), which the first launch did not have, and removes dangling images from the instance's disk afterwards. Log files of the container are capped at 3 files of 10 MB.

The image's own start command is `./start.sh`:

```sh
python3 fig47/static_server.py &      # port 8767
exec python3 server_128.py            # port 8766 (the container sets SERVER_HOST=0.0.0.0)
```

## 5. The image and the ECR repository

### 5.1 What is inside the image

Built from the `Dockerfile` in the repository root:

1. Base `python:3.11-slim-bookworm`.
2. `apt-get install git` (needed to install the CLRS library from GitHub).
3. `pip install -r requirements.txt` (`jax==0.4.34`, `jaxlib==0.4.34`, `dm-haiku==0.0.12`, `numpy==2.4.6`, and `git+https://github.com/google-deepmind/clrs.git`).
4. `COPY . .` (everything not excluded by `.dockerignore`: so `docs/`, `verification/`, `.git`, `.github` and the readme files stay out).
5. `ENV CKPT_DIR=/app/checkpoints_v3` and `ENV SERVER_HOST=0.0.0.0`.
6. `RUN python model_export.py models` which writes `models/<algo>.json` and `.bin` for all twelve algorithms from the checkpoints (the files are not in git; they exist only in the image).
7. `EXPOSE 8766 8767`, `CMD ["./start.sh"]`.

The edge-site transcoder files (`edge_tc/`) and the node-site files (`algos/`) are plain files copied in step 4. The image therefore contains, besides Python and JAX: the twelve checkpoints (65 MB), the generated model bundles (about 18 MB), `edge_tc/` (about 19 MB) and `algos/` (about 5 MB).

### 5.2 The ECR repository

Repository `clrs-circuit-tracer` in `us-east-1`. When `build-and-push-ecr.yml` creates it (the "create repository if missing" option) it enables scan-on-push and tags it `Name` and `Project`. The workflow always pushes the tag `latest`.

What the read-only listing of 24 November 2026 showed:

| fact | value |
|---|---|
| image records in the repository | 95 |
| records that are real images (compressed size 0.97 to 1.08 GB each) | 63 |
| tiny records (about 9 KB: manifest lists / attestations pushed by the builder) | 32 |
| oldest real image | 2026-09-25 02:39 UTC (975 MB) |
| newest real image at the time | 2026-11-24 07:31 UTC (1,077 MB) |
| real images pushed in September / November | 36 / 27 |
| images carrying a tag in the listing | the listing printed `None` for every record in the column read; the newest push always holds `latest` and the others are untagged |

"Compressed size" is the size of the image's layers as stored in ECR. Images share layers (the Python base and the pip layer are identical between builds unless the requirements change), so the repository's real storage is the sum of the *distinct* layers, not 63 x 1 GB (the API used by the diagnose workflow does not report the distinct total). Even so, ECR bills by GB-month for stored layers, and a repository that gains one image per deployment and never removes any grows without limit.

**Cost-saving option (not applied):** an ECR *lifecycle policy* that expires untagged images after a few days (and keeps the newest few). It costs nothing to configure and caps the storage; it deletes images, so it was not applied without asking. A rollback to an older image is only possible while it still exists, so a policy that keeps, say, the five most recent images is the safe form.

### 5.3 Rolling back

There is no workflow for it, but the pieces exist: every push leaves the previous image in the repository (untagged). To roll back one would retag an older image's digest as `latest` (`aws ecr batch-get-image` to read its manifest, `aws ecr put-image` to write it under the tag `latest`) and then run the `update-container` workflow. The section "Rollback" of `12-operations-runbook.md` spells the steps out. A rollback of the *page* alone is easier: revert the commit on `main`, build, and update.

## 6. The cost of the setup

Prices below are approximate on-demand list prices for `us-east-1` and should be checked against the AWS price pages; they explain the order of magnitude, not an invoice. The project's own target is a monthly bill under about 30 dollars (it is currently in the high teens).

| item | basis | approx. per month |
|---|---|---|
| EC2 t3.small, Linux, on demand | about 0.0208 USD per hour x 730 h | 15.2 USD |
| EBS gp3, 16 GB | about 0.08 USD per GB-month | 1.3 USD |
| public IPv4 address | about 0.005 USD per hour (charged since 2024) | 3.7 USD |
| ECR storage | about 0.10 USD per GB-month of distinct layers | roughly 1 to 3 USD, grows with every unpruned push |
| data transfer out | a first allowance each month is free; pages and models are small and cached | well under 1 USD at current traffic |
| CloudWatch (basic metrics) | free | 0 |
| GitHub Actions | manual runs only (a build is about 5 minutes); within the free minutes of a normal plan | 0 |
| **Total** | | **about 18 to 22 USD** |

How cost would change:

- CloudFront in front of the site (workflow ready, not applied): a small usage-based charge, a few dollars at most at demo traffic; it would also give an `https://` address and nearer caches for far-away visitors.
- Resizing the instance up or down: the project rule is "no resize" (a t3.small is the smallest that holds the Python process comfortably).
- An Elastic IP attached to the running instance costs the same as the automatic public address; a fixed IP would avoid the address change on stop/start.
- A lifecycle policy on ECR (section 5.2) would cap storage cost.

## 7. Security notes

- **Open ports** are 80 and 8080 only, to the whole internet, plain HTTP. There is no login. The service only serves public files and computes sampled graphs; there is no user data.
- **Plain HTTP** means traffic is readable on the way. Chrome 157 and later treat plain HTTP to a public address as something to warn about; see section 9.
- **Credentials.** The workflows use an IAM user's access key stored as GitHub secrets (or an OIDC role if `AWS_ROLE_ARN` is set). Keys that have ever been pasted into a chat or a file should be rotated; the tokens used for pushing to GitHub likewise. This is on the project's open list.
- **Guards in every workflow that changes something:** the manual trigger, the typed `confirmed`, the tag check (exactly one running instance with both tags), and for several the expected instance ID. The workflows never touch other instances, groups or key pairs.
- **No SSH.** A lost key cannot expose a shell; SSM access is controlled by IAM.
- **Scanners** hit the site constantly (see section 3.4). Nothing they request exists, apart from the public files.
- **Static server on 8080** is Python's plain file server: it serves every file under `fig47/` and, because no index page exists, it also shows a **directory listing** of that folder at `http://98.86.163.27:8080/` (checked on 24 November 2026). Everything listed is also public in the repository.

## 8. Health checks and how to look

| question | how |
|---|---|
| is the site up? | open `http://98.86.163.27/`, or `curl http://98.86.163.27/api/algos` (answers 200 with a JSON list) |
| which build is live? | the page source contains `const BUILD = '<id>'`; `curl -s http://98.86.163.27/ | grep -o "BUILD = '[0-9a-f]*'"` (add `-H 'Accept-Encoding: gzip'` and decompress if you want the compressed form) |
| are the models and transcoders served? | `GET /models/bfs.json?v=<build>` and `GET /edge_tc/bfs.bin?v=<build>` |
| instance and container state, memory, logs, CPU credits | run the `ec2-diagnose` workflow (read-only) |
| do all twelve algorithms load in a browser? | run the `pre-demo-check` workflow with the address (see the note on HTTPS-first Chrome in `12-operations-runbook.md`) |
| do the main browsers work? | run the `cross-browser-check` workflow |

What the diagnose workflow printed on 24 November 2026 (abridged): instance `i-0877713b5583c8533`, launched 2026-11-04T19:23:23Z, credit mode `standard`; CPU credit balance 576.0 in every five-minute sample of the last hour; CPU 0.43 to 0.48%; the container `tracer` up 6 hours with ports `0.0.0.0:80->8766/tcp` and `0.0.0.0:8080->8767/tcp`; log lines `serving fig47/ static files on http://0.0.0.0:8767/viewer_fig47.html`, `ready: models load lazily on first request per algorithm`, `serving on http://0.0.0.0:8766/viewer3.html`, `loaded bfs`, `warmed bfs`, `loaded dfs`, `warmed dfs`, `warm-up done`; memory 1909 MiB total, 837 MiB used, 913 MiB available; swap 2047 MiB with 31 MiB used; load average 0.00.

## 9. Things prepared but not applied

### 9.1 HTTPS on the instance itself (`enable-https.yml`)

A small Caddy container (`caddy:2.8.4-alpine`, limited to 200 MB) would run on the host network next to the tracer, listen on port 443 and obtain a free certificate for `98-86-163-27.sslip.io` (a public wildcard DNS service that resolves a name containing an IP address to that address). The certificate would be requested with the TLS-ALPN challenge on port 443 only, so nothing needs port 80. The workflow opens TCP 443 on the project-tagged security group and writes the Caddyfile through SSM; a `check` mode shows everything read-only, `disable` removes only the Caddy container. Results of the read-only check: the security group is the tagged one, only 80 and 8080 are open, the key may open 443, the name resolves to the instance. Caveats: it depends on the third-party name service, the name changes if the instance's IP changes, and it gives no nearer caches to far-away visitors.

*Why it matters:* current Chrome versions block or warn on plain `http://` addresses; investors opening the link in Chrome may see a warning page instead of the site. Safari and Firefox open the page normally.

### 9.2 CloudFront (`setup-cloudfront.yml`)

Creates (only if missing, tagged with the project) one cache policy and one distribution on the default `*.cloudfront.net` name. The origin is the instance's *public DNS name* over HTTP on port 80. Six paths are cached for as long as the origin's headers allow (`/models/*`, `/api/weights`, `/api/feature_labels`, `/api/instance`, `/api/algos`, `*.js`), everything else is not stored; viewers are redirected to HTTPS; HTTP/2 and HTTP/3 are enabled. No Elastic IP and no alarm are created. Caveat noted in the workflow: the origin name changes if the instance is stopped and started. `disable` switches the distribution off without deleting it.

### 9.3 ECR lifecycle policy

See section 5.2.

## 10. Quick reference

| thing | value |
|---|---|
| address of the site | http://98.86.163.27/ |
| previous version | http://98.86.163.27/demo |
| figures page | http://98.86.163.27:8080/viewer_fig47.html |
| instance id | `i-0877713b5583c8533` |
| security group | `sg-0d5fcca95d559bf57` (`clrs-circuit-tracer-app-sg`) |
| ECR repository | `clrs-circuit-tracer`, tag `latest` |
| container name | `tracer` |
| region / zone | `us-east-1` / `us-east-1a` |
| instance type, disk | t3.small, 16 GB gp3 |
| deploy | `Build and push image to ECR`, then `Update running container` with `confirm=confirmed` |
