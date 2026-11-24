# Documentation index

Everything about the CLRS-30 Circuit Tracer site, in order of how often it is needed. Files marked *generated* are written by `tools/clrs_docs_generate.py` from the data files and the page code; do not edit them by hand, regenerate them (see `12-operations-runbook.md`, section 6).

| file | contents |
|---|---|
| [01-overview.md](01-overview.md) | what the project is, what it can and cannot claim, key numbers |
| [02-glossary.md](02-glossary.md) | every keyword used on the site and in the code, alphabetical by topic |
| [03-repository-file-map.md](03-repository-file-map.md) | *generated*: every tracked file, what it is, whether it goes into the Docker image |
| [04-the-12-algorithms.md](04-the-12-algorithms.md) | *generated*: one section per algorithm: task, inputs, hints, training, scores, pseudo-code, latent meanings |
| [05-model-and-math.md](05-model-and-math.md) | the MPNN, the transcoders, the attribution graph, the checks, the formulas |
| [06-page-reference.md](06-page-reference.md) | the page, part by part, with a worked example |
| [06b-hover-text-catalog.md](06b-hover-text-catalog.md) | *generated*: every hover text on the page |
| [07-data-files-and-formats.md](07-data-files-and-formats.md) | binary and JSON formats of the model and transcoder files |
| [08-server-and-api.md](08-server-and-api.md) | the server, its routes, caching, build id, limits |
| [09-aws-infrastructure.md](09-aws-infrastructure.md) | EC2 instance, security group, SSM, ECR, container, cost |
| [10-github-actions.md](10-github-actions.md) | every workflow: purpose, inputs, steps, guards |
| [11-testing-and-verification.md](11-testing-and-verification.md) | every test, what it proves, how to run it |
| [12-operations-runbook.md](12-operations-runbook.md) | deploy, verify, roll back, demo checklist, troubleshooting, maintenance |
| [13-project-history-and-decisions.md](13-project-history-and-decisions.md) | timeline of commits and the reasons behind decisions |
| [14-faq.md](14-faq.md) | short answers to common questions |
| [15-latent-dictionaries.md](15-latent-dictionaries.md) | *generated*: validated latent meanings per algorithm |
| [16-transcoder-training-reports.md](16-transcoder-training-reports.md) | *generated*: training reports of the edge-site transcoders |
| [17-fig47-study-results.md](17-fig47-study-results.md) | *generated*: results of the Figure 4 / Figure 7 study |

## Reading paths

- *Handing the project to someone*: 01, 03, 09, 10, 12, 13.
- *Explaining the science*: 01, 05, 02, 04, 15.
- *Checking a number on the page*: 06, 06b, then the relevant generated file.
- *Preparing a demo*: 12 (section 4), 14.

## Conventions

- Dates are 2026; the build that was live when these documents were written is the one named in the page source (`BUILD = '...'`).
- No credentials, account numbers or personal addresses appear in these files. Commands use placeholders.
- A statement that could not be verified was left out rather than guessed.
