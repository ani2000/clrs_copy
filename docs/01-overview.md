# Overview

## 1. What this project is

The CLRS-30 Circuit Tracer is a web site and the code, data and infrastructure behind it. It lets a visitor look inside small neural networks that were trained to execute twelve classic graph algorithms (from the CLRS-30 benchmark: breadth-first search, depth-first search, Bellman-Ford, Dijkstra, DAG shortest paths, Prim, Kruskal, topological sort, articulation points, bridges, strongly connected components, Floyd-Warshall).

For one chosen algorithm and one small random graph the page shows three things (the simulation and the pseudo-code side by side, with the attribution graph below them at full width):

1. a **simulation** of the algorithm, step by step (what the algorithm's own state is);
2. an **attribution graph** for the prediction the network makes at a chosen node and step: which internal features, carried on which edges, produced the answer, and how much each contributed;
3. a **pseudo-code pane** that highlights the line (and the nested loop) of the algorithm that corresponds to the current step.

Below these there are panels with the evidence: the training scores of the model, the features most active at each step, a dictionary of features with validated meanings, a message map, interventions (what happens if a feature is changed), and the checks the graph satisfies.

## 2. What is in the world around it

| piece | where | what it is |
|---|---|---|
| the main page | `http://98.86.163.27/` | attribution graphs from the edge-site transcoder (1,280 features per edge) |
| the previous page | `http://98.86.163.27/demo` | attribution graphs from the node-site transcoder (192 features), kept as it was |
| the figures page | `http://98.86.163.27:8080/viewer_fig47.html` | Figure 4 / Figure 7 study viewer |
| the repository | `https://github.com/ani2000/clrs_copy`, branch `main` | code, checkpoints (Git LFS), exported data, tests, workflows, these documents |
| the machine | one EC2 t3.small in us-east-1 | runs one Docker container with the server |
| the image store | ECR repository `clrs-circuit-tracer` | the images the container is started from |
| the workflows | GitHub Actions, all manual | build, deploy, update, diagnose, reboot, network tuning, checks, HTTPS/CloudFront (prepared) |

## 3. The idea in five sentences

A message-passing neural network (MPNN) computes, at each step, a message `y_ij` for every pair of nodes from an input `s_ij`. A *transcoder* replaces this computation by a wide, sparse one: 1,280 candidate features, of which the 8 strongest are kept for each edge. Sparse features can be inspected one at a time, and whatever the transcoder does not reproduce is kept as explicit *error nodes*, so nothing is hidden. Following the contributions from the output back through features, edges and earlier steps gives an *attribution graph*, which the page computes in the browser for the clicked prediction. A feature gets a human-readable *meaning* only when a simple rule, fitted on some graphs, predicts it on other graphs.

## 4. What the site can honestly claim

- The attribution graph is **computed**, for each click, from the real checkpoint and the real transcoder. Tests compare it with an independent recomputation (`11-testing-and-verification.md`).
- The graph is **complete up to its stated error**: the contributions, error nodes and bias add up (the sum test), and the error share is displayed.
- Feature **meanings** that are shown passed a held-out test. About 3 to 18 of the roughly 650 to 1,190 features that fire per algorithm have such a meaning.
- The models' **scores** are the real validation and test scores of the training run. Seven algorithms generalise from 16 to 64 nodes (CLEAN); five do not (SEVERE).

## 5. What the site cannot claim

- That it explains how the network "understands" an algorithm in general. It explains single predictions on small graphs.
- That unnamed features are meaningless; they are unexplained.
- That interventions are fully predicted: the graph's clamp prediction pointed in the right direction for about 69 percent of tests; the measured outcome of an intervention is the reliable number.
- That the five SEVERE algorithms are understood at 64 nodes; their models are wrong there.
- That the page is private or secure: it is served over plain HTTP and contains no user data.

## 6. Numbers worth knowing

| item | value |
|---|---|
| algorithms | 12 |
| hidden size of the model | 128 |
| edge-site features / active per edge | 1,280 / 8 |
| node-site features | 192 (from 384 inputs) |
| model bundle in the browser | about 1.2 to 2.1 MB per algorithm |
| edge transcoder in the browser | about 1.3 MB per algorithm |
| default graph | n = 16, p = 0.4, seed 42, bfs |
| latency targets | at most 2 s, normally below 500 ms |
| infrastructure | one t3.small, 16 GB gp3, 2 GiB swap, container limit 1500 MB (swap total 1800 MB) |
| monthly budget | at most 30 dollars (10 more if needed) |
| tests | numbered 1 to 9, truth test, layout test, pseudo tests, UI tests |

## 7. How to read these documents

Start with the glossary if a word is unfamiliar, with the page reference if you want to know what a thing on the page means, and with the runbook if you have to do something. The index is `README.md` in this folder.
