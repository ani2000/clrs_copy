#!/bin/bash
# 60 fresh n = 16 graphs per algorithm from the tracer's own sampler (seeds 2000..2059), one request at a time, into <folder>.
#   BASE=http://98.86.163.27 BUILD=<current build id> ./fetch_graphs.sh <folder>
# (the sampler may answer a request with a graph of another seed when it rejects one and retries; the file keeps the REQUESTED seed, which is what the page loads)
dir=${1:?folder}; mkdir -p "$dir"; cd "$dir"
for a in bfs dijkstra bellman_ford dag_shortest_paths mst_prim dfs topological_sort articulation_points bridges strongly_connected_components mst_kruskal floyd_warshall; do
  for s in $(seq 2000 2059); do
    f=${a}_$s.json; [ -s $f ] && continue
    curl -s -m 60 -H 'Accept-Encoding: gzip' "$BASE/api/instance?algo=$a&n=16&p=0.4&seed=$s&v=$BUILD" | gzip -dc > $f.tmp 2>/dev/null && [ -s $f.tmp ] && mv $f.tmp $f || rm -f $f.tmp
    sleep 0.2
  done
done
