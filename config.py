"""Single source of truth for the trained-model checkpoint location used by
every script in this project (pipeline.py, server_128.py, fig47/*.py).

ONLY checkpoints_v3 is ever to be used here. Do not point this at any other
checkpoint directory on disk -- in particular, /Users/anindya/CLRS/checkpoints/mpnn/
is a DIFFERENT, unrelated 30-algorithm checkpoint set (verified: same param
shapes, different trained weight values) and must never be referenced.
"""
import os

# CKPT_DIR env var lets the container override this to /app/checkpoints_v3
# without touching local dev, which still uses the absolute Mac path below.
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/Users/anindya/CLRS/new run/final results and notebook(v3 versions)/checkpoints_v3/mpnn_12graph_algo_checkpoints",
)

ALGOS = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim',
        'mst_kruskal', 'topological_sort', 'articulation_points', 'bridges',
        'strongly_connected_components', 'floyd_warshall']

# Fail loudly at import time if the expected checkpoints aren't where they
# should be, rather than silently falling back to some other path.
assert os.path.isdir(CKPT_DIR), f"checkpoints_v3 not found at: {CKPT_DIR}"
for _algo in ALGOS:
    _p = os.path.join(CKPT_DIR, _algo, 'best.pkl')
    assert os.path.isfile(_p), f"missing checkpoints_v3 checkpoint for '{_algo}': {_p}"
