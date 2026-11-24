# The server and its API

The application server is one Python file, `server_128.py`, written with the standard library (`http.server`) only. There is no web framework. It serves the pages, the scripts, the model files and a small JSON API. A second, even simpler server (`fig47/static_server.py`) serves the static Figure 4 / Figure 7 files.

## 1. The two servers

| server | file | port in the container | port on the host | what it serves |
|---|---|---|---|---|
| application | `server_128.py` | 8766 | 80 | `/`, `/demo`, scripts, `/models/*`, `/edge_tc/*`, `/api/*` |
| static files | `fig47/static_server.py` | 8767 | 8080 | every file of `fig47/` as it is (Python's `SimpleHTTPRequestHandler`), including a directory listing |

`start.sh` starts the static server in the background and then runs `server_128.py` in the foreground:

```sh
#!/bin/sh
set -e
python3 fig47/static_server.py &
exec python3 server_128.py
```

## 2. Configuration (environment variables)

| variable | default | meaning |
|---|---|---|
| `PORT` | `8766` | port of the application server |
| `SERVER_HOST` | `127.0.0.1` (the image sets `0.0.0.0`) | address to listen on |
| `CKPT_DIR` | the author's Mac path (the image sets `/app/checkpoints_v3`) | where the checkpoints are; `config.py` asserts that all twelve `best.pkl` files exist |
| `MODEL_CAP` | `3` | the most JAX models kept in memory at once (least recently used is evicted) |
| `VARIANT_CAP` | `3` | the most compiled program variants JAX may keep before the caches are cleared |
| `WARM` | `1` | set to `0` to skip the start-up warm-up |
| `WARM_ALGOS` | `bfs,dfs` | algorithms loaded and compiled at start-up |
| `WARM_MIN_MB` | `450` | the warm-up stops early if less than this much memory is available |

## 3. What happens at start-up

1. `config.py` is imported: it checks the checkpoint folder and the twelve checkpoint files and fails loudly otherwise.
2. `META` is read from `algos/<algo>/meta.json` for all algorithms.
3. The build id is computed (section 5).
4. The text `ready: models load lazily on first request per algorithm (checkpoints_v3).` is printed. No model is loaded yet.
5. The HTTP server starts on `PORT` (`QuietServer`: daemon threads, a request queue of 64, broken-pipe and connection-reset errors ignored because they mean the visitor left).
6. A background thread warms the models in `WARM_ALGOS`: for each it takes the two-run semaphore and runs one graph (`run_graph(a, 16, 0.4, 1)`), which loads the model and compiles the JAX program, so the first real request is fast; it stops early when available memory falls below `WARM_MIN_MB`.

The container log of the live instance shows exactly this sequence (`loaded bfs`, `warmed bfs`, `loaded dfs`, `warmed dfs`, `warm-up done`).

## 4. Endpoints

All paths are relative to `http://98.86.163.27`. "Versioned" means the request carries `?v=<build id>`.

### 4.1 Pages and files

| path | answer |
|---|---|
| `/` and `/viewer3.html` | the main page with `__BUILD__` and `__ALGOS_JSON__` replaced; `Cache-Control: no-cache` plus an `ETag` (the browser revalidates; a 304 saves the transfer) |
| `/demo` | `demo.html`, the previous version, same treatment |
| `/demo/` | 301 redirect to `/demo` (relative script paths need no trailing slash) |
| `/explain.js`, `/attrview.js`, `/attrib.js`, `/clrs-edge-attrib.js`, `/mpnn.js`, `/mpnn_worker.js` | the scripts; versioned requests get `Cache-Control: public, max-age=31536000, immutable` |
| `/models/<algo>.json`, `/models/<algo>.bin` | the model bundle (generated at image build time); immutable |
| `/edge_tc/<algo>.json`, `/edge_tc/<algo>.bin`, `/edge_tc/<algo>.latents.json`, `/edge_tc/clrs-edge-summary.json` | the edge-site transcoder, its manifest, its latent table, the quality summary; immutable |

Unknown paths and names that do not match the patterns answer 404. The patterns are strict regular expressions (`^models/([a-z_]+)\.(json|bin)$` and the corresponding one for `edge_tc/`), the algorithm name must be one of the twelve, and nothing outside these patterns is read from disk, so `../` tricks do not work.

### 4.2 The JSON API

| endpoint | parameters | answer | notes |
|---|---|---|---|
| `GET /api/algos` | `v` | `{algo: {target, target_type, fvu, val_f1, test_f1}}` for the twelve | also embedded in the page so the first paint needs no request |
| `GET /api/instance` | `algo`, `n` (4 to 32), `p` (0.05 to 0.95), `seed`, `v` | the compact instance of a sampled graph (section 4.3) | cheap: the CLRS sampler only, no model run; 400 for bad parameters; cacheable when versioned because it is deterministic |
| `GET /api/weights` | `algo`, `v` | the node-site transcoder weights as JSON (about 1.3 MB, 0.5 MB gzipped) | used by `/demo` only; built once per algorithm and kept in memory; ETag |
| `GET /api/feature_labels` | `algo` | the labels of the 192 node-site features | used by `/demo` only; file content with ETag |
| `GET /api/contexts` | `algo`, `feature`, `n`, `p` | the three strongest real activations of a node-site feature found in 200 fresh episodes | used by `/demo`; heavy: one at a time (`CTX_SEM`), results cached (up to 64), abandoned requests are cancelled; exempt from the stale check |
| `GET /api/graph` | `algo`, `n`, `p`, `seed` | runs the real model on the server for one graph and returns node features, messages and truths | a fallback when the browser cannot run the model; at most two at once (`GRAPH_SEM`); exempt from the stale check; degenerate graphs (length below 3) are resampled with a shifted seed (+1000, +1001, ...) up to 8 times |

### 4.3 The instance (`/api/instance`)

```json
{
  "n": 16, "length": 5, "seed": 101,
  "adj": [[...16 numbers...] ... 16 rows],
  "source": 5,
  "target_true": [[...16 numbers...] ... 5 rows],
  "inputs": {"pos": [...], "s": [...], "A": [[...]], "adj": [[...]]},
  "hints0": {"reach_h": [...], "pi_h": [...]}
}
```

| field | meaning |
|---|---|
| `n` | number of nodes |
| `length` | number of hint rows (`length - 1` message-passing steps) |
| `seed` | the seed actually used (differs from the requested one when the sampler had to retry) |
| `adj` | the adjacency matrix |
| `source` | the source node, or `null` for algorithms without one |
| `target_true` | the true hint of the target for every row (one number per node; for categorical targets the class index) |
| `inputs` | every input field of CLRS for this graph (names depend on the algorithm) |
| `hints0` | the first row of every hint field (the starting state the model is given) |

The browser runs the model from this instance alone.

### 4.4 Errors and the stale answer

A request that carries a build id different from the server's (for example from a page that was open during a deployment) gets

```http
HTTP/1.1 409 Conflict
Content-Type: application/json
Cache-Control: no-store

{"error": "stale", "build": "<the server's build id>"}
```

The page turns this into the message "This page was just updated. Reload to continue." with a Reload button (`showStale()` in `viewer3.html`). `/api/graph` and `/api/contexts` are exempt. Unexpected exceptions inside a handler are returned as a JSON object `{"error": ..., "trace": ...}` with the Python traceback (this is a development convenience; it also means an internal error reveals a stack trace to the caller). Connection errors while writing (the visitor closed the tab) just close the connection.

## 5. The build id

```python
def _build_id():
    h = hashlib.sha1()
    for f in sorted(STATIC_FILES):                # the page(s) and the scripts
        h.update(f.encode()); h.update(open(f, 'rb').read())
    for a in sorted(ALGOS):                       # the model bundles
        for ext in ('json', 'bin'):
            h.update(open(f'models/{a}.{ext}', 'rb').read())
    for f in sorted(os.listdir('edge_tc')):       # the edge-site transcoder files
        h.update(f.encode()); h.update(open(f'edge_tc/{f}', 'rb').read())
    return h.hexdigest()[:10]
```

(`dev` is used if a file cannot be read.) So the id changes whenever the page, any script, any model bundle or any `edge_tc/` file changes, and only then. Every URL the page builds contains it. Consequences:

- the browser may cache every script, model and transcoder for a year without ever showing an old version (immutable caching);
- a new deployment gives a new id, so nothing stale is reused;
- an old page is told to reload instead of mixing old and new files.

The documentation folder is not part of the id (it is also excluded from the image).

## 6. Compression and caching in detail

| kind of answer | compressed? | cache header | validator |
|---|---|---|---|
| pages (`viewer3.html`, `demo.html`) | gzip level 9, computed once per file content and kept in memory (`GZ_STATIC`) | `no-cache` | `ETag` = first 20 hex characters of the SHA-1 of the served bytes, with `-gz` appended for the compressed form; `Vary: Accept-Encoding` |
| scripts | gzip level 9 (cached) | immutable when versioned, `no-cache` otherwise | no ETag |
| `<algo>.latents.json` | gzip level 9 (cached) | immutable | ETag |
| other `edge_tc` files, model bundles | not compressed (float data compresses by only about 8%) | immutable | ETag |
| JSON API answers | gzip level 3 on the fly (or a precomputed gzip for `/api/weights`) | immutable only when versioned (`/api/instance`, `/api/weights`, `/api/feature_labels`) | ETag for weights and labels |

Examples measured on the live site: the page 95 KB uncompressed, 36 KB over the wire; `bfs.latents.json` 201 KB, 49 KB over the wire.

Writes are done in 64 KB chunks with a 120-second timeout *per chunk*, so a slow connection can finish a big download without the 30-second request timeout aborting it. HTTP/1.1 keep-alive is on (`protocol_version = 'HTTP/1.1'`).

## 7. Memory and concurrency controls

The model runs on the server only for `/api/graph` and `/api/contexts`. The page runs the model itself in the browser, so under normal use the server mostly hands out static files. The controls exist because the host has under 2 GiB of RAM.

- **`LazyRegistry(MODEL_CAP)`**: loads a model on first use, keeps at most `MODEL_CAP` models, evicts the least recently used, serialises loads (two at once would double the peak) and calls `release_memory()` after an eviction. A request that holds a model keeps it alive until it finishes.
- **`release_memory()`**: `gc.collect()` and `malloc_trim(0)` through `libc`, to give freed heap pages back to the operating system.
- **`trim_compiled()`**: every new graph length makes JAX compile (and keep) another program, about 50 MB for the long algorithms; when more than `VARIANT_CAP` are held, `jax.clear_caches()` drops them (they are rebuilt on demand).
- **`GRAPH_SEM = Semaphore(2)`** and **`CTX_SEM = Semaphore(1)`**: at most two model runs and one context search at once.
- **`CTX_CACHE`**: results of the context search, cleared when it holds more than 64 entries.
- **Start-up warm-up** stops when memory is low (section 3).
- **Container limit**: `--memory 1500m --memory-swap 1800m` (see `09-aws-infrastructure.md`).

## 8. How the server computes a graph (`/api/graph`)

1. `clrs.build_sampler(algo, num_samples=1, seed, length=n, p=(p,))` samples one graph and its reference trace.
2. The network is built with `CaptureNet` (a wrapper of the CLRS network that also returns the internal activations), the parameters are merged from `best.pkl` and the function is `jax.jit`-compiled.
3. One forward pass returns the output predictions, the hint predictions, the hidden states, `z` and the messages.
4. The result contains `z` and `msgs` for every step (the browser derives `node_fts` from the first 128 columns of `z`), the adjacency, the source, the true hints of the target, the length and the seed.

`/api/instance` (used by the page) does steps 1 and the packaging only; the forward pass is done in the browser by `mpnn.js` and gives the same activations (checked by `verification/verify_js.js` and the parity tests).

## 9. The static server (`fig47/static_server.py`)

```python
PORT = 8767
handler = partial(SimpleHTTPRequestHandler, directory=here)
ThreadingHTTPServer(('0.0.0.0', PORT), handler).serve_forever()
```

It serves the files of `fig47/` unchanged. It has no caching headers, no compression and no path whitelist, and it prints an access line to the container's log for every request (which is why the log shows scanner traffic from port 8080). Requests to the application server on port 80 are not logged (its `log_message` does nothing).

## 10. What the server does not do

- no authentication, sessions, cookies or user data;
- no HTTPS (see `09-aws-infrastructure.md` section 9);
- no rate limiting beyond the two semaphores;
- no writing to disk: the container needs no volume and loses nothing when restarted.
