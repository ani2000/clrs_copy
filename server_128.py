"""Local API for the interactive circuit tracer on the REAL 128-hidden
checkpoints_v3, all 12 graph algorithms. One CaptureNet per algorithm
(same class as trace36/circuit_tracer, unmodified -- it's spec-generic),
loaded once at startup. Attribution/intervention math still runs
client-side in JS; this server only runs the real jax forward pass.
"""
import json, pickle, os, threading, gzip, select, socket, re, hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np, jax, haiku as hk, clrs
from capture_net import CaptureNet, get_capture_processor_factory
from config import CKPT_DIR, ALGOS  # single source of truth -- checkpoints_v3 only
from model_export import build_instance

HIDDEN = 128

META = {a: json.load(open(f'algos/{a}/meta.json')) for a in ALGOS}


def _f32(a):
    # shortest exact float32 repr: lossless for float32 data, ~40% smaller JSON
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return [float(str(v)) for v in a]
    return [[float(str(v)) for v in row] for row in a]


def _load(algo):
    spec = clrs.SPECS[algo]
    meta = META[algo]
    t = dict(np.load(f'algos/{algo}/transcoder.npz'))
    sampler0, _ = clrs.build_sampler(algo, num_samples=1, seed=0, length=8, p=(0.4,))
    fb0 = sampler0.next()
    nb_dims = {}
    for inp in fb0.features.inputs: nb_dims[inp.name] = inp.data.shape[-1]
    for hint in fb0.features.hints: nb_dims[hint.name] = hint.data.shape[-1]
    for outp in fb0.outputs: nb_dims[outp.name] = outp.data.shape[-1]

    def make_net_fn(spec=spec, nb_dims=nb_dims):
        def net_fn(features_list, repred, algorithm_index, return_hints, return_all_outputs):
            return CaptureNet([spec], HIDDEN, True, True, get_capture_processor_factory(True, 8),
                              False, 'xavier_on_scalars', 0.0, 0.0, 'soft', [nb_dims], 1, False)(
                features_list, repred, algorithm_index, return_hints, return_all_outputs)
        return hk.transform(net_fn)

    net = make_net_fn()
    params = net.init(jax.random.PRNGKey(0), [fb0.features], True, 0, True, False)
    ckpt = pickle.load(open(f'{CKPT_DIR}/{algo}/best.pkl', 'rb'))['params']
    assert set(params) == set(ckpt), f'{algo}: checkpoint mismatch'
    params = hk.data_structures.merge(params, ckpt)

    weights_json = dict(
        w_enc=_f32(t['w_enc']), b_enc=_f32(t['b_enc']), w_dec=_f32(t['w_dec']), b_dec=_f32(t['b_dec']),
        mu_x=_f32(t['mu_x']), sd_x=_f32(t['sd_x']), F=int(t['F']), K=int(t['K']), hidden=HIDDEN,
        W_dec=meta['W_dec'], b_dec_target=meta['b_dec'],
        target=meta['target'], target_stage=meta['target_stage'], target_type=meta['target_type'],
        source_field=meta['source_field'], fvu=meta['fvu'], val_f1=meta['val_f1'], test_f1=meta['test_f1'],
    )
    weights_bytes = json.dumps(weights_json).encode()
    weights_gz = gzip.compress(weights_bytes, compresslevel=6)
    print(f'  loaded {algo}: target={meta["target"]} ({meta["target_type"]}), FVU={meta["fvu"]:.4f}', flush=True)
    apply_jit = jax.jit(lambda params, rng, feats: net.apply(params, rng, [feats], True, 0, True, False))
    return dict(apply=apply_jit, net=net, params=params, spec=spec, nb_dims=nb_dims, meta=meta, weights_bytes=weights_bytes, weights_gz=weights_gz)


class LazyRegistry:
    """Loads each algorithm's model on first use (keeps peak RAM low on small hosts)."""
    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def __getitem__(self, algo):
        if algo not in self._d:
            with self._lock:
                if algo not in self._d:
                    self._d[algo] = _load(algo)
        return self._d[algo]


REG = LazyRegistry()
GRAPH_SEM = threading.Semaphore(2)      # bounded concurrent model runs (protects a small host)
CTX_SEM = threading.Semaphore(1)        # the 200-episode context search is heavy: one at a time
CTX_CACHE = {}                          # (algo,feature,n,p) -> result
print('ready: models load lazily on first request per algorithm (checkpoints_v3).', flush=True)


def _run_graph_once(algo, n, p, seed):
    r = REG[algo]
    sampler, _ = clrs.build_sampler(algo, num_samples=1, seed=seed, length=n, p=(p,))
    fb = sampler.next()
    out_preds, hint_preds, hiddens, z, msgs = r['apply'](
        r['params'], jax.random.PRNGKey(seed + 1), fb.features)
    z, msgs, hiddens = np.asarray(z)[:, 0], np.asarray(msgs)[:, 0], np.asarray(hiddens)[:, 0]
    node_fts = z[:, :, :HIDDEN]
    L = int(np.asarray(fb.features.lengths)[0])

    names_i = [i.name for i in fb.features.inputs]
    adj = np.asarray(fb.features.inputs[names_i.index('adj')].data)[0]
    source = None
    sf = r['meta']['source_field']
    if sf:
        s_in = np.asarray(fb.features.inputs[names_i.index(sf)].data)[0]
        source = int(np.argmax(s_in))

    target, stage, ttype = r['meta']['target'], r['meta']['target_stage'], r['meta']['target_type']
    target_true = None
    if stage == 'hint':
        names_h = [h.name for h in fb.features.hints]
        target_true = np.asarray(fb.features.hints[names_h.index(target)].data)[:, 0]  # (T,N[,C])
    if ttype == 'categorical':
        target_true = target_true.argmax(-1).astype(float) if target_true is not None else None

    return dict(
        n=n, length=L, adj=adj.tolist(), source=source,
        target_true=target_true[:L].tolist() if target_true is not None else None,
        z=z[:L - 1].tolist(), msgs=msgs[:L - 1].tolist(),   # node_fts == z[..., :HIDDEN]; the viewer derives it
    )


def run_graph(algo, n, p, seed):
    # resample degenerate (length<3) graphs here instead of costing the client extra round trips
    for tries in range(8):
        out = _run_graph_once(algo, n, p, seed)
        if out['length'] >= 3:
            break
        seed = seed + 1000 + tries
    out['seed'] = seed
    return out



def find_contexts(algo, feature, n, p, n_episodes=200, seed=54321, cancelled=lambda: False):
    r = REG[algo]; t = np.load(f'algos/{algo}/transcoder.npz')
    w_enc, b_enc, mu_x, sd_x = t['w_enc'], t['b_enc'], t['mu_x'], t['sd_x']
    K = int(t['K'])
    sampler, _ = clrs.build_sampler(algo, num_samples=1, seed=seed, length=n, p=(p,))
    names_i_cache = None
    best = []  # (activation, adj, source, target_true_seq, step, node)
    for ep in range(n_episodes):
        if ep % 4 == 0 and cancelled():
            return None                      # client left: stop burning CPU
        fb = sampler.next()
        L = int(np.asarray(fb.features.lengths)[0])
        if L < 2:
            continue
        out_preds, hint_preds, hiddens, z, msgs = r['apply'](
            r['params'], jax.random.PRNGKey(ep + 1), fb.features)
        z, msgs = np.asarray(z)[:, 0], np.asarray(msgs)[:, 0]
        names_i = [i.name for i in fb.features.inputs]
        adj = np.asarray(fb.features.inputs[names_i.index('adj')].data)[0]
        source = None
        if r['meta']['source_field']:
            s_in = np.asarray(fb.features.inputs[names_i.index(r['meta']['source_field'])].data)[0]
            source = int(np.argmax(s_in))
        target_true = None
        if r['meta']['target_stage'] == 'hint':
            names_h = [h.name for h in fb.features.hints]
            target_true = np.asarray(fb.features.hints[names_h.index(r['meta']['target'])].data)[:, 0]
            if r['meta']['target_type'] == 'categorical':
                target_true = target_true.argmax(-1)
        N = adj.shape[0]
        for step in range(min(z.shape[0], L - 1)):
            for node in range(N):
                x = np.concatenate([z[step, node], msgs[step, node]])
                xn = (x - mu_x) / sd_x
                act = np.maximum(xn @ w_enc + b_enc, 0)
                topi = np.argsort(-act)[:K]
                if feature in topi and act[feature] > 0:
                    best.append((float(act[feature]), adj.tolist(), source,
                               target_true[:L].tolist() if target_true is not None else None, step, node))
    best.sort(key=lambda x: -x[0])
    top = best[:3]
    return [dict(activation=a, adj=adj, source=source, target_true=tt, step=step, node=node)
           for a, adj, source, tt, step, node in top]


CACHE_IMMUTABLE = 'public, max-age=31536000, immutable'   # safe: client URLs carry ?v=BUILD (changes with every new image)
_ETAGS = {}


def _etag(data):
    k = id(data)
    if k not in _ETAGS:
        _ETAGS[k] = '"' + hashlib.sha1(data).hexdigest()[:20] + '"'
    return _ETAGS[k]


def _build_id():
    try:
        return hashlib.sha1(b''.join(open(f'models/{a}.json', 'rb').read() for a in sorted(ALGOS))).hexdigest()[:10]
    except OSError:
        return 'dev'


BUILD = _build_id()


STATIC_FILES = {'viewer3.html': 'text/html; charset=utf-8', 'mpnn.js': 'application/javascript',
                'mpnn_worker.js': 'application/javascript', 'explain.js': 'application/javascript', 'attrib.js': 'application/javascript', 'attrview.js': 'application/javascript'}
MODEL_RE = re.compile(r'^models/([a-z_]+)\.(json|bin)$')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'   # keep-alive: no new TCP handshake per request over a long network path
    timeout = 30
    def log_message(self, fmt, *args):
        pass

    def _send(self, body, gz=None, cache=False, etag=None):
        if etag and self.headers.get('If-None-Match') == etag:
            self.send_response(304); self.send_header('ETag', etag)
            self.send_header('Cache-Control', CACHE_IMMUTABLE); self.end_headers(); return
        if 'gzip' in self.headers.get('Accept-Encoding', ''):
            body, enc = (gz if gz is not None else gzip.compress(body, compresslevel=3)), 'gzip'
        else:
            enc = None
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        if cache:
            self.send_header('Cache-Control', CACHE_IMMUTABLE)
        if etag:
            self.send_header('ETag', etag); self.send_header('Vary', 'Accept-Encoding')
        if enc:
            self.send_header('Content-Encoding', enc)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self._write(body)

    def _write(self, data):
        # Slow links: write in 64 KB chunks. The socket timeout then applies per chunk (progress-based)
        # instead of to the whole body, which used to abort big downloads after 30 s.
        self.connection.settimeout(120)
        mv = memoryview(data)
        for i in range(0, len(mv), 65536):
            self.wfile.write(mv[i:i + 65536])
        self.connection.settimeout(self.timeout)

    def _client_gone(self):
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b''
        except OSError:
            return True

    def _json(self, obj):
        self._send(json.dumps(obj).encode())

    def _raw(self, body, gz=None):
        self._send(body, gz, cache=True, etag=_etag(body))

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            if u.path == '/api/algos':
                self._json({a: dict(target=META[a]['target'], target_type=META[a]['target_type'],
                                   fvu=META[a]['fvu'], val_f1=META[a]['val_f1'],
                                   test_f1=META[a]['test_f1']) for a in ALGOS})
            elif u.path == '/api/weights':
                algo = qs.get('algo', ['bfs'])[0]
                r = REG[algo]; self._raw(r['weights_bytes'], r['weights_gz'])
            elif u.path == '/api/graph':
                algo = qs.get('algo', ['bfs'])[0]
                n = int(qs.get('n', ['16'])[0]); p = float(qs.get('p', ['0.4'])[0]); seed = int(qs.get('seed', ['42'])[0])
                with GRAPH_SEM:
                    if self._client_gone():
                        return                   # request was abandoned while queued
                    self._json(run_graph(algo, n, p, seed))
            elif u.path == '/api/feature_labels':
                algo = qs.get('algo', ['bfs'])[0]
                body = open(f'algos/{algo}/feature_labels.json', 'rb').read()
                self._send(body, cache=True, etag='"' + hashlib.sha1(body).hexdigest()[:20] + '"')
            elif u.path == '/api/contexts':
                algo = qs.get('algo', ['bfs'])[0]; feature = int(qs.get('feature', ['0'])[0])
                n = int(qs.get('n', ['16'])[0]); p = float(qs.get('p', ['0.4'])[0])
                key = (algo, feature, n, p)
                if key in CTX_CACHE:
                    self._json(CTX_CACHE[key])
                else:
                    with CTX_SEM:
                        if self._client_gone():
                            return
                        res = find_contexts(algo, feature, n, p, cancelled=self._client_gone)
                    if res is None:
                        return
                    if len(CTX_CACHE) > 64:
                        CTX_CACHE.clear()
                    CTX_CACHE[key] = res
                    self._json(res)
            elif u.path == '/api/instance':
                algo = qs.get('algo', ['bfs'])[0]
                n = int(qs.get('n', ['16'])[0]); p = float(qs.get('p', ['0.4'])[0]); seed = int(qs.get('seed', ['42'])[0])
                if algo not in META or not (4 <= n <= 32) or not (0.05 <= p <= 0.95):
                    self.send_error(400, 'bad parameters'); return
                inst, _ = build_instance(algo, n, p, seed, META[algo])   # sampler only: cheap, no model run
                self._json(inst)
            else:
                path = u.path.lstrip('/') or 'viewer3.html'
                m = MODEL_RE.match(path)
                if path in STATIC_FILES:
                    ctype, cache = STATIC_FILES[path], 'no-cache'
                elif m and m.group(1) in META:
                    ctype = 'application/json' if m.group(2) == 'json' else 'application/octet-stream'
                    cache = 'public, max-age=31536000, immutable'        # model bundles never change for a given image
                else:
                    self.send_error(404, 'not found'); return
                try:
                    data = open(path, 'rb').read()
                except OSError:
                    self.send_error(404, 'not found'); return
                if path == 'viewer3.html':
                    data = data.replace(b'__BUILD__', BUILD.encode())
                tag = None
                if m:
                    tag = '"' + hashlib.sha1(data).hexdigest()[:20] + '"'
                    if self.headers.get('If-None-Match') == tag:
                        self.send_response(304); self.send_header('ETag', tag); self.send_header('Cache-Control', cache); self.end_headers(); return
                self.send_response(200); self.send_header('Content-Type', ctype)
                self.send_header('Cache-Control', cache)
                if tag: self.send_header('ETag', tag)
                self.send_header('Content-Length', str(len(data))); self.end_headers()
                self._write(data)
        except OSError:
            self.close_connection = True      # client gone or write timed out mid-response: nothing more to send
        except Exception as e:
            import traceback
            self._json({'error': str(e), 'trace': traceback.format_exc()})


def _mem_available_mb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 10 ** 9


def _warm_all():
    # loads each model and triggers its JIT compile so the first real request is fast.
    # Stops early if memory gets low (small host); WARM=0 disables.
    min_mb = int(os.environ.get('WARM_MIN_MB', '450'))
    for a in ALGOS:
        if _mem_available_mb() < min_mb:
            print(f'warm-up stopped before {a}: only {_mem_available_mb()} MB available', flush=True); return
        try:
            with GRAPH_SEM:
                run_graph(a, 16, 0.4, 1)
            print(f'  warmed {a}', flush=True)
        except Exception as e:
            print(f'  warm-up {a} failed: {e}', flush=True)
    print('warm-up done', flush=True)


if __name__ == '__main__':
    host = os.environ.get('SERVER_HOST', '127.0.0.1')
    print(f'serving on http://{host}:8766/viewer3.html', flush=True)
    class QuietServer(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

        def handle_error(self, request, client_address):
            import sys
            if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
                return                     # client aborted the request: expected, not an error
            super().handle_error(request, client_address)

    srv = QuietServer((host, int(os.environ.get('PORT', '8766'))), Handler)
    if os.environ.get('WARM', '1') != '0':
        threading.Thread(target=_warm_all, daemon=True).start()
    srv.serve_forever()
