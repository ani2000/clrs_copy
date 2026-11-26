// Local test server: serves this repository's page, scripts and edge_tc files; everything else (model bundles, /api/*) is fetched from the live site and cached on disk.
//   env: PORT (8792), REPO (repository folder), CACHE (cache folder), LIVE_BUILD (build id of the live site; read from it when absent)
// Local test server for the edge-site branch: serves the repo's page, scripts and edge_tc files; everything else (model bundles, /api/*) is proxied to the live site and cached.
import http from 'node:http'; import zlib from 'node:zlib'; import fs from 'node:fs'; import crypto from 'node:crypto';
const REPO = (process.env.REPO || process.cwd()) + '/', CACHE = process.env.CACHE || '/tmp/clrs-proxycache';
const ORIGIN = 'http://98.86.163.27', PORT = +(process.env.PORT || 8792);
let BUILD = process.env.LIVE_BUILD, algos = null;
fs.mkdirSync(CACHE, { recursive: true });
if (!BUILD) { const t = await (await fetch(ORIGIN + '/viewer3.html')).text(); BUILD = /BUILD = '([0-9a-f]{10})'/.exec(t)[1]; }
const LOCAL = new Set(['explain.js', 'attrview.js', 'attrib.js', 'mpnn.js', 'mpnn_worker.js', 'clrs-edge-attrib.js']);
const getAlgos = async () => algos || (algos = await (await fetch(ORIGIN + '/api/algos?v=' + BUILD)).text());
// like the real server: pages, scripts, json and the weight files are sent gzip-compressed
const gzOK = (req, type) => /gzip/.test(req.headers['accept-encoding'] || '') && /html|javascript|json|octet-stream/.test(type);
const sendBody = (req, res, status, headers, body) => { if (gzOK(req, headers['Content-Type'] || '')) { const raw = body.length; body = zlib.gzipSync(body, { level: 6 }); headers = { ...headers, 'Content-Encoding': 'gzip', ...(/octet-stream/.test(headers['Content-Type']) ? { 'X-Raw-Length': raw } : {}) }; } res.writeHead(status, { ...headers, 'Content-Length': body.length }); res.end(body); };
http.createServer(async (req, res) => {
  try {
    const u = new URL(req.url, 'http://x'), path = u.pathname.replace(/^\//, '') || 'viewer3.html';
    if (path === 'demo' || path === 'demo.html') { const html = fs.readFileSync(REPO + 'demo.html', 'utf8').replaceAll('__BUILD__', BUILD).replace('__ALGOS_JSON__', await getAlgos()); return sendBody(req, res, 200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' }, Buffer.from(html)); }
    if (path === 'viewer3.html') { const html = fs.readFileSync(REPO + 'viewer3.html', 'utf8').replaceAll('__BUILD__', BUILD).replace('__ALGOS_JSON__', await getAlgos()); return sendBody(req, res, 200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' }, Buffer.from(html)); }
    if (LOCAL.has(path)) return sendBody(req, res, 200, { 'Content-Type': 'application/javascript', 'Cache-Control': 'no-store' }, fs.readFileSync(REPO + path));
    if (/^edge_tc\/[a-z_.-]+\.(json|bin)$/.test(path)) return sendBody(req, res, 200, { 'Content-Type': path.endsWith('json') ? 'application/json' : 'application/octet-stream', 'Cache-Control': 'no-store' }, fs.readFileSync(REPO + path));
    const key = CACHE + '/' + crypto.createHash('sha1').update(req.url).digest('hex');
    if (fs.existsSync(key + '.bin')) { const m = JSON.parse(fs.readFileSync(key + '.json')); return sendBody(req, res, m.status, m.headers, fs.readFileSync(key + '.bin')); }
    const r = await fetch(ORIGIN + req.url, { headers: { 'Accept-Encoding': 'identity' } }); const buf = Buffer.from(await r.arrayBuffer());
    const headers = { 'Content-Type': r.headers.get('content-type') || 'application/octet-stream', 'Cache-Control': 'no-store' };
    if (r.status === 200) { fs.writeFileSync(key + '.bin', buf); fs.writeFileSync(key + '.json', JSON.stringify({ status: 200, headers })); }
    sendBody(req, res, r.status, headers, buf);
  } catch (e) { res.writeHead(502); res.end(String(e)); }
}).listen(PORT, '127.0.0.1', () => console.log('edge test server on', PORT, 'live build', BUILD));
