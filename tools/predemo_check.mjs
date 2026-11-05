// Pre-demo check: opens the public URL in a real (headless) browser, loads all 12 algorithms one after the other and prints
// pass/fail with timings. Dependency-free (talks to Chrome over the DevTools protocol).
// usage: node --experimental-websocket tools/predemo_check.mjs https://xxxx.cloudfront.net [maxSeconds=30]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const base = (process.argv[2] || '').replace(/\/+$/, '').replace(/\/viewer3\.html$/, '');
const LIMIT = (+process.argv[3] || 30) * 1000;
if (!/^https?:\/\//.test(base)) { console.error('usage: predemo_check.mjs <https://address> [maxSeconds]'); process.exit(2); }
const CHROME = process.env.CHROME || 'google-chrome';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const ALL = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim', 'mst_kruskal', 'topological_sort', 'articulation_points', 'bridges', 'strongly_connected_components', 'floyd_warshall'];

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'predemo-'));
const port = 9400 + Math.floor(Math.random() * 400);
const proc = spawn(CHROME, ['--headless=new', '--no-sandbox', '--disable-gpu', `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, 'about:blank'], { stdio: 'ignore' });
let ws, id = 0; const cb = new Map(); const events = [];
for (let i = 0; i < 100 && !ws; i++) { try { ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl; } catch (e) { await sleep(150); } }
if (!ws) { console.error('browser did not start'); process.exit(2); }
const sock = new WebSocket(ws); await new Promise(r => sock.addEventListener('open', r));
sock.addEventListener('message', m => { const d = JSON.parse(m.data); if (d.id && cb.has(d.id)) { const { res, rej } = cb.get(d.id); cb.delete(d.id); d.error ? rej(new Error(d.error.message)) : res(d.result); } else events.push(d); });
const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const i = ++id; cb.set(i, { res, rej }); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });
const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
const S = (m, p) => send(m, p, sessionId);
for (const d of ['Network', 'Runtime', 'Page', 'Log']) await S(d + '.enable');
const problems = [];
const failedReqs = [];
const reqUrl = new Map();
function drain() {
  while (events.length) {
    const d = events.shift(); if (d.sessionId !== sessionId) continue; const p = d.params;
    if (d.method === 'Network.requestWillBeSent') reqUrl.set(p.requestId, p.request.url);
    else if (d.method === 'Network.loadingFailed' && !p.canceled) failedReqs.push(`${reqUrl.get(p.requestId) || '?'} (${p.errorText})`);
    else if (d.method === 'Network.responseReceived' && p.response.status >= 400) failedReqs.push(`${p.response.url} (HTTP ${p.response.status})`);
    else if (d.method === 'Runtime.exceptionThrown') problems.push('page error: ' + ((p.exceptionDetails.exception && p.exceptionDetails.exception.description) || p.exceptionDetails.text).split('\n')[0].slice(0, 140));
    else if (d.method === 'Log.entryAdded' && p.entry.level === 'error') problems.push('console: ' + p.entry.text.slice(0, 140) + ' ' + (p.entry.url || ''));
  }
}
async function ev(expr) { const r = await S('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error('eval'); return r.result.value; }
async function waitFor(expr, ms) { const t = Date.now(); while (Date.now() - t < ms) { drain(); try { if (await ev(expr)) return Date.now() - t; } catch (e) {} await sleep(25); } return null; }

console.log(`Pre-demo check of ${base}  (each step must finish within ${LIMIT / 1000} s)`);
const head = await fetch(base + '/viewer3.html', { redirect: 'manual' }).catch(e => ({ status: 0, err: e.message }));
console.log(`page request: HTTP ${head.status}${head.err ? ' ' + head.err : ''}`);
const rows = []; let ok = head.status === 200;
if (!ok) problems.push(`page did not answer with 200 (got ${head.status})`);

const t0 = Date.now();
await S('Page.navigate', { url: base + '/viewer3.html' });
const first = await waitFor(`typeof G !== 'undefined' && G && !G._preview && curAlgo === 'bfs'`, LIMIT * 2);
console.log(`first graph on screen: ${first === null ? 'NOT REACHED' : first + ' ms'}`);
if (first === null) ok = false;
for (const a of ALL) {
  const t = Date.now();
  await ev(`(() => { const s = document.getElementById('inAlgo'); s.value = '${a}'; s.dispatchEvent(new Event('change')); })()`);
  const g = await waitFor(`curAlgo === '${a}' && G && !G._preview && document.querySelectorAll('#graphSvg circle').length > 0`, LIMIT);
  const at = g === null ? null : await waitFor(`!!(AV.data && AV.data.key.startsWith('${a}|'))`, LIMIT);
  let click = null;
  if (at !== null) {
    const k = await ev(`(G.n > 1 ? ((selected ? selected.node : 0) + 1) % G.n : 0)`);
    const tc = Date.now(); await ev(`selectTarget(${k})`);
    const c = await waitFor(`!!(AV.data && AV.data.i === ${k} && AV.data.key.startsWith('${a}|'))`, LIMIT);
    click = c;
  }
  rows.push({ a, graph: g, attr: at, click, pass: g !== null && at !== null && click !== null });
}
drain();
console.log('\nalgorithm                        graph     attribution   node click   result');
for (const r of rows) {
  const f = x => (x === null || x === undefined ? 'FAILED' : x + ' ms').padEnd(12);
  console.log(`${r.a.padEnd(32)} ${f(r.graph)} ${f(r.attr)}  ${f(r.click)} ${r.pass ? 'PASS' : 'FAIL'}`);
  if (!r.pass) ok = false;
}
if (failedReqs.length) { console.log('\nfailed requests:'); [...new Set(failedReqs)].slice(0, 10).forEach(x => console.log('  ' + x)); ok = false; }
const mixed = problems.filter(p => /mixed content|blocked/i.test(p));
if (problems.length) { console.log('\nbrowser problems:'); [...new Set(problems)].slice(0, 10).forEach(x => console.log('  ' + x)); ok = false; }
console.log(`\ntotal ${(Date.now() - t0) / 1000} s;  ${rows.filter(r => r.pass).length}/12 algorithms passed;  mixed-content problems: ${mixed.length}`);
console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
try { sock.close(); } catch (e) {} proc.kill();
process.exit(ok ? 0 : 1);
