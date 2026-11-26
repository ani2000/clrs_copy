// Cold-load timeline under an emulated slow, far-away connection (no live traffic: point URL at a local test server).
// Prints when the first graph, the pseudo-code and the attribution graph appear, plus the request waterfall.
//   env: URL (page), LAT (extra round trip in ms, default 250), KBPS (download speed in KB/s, default 450), RUNS (default 1), WATERFALL=1
import { spawn } from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
const URLB = process.env.URL || 'http://127.0.0.1:8795/viewer3.html', LAT = +(process.env.LAT || 250), KBPS = +(process.env.KBPS || 450), RUNS = +(process.env.RUNS || 1);
const sleep = ms => new Promise(r => setTimeout(r, ms)), port = 9700 + Math.floor(Math.random() * 200), dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lat-'));
const proc = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless=new', '--disable-gpu', ...(process.env.CHROME_FLAGS || '').split(' ').filter(Boolean), `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, 'about:blank'], { stdio: 'ignore' });
let ws; for (let i = 0; i < 100 && !ws; i++) { try { ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl; } catch { await sleep(150); } }
const sock = new WebSocket(ws); await new Promise(r => sock.addEventListener('open', r));
let id = 0; const cb = new Map(); let events = [];
sock.addEventListener('message', m => { const d = JSON.parse(m.data); if (d.id && cb.has(d.id)) { cb.get(d.id)(d.result); cb.delete(d.id); } else events.push(d); });
const send = (method, params = {}, sessionId) => new Promise(res => { const i = ++id; cb.set(i, res); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });
let sessionId, S, ev;
async function freshTab() {          // every run in its own tab, so nothing from an earlier run is reused
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  ({ sessionId } = await send('Target.attachToTarget', { targetId, flatten: true }));
  S = (m, p) => send(m, p, sessionId);
  for (const d of ['Page', 'Runtime', 'Network']) await S(d + '.enable');
  await S('Network.setCacheDisabled', { cacheDisabled: true });
  await S('Network.emulateNetworkConditions', { offline: false, latency: LAT, downloadThroughput: KBPS * 1024, uploadThroughput: 200 * 1024 });
  ev = async e => { const r = await S('Runtime.evaluate', { expression: e, returnByValue: true }); return r.result && r.result.value; };
  return targetId;
}
const median = a => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
const res = { first: [], code: [], attr: [], model: [] };
for (let run = 0; run < RUNS; run++) {
  const tid = await freshTab(); await sleep(200); events = [];
  const t0 = Date.now(); await S('Page.navigate', { url: URLB });
  let tFirst = null, tCode = null, tAttr = null, tModel = null;
  while (Date.now() - t0 < 60000 && (tFirst === null || tCode === null || tAttr === null || tModel === null)) {
    const st = await ev(`(() => { try { return { g: document.querySelectorAll('#graphSvg circle:not(.sk)').length, c: document.querySelectorAll('#pcCode .pl').length, a: document.querySelectorAll('#attribSvg circle, #attribSvg rect').length, m: (typeof LOCAL !== 'undefined' && LOCAL.models && LOCAL.models.bfs && LOCAL.models.bfs.state === 'ready') ? 1 : 0 }; } catch (e) { return null; } })()`);
    const now = Date.now() - t0;
    if (st) { if (tFirst === null && st.g > 0) tFirst = now; if (tCode === null && st.c > 0) tCode = now; if (tAttr === null && st.a > 0) tAttr = now; if (tModel === null && st.m) tModel = now; }
    await sleep(25);
  }
  await send('Target.closeTarget', { targetId: tid });
  res.first.push(tFirst); res.code.push(tCode); res.attr.push(tAttr); res.model.push(tModel);
  console.log(`run ${run + 1}: first graph ${tFirst} ms, model ready ${tModel} ms, attribution graph ${tAttr} ms`);
  if (process.env.WATERFALL && run === 0) {
    const req = new Map(); let base = null;
    for (const d of events) { const p = d.params; if (d.sessionId !== sessionId) continue;
      if (d.method === 'Network.requestWillBeSent') { if (base === null) base = p.timestamp; req.set(p.requestId, { url: p.request.url.replace(/^https?:\/\/[^/]+/, '').replace(/\?v=[0-9a-f]+/, ''), s: p.timestamp - base }); }
      if (d.method === 'Network.loadingFinished' && req.has(p.requestId)) { const r = req.get(p.requestId); r.e = p.timestamp - base; r.b = p.encodedDataLength; } }
    for (const r of [...req.values()].sort((a, b) => a.s - b.s)) console.log(`   ${(r.s * 1000) | 0}\t-> ${r.e === undefined ? '?' : (r.e * 1000) | 0}\tms\t${r.b === undefined ? '' : r.b}\t${r.url}`);
  }
}
if (RUNS > 1) console.log(`median of ${RUNS}: first graph ${median(res.first)} ms, model ready ${median(res.model)} ms, attribution graph ${median(res.attr)} ms`);
try { sock.close(); } catch {} proc.kill(); process.exit(0);
