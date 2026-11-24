// /demo in a real browser: it must still be the previous version (node-site graph, labelled features, label table, no edge-site pieces).
//   env: URL (default http://127.0.0.1:8792/demo), ALGOS, CHROME_FLAGS
// /demo must behave as the previous version: node-site graph, labelled features, label table, node table with predictions, no edge-site pieces.
import { spawn } from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
const URLB = process.env.URL || 'http://127.0.0.1:8792/demo', ALGOS = (process.env.ALGOS || 'bfs,dijkstra').split(',');
const sleep = ms => new Promise(r => setTimeout(r, ms)); const port = 9200 + Math.floor(Math.random() * 300);
const p = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless=new', '--disable-gpu', ...(process.env.CHROME_FLAGS || '').split(' ').filter(Boolean), `--remote-debugging-port=${port}`, `--user-data-dir=${fs.mkdtempSync(path.join(os.tmpdir(), 'du-'))}`, 'about:blank'], { stdio: 'ignore' });
let ws; for (let i = 0; i < 100 && !ws; i++) { try { ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl; } catch { await sleep(150); } }
const sock = new WebSocket(ws); await new Promise(r => sock.addEventListener('open', r)); let id = 0; const cb = new Map(), problems = [];
sock.addEventListener('message', m => { const d = JSON.parse(m.data); if (d.id && cb.has(d.id)) { cb.get(d.id)(d.result); cb.delete(d.id); }
  else if (d.method === 'Runtime.exceptionThrown') problems.push('page error: ' + (d.params.exceptionDetails.exception?.description || '').split('\n')[0].slice(0, 200));
  else if (d.method === 'Log.entryAdded' && d.params.entry.level === 'error') problems.push('console: ' + d.params.entry.text.slice(0, 160) + ' ' + (d.params.entry.url || ''));
  else if (d.method === 'Network.responseReceived' && d.params.response.status >= 400) problems.push(`HTTP ${d.params.response.status} ${d.params.response.url}`); });
const send = (method, params = {}, sessionId) => new Promise(r => { const i = ++id; cb.set(i, r); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });
const { targetId } = await send('Target.createTarget', { url: 'about:blank' }); const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true }); const S = (m, q) => send(m, q, sessionId);
for (const d of ['Page', 'Runtime', 'Log', 'Network']) await S(d + '.enable');
const ev = async e => { const r = await S('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error('eval: ' + (r.exceptionDetails.exception?.description || '').slice(0, 200)); return r.result.value; };
const waitFor = async (e, ms = 90000) => { const t = Date.now(); while (Date.now() - t < ms) { try { if (await ev(e)) return Date.now() - t; } catch {} await sleep(50); } return null; };
await S('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
let fails = 0; const fail = m => { fails++; console.log('  FAIL', m); };
await S('Page.navigate', { url: URLB });
for (const a of ALGOS) {
  if (a !== 'bfs') await ev(`(()=>{const s=document.getElementById('inAlgo');s.value='${a}';s.dispatchEvent(new Event('change'))})()`);
  const t = await waitFor(`typeof AV!=='undefined' && AV.data && AV.data.mode==='node' && curAlgo==='${a}' && document.querySelectorAll('#attribSvg g.nd').length>3`);
  console.log(`${a}: node-site graph`, t === null ? 'NOT SHOWN' : t + ' ms'); if (t === null) { fail(a + ' graph'); continue; }
  const info = await ev(`({ mode: ATTR.mode, toggle: !!document.getElementById('attrSeg'), edgeNote: !!document.getElementById('edgeNote'), labels: document.querySelectorAll('#featTable tbody tr').length, fvu: document.getElementById('gstat').innerText.slice(0, 120), title: document.title, link: document.querySelector('a.navlink').getAttribute('href'), predCells: [...document.querySelectorAll('#nodeTableBody td[data-tip$=":pred"]')].filter(td => td.innerText !== '…').length, labelled: Object.values(AV.data.nodesById).filter(n => n.type === 'feature' && n.label && n.label !== '(unlabeled)').length })`);
  console.log('   ', JSON.stringify(info)); if (info.mode !== 'node' || info.toggle || info.edgeNote || info.link !== '/' || !/previous demo/.test(info.title) || info.labels < 5 || info.labelled < 1) fail('demo page is not the previous version');
}
console.log(problems.length ? 'browser problems:\n  ' + [...new Set(problems)].slice(0, 10).join('\n  ') : 'no console errors, no failed requests'); if (problems.length) fails++;
console.log(fails ? `RESULT: ${fails} PROBLEM(S)` : 'RESULT: PASS'); p.kill(); process.exit(fails ? 1 : 0);
