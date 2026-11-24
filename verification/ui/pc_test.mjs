// Layout and pseudo-code pane in a real browser: three panels side by side, one highlighted line per step for every algorithm, rings on the graph match the engine,
// the page shows what the engine says, node click, Play, tablet and phone layouts.   env: URL, ALGOS, CHROME_FLAGS, SHOTS
// Drives headless Chrome (DevTools protocol, no dependencies) through the new layout + pseudo-code pane.
import { spawn } from 'node:child_process';
import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
const SHOTS = process.env.SHOTS || os.tmpdir();
const URL = process.env.URL || 'http://127.0.0.1:8792/viewer3.html';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const ALL = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim', 'mst_kruskal', 'topological_sort', 'articulation_points', 'bridges', 'strongly_connected_components', 'floyd_warshall'];
const only = (process.env.ALGOS || '').split(',').filter(Boolean);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pct-')); const port = 9500 + Math.floor(Math.random() * 400);
const proc = spawn(CHROME, ['--headless=new', '--disable-gpu', ...(process.env.CHROME_FLAGS||'').split(' ').filter(Boolean), `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, 'about:blank'], { stdio: 'ignore' });
let ws; for (let i = 0; i < 100 && !ws; i++) { try { ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl; } catch { await sleep(150); } }
const sock = new WebSocket(ws); await new Promise(r => sock.addEventListener('open', r));
let id = 0; const cb = new Map(); const events = [];
sock.addEventListener('message', m => { const d = JSON.parse(m.data); if (d.id && cb.has(d.id)) { const { res, rej } = cb.get(d.id); cb.delete(d.id); d.error ? rej(new Error(d.error.message)) : res(d.result); } else events.push(d); });
const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const i = ++id; cb.set(i, { res, rej }); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });
const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
const S = (m, p) => send(m, p, sessionId);
for (const d of ['Page', 'Runtime', 'Log', 'Network']) await S(d + '.enable');
const problems = [];
const drain = () => { while (events.length) { const d = events.shift(); if (d.sessionId !== sessionId) continue; const p = d.params;
  if (d.method === 'Runtime.exceptionThrown') problems.push('page error: ' + (p.exceptionDetails.exception?.description || p.exceptionDetails.text).split('\n')[0].slice(0, 160));
  else if (d.method === 'Log.entryAdded' && p.entry.level === 'error') problems.push('console: ' + p.entry.text.slice(0, 160) + ' ' + (p.entry.url || ''));
  else if (d.method === 'Network.responseReceived' && p.response.status >= 400) problems.push(`HTTP ${p.response.status} ${p.response.url}`); } };
const ev = async e => { const r = await S('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error('eval: ' + (r.exceptionDetails.exception?.description || r.exceptionDetails.text)); return r.result.value; };
const waitFor = async (e, ms = 60000) => { const t = Date.now(); while (Date.now() - t < ms) { drain(); try { if (await ev(e)) return Date.now() - t; } catch {} await sleep(40); } return null; };
const shot = async (name) => { const r = await S('Page.captureScreenshot', { format: 'png' }); fs.writeFileSync(SHOTS + '/' + name + '.png', Buffer.from(r.data, 'base64')); };
const setView = (w, h, mobile) => S('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: !!mobile });

let fails = 0; const fail = m => { fails++; console.log('  FAIL', m); };
await setView(1440, 900);
await S('Page.navigate', { url: URL });
const t1 = await waitFor(`typeof G !== 'undefined' && G && !G._preview && curAlgo === 'bfs' && document.querySelectorAll('#pcCode .pl').length > 0`);
console.log('first graph + pseudo-code pane:', t1 === null ? 'NOT REACHED' : t1 + ' ms'); if (t1 === null) fail('first graph');
await sleep(400);
// layout: three panels side by side in the order simulation | attribution | pseudo-code
const lay = await ev(`(() => { const r = id => document.getElementById(id).getBoundingClientRect(); const a = r('pnlSim'), b = r('pnlAttr'), c = r('pnlCode');
  return { sim: [a.left, a.top, a.width].map(Math.round), attr: [b.left, b.top, b.width].map(Math.round), code: [c.left, c.top, c.width].map(Math.round), overflowX: document.documentElement.scrollWidth - innerWidth }; })()`);
console.log('desktop 1440 layout (left, top, width):', JSON.stringify(lay));
if (!(lay.sim[0] < lay.attr[0] && lay.attr[0] < lay.code[0])) fail('panel order'); if (Math.abs(lay.sim[1] - lay.attr[1]) > 4 || Math.abs(lay.attr[1] - lay.code[1]) > 4) fail('panels not on one row'); if (lay.overflowX > 0) fail('horizontal overflow');
await shot('desktop_bfs');

const list = only.length ? only : ALL; const rowsOut = [];
for (const a of list) {
  await ev(`(() => { const s = document.getElementById('inAlgo'); s.value = '${a}'; s.dispatchEvent(new Event('change')); })()`);
  const g = await waitFor(`curAlgo === '${a}' && G && !G._preview && !G._computing && document.querySelectorAll('#graphSvg circle').length > 0`, 90000);
  if (g === null) { fail(a + ': graph not shown'); continue; }
  const T = await ev('G.length');
  let hiOK = 0, noHi = [], heads = new Set(), ringSteps = 0;
  for (let t = 0; t < T; t++) {
    await ev(`(() => { const r = document.getElementById('stepRange'); r.value = ${t}; r.dispatchEvent(new Event('input')); })()`);
    const info = await ev(`(() => ({ on: [...document.querySelectorAll('#pcCode .pl.on')].map(e => +e.dataset.i), head: document.getElementById('pcHead').textContent, note: document.getElementById('pcNote').textContent,
      rings: [...document.querySelectorAll('#graphSvg circle[stroke-dasharray]')].length, ringIds: [...document.querySelectorAll('#graphSvg circle[stroke-dasharray]')].length, eng: (() => { const P = pseudoNow(); return P ? { cur: P.cur, ctx: P.ctx, nodes: P.nodes } : null; })(), cx: [...document.querySelectorAll('#pcCode .pl.cx')].map(e => +e.dataset.i), tags: [...document.querySelectorAll('#pcCode .bd:not([hidden])')].map(e => e.textContent), nState: document.querySelectorAll('#pcState dt').length, step: curStep }))()`);
    if (info.step !== t) { fail(`${a}: step ${t} not applied`); break; }
    if (info.eng && (JSON.stringify(info.on) !== JSON.stringify(info.eng.cur) || JSON.stringify(info.cx) !== JSON.stringify(info.eng.ctx))) fail(`${a}: step ${t} page shows lines ${info.on}/${info.cx} but the engine says ${info.eng.cur}/${info.eng.ctx}`);
    if (info.eng && info.rings !== info.eng.nodes.filter(i => i >= 0).length) fail(`${a}: step ${t} ${info.rings} rings drawn for ${info.eng.nodes.length} nodes`);
    const narrow = await ev(`[...document.querySelectorAll('#pcCode .pl')].map(l => { const x = l.querySelector('.tx'); return [x.getBoundingClientRect().width, x.getBoundingClientRect().height, parseFloat(getComputedStyle(x).lineHeight) || 18]; }).filter(([w, h, lh]) => w < 70 || h > lh * 5).length`);
    if (narrow) fail(`${a}: step ${t} ${narrow} code line(s) squeezed or stacked letter by letter`);
    if (info.tags.some(x => x.length > 24)) fail(`${a}: step ${t} long tag ${info.tags}`);
    if (info.on.length && info.head && info.note) hiOK++; else noHi.push(t);
    heads.add(info.head.replace(/^Step \d+: /, '')); if (info.rings) ringSteps++;
  }
  // a mid-trace screenshot for a few algorithms
  if (['dijkstra', 'dfs', 'floyd_warshall', 'mst_kruskal'].includes(a)) { const mid = Math.floor(T / 2); await ev(`(() => { const r = document.getElementById('stepRange'); r.value = ${mid}; r.dispatchEvent(new Event('input')); })()`); await sleep(250); await shot('desktop_' + a); }
  const pass = noHi.length === 0; if (!pass) fail(`${a}: steps without a highlighted line: ${noHi.slice(0, 8)}`);
  rowsOut.push([a, T, hiOK, [...heads].join(' / '), ringSteps]); console.log(`${pass ? 'ok  ' : 'FAIL'} ${a.padEnd(30)} steps ${String(T).padStart(3)}  highlighted ${hiOK}/${T}  rings on ${ringSteps} steps  [${[...heads].join(', ')}]`);
}
// the existing features still work: click a node -> attribution graph appears, pseudo-code keeps following
await ev(`(() => { const s = document.getElementById('inAlgo'); s.value = 'bfs'; s.dispatchEvent(new Event('change')); })()`);
await waitFor(`curAlgo === 'bfs' && G && !G._preview && !G._computing`);
await ev(`(() => { const r = document.getElementById('stepRange'); r.value = 2; r.dispatchEvent(new Event('input')); })()`);
await ev(`(() => { const c = document.querySelectorAll('#graphSvg circle[data-tip^="node:"]')[3]; c.dispatchEvent(new MouseEvent('click', { bubbles: true })); })()`);
const at = await waitFor(`!!(AV && AV.data && AV.data.key.startsWith('bfs|')) && document.querySelectorAll('#attribSvg circle, #attribSvg rect').length > 0`, 20000);
console.log('node click -> attribution graph:', at === null ? 'NOT SHOWN' : at + ' ms'); if (at === null) fail('attribution after click');
await ev(`(() => { const r = document.getElementById('stepRange'); r.value = 3; r.dispatchEvent(new Event('input')); })()`); await sleep(300);
const still = await ev(`document.querySelectorAll('#pcCode .pl.on').length > 0 && document.querySelectorAll('#attribSvg *').length > 5`); if (!still) fail('pseudo-code or attribution lost after click');
await shot('desktop_attribution');
// play button advances the steps and the pseudo-code follows
await ev(`(() => { document.getElementById('btnPlay').click(); })()`); await sleep(3600);
const played = await ev(`curStep`); await ev(`document.getElementById('btnPlay').click()`); console.log('play: step advanced to', played); if (!(played > 3)) fail('play did not advance');
// medium and phone widths
for (const [w, h, m, name] of [[1024, 800, false, 'tablet'], [390, 844, true, 'phone']]) {
  await setView(w, h, m); await sleep(500);
  const l = await ev(`(() => { const r = id => document.getElementById(id).getBoundingClientRect(); const o = {}; for (const k of ['pnlSim','pnlCode','pnlAttr']) { const q = r(k); o[k] = [Math.round(q.left), Math.round(q.top), Math.round(q.width)]; } o.overflowX = document.documentElement.scrollWidth - innerWidth; return o; })()`);
  console.log(`${name} ${w}px:`, JSON.stringify(l)); if (l.overflowX > 0) fail(name + ' horizontal overflow'); await shot(name);
}
drain(); const uniq = [...new Set(problems)];
console.log(uniq.length ? 'browser problems:\n  ' + uniq.slice(0, 12).join('\n  ') : 'no console errors, no failed requests');
if (uniq.length) fails++;
console.log(fails ? `RESULT: ${fails} PROBLEM(S)` : 'RESULT: PASS');
try { sock.close(); } catch {} proc.kill(); process.exit(fails ? 1 : 0);
