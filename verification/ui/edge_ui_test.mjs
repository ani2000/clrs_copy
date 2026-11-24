// Main page in a real browser: new graph, inspector, clamp, supernodes, by-meaning view, panels and the example loader for each algorithm; no old (node-site) pieces; no console errors.
//   env: URL (default http://127.0.0.1:8792/viewer3.html), ALGOS, PANELS_ALL=1 (full panel checks for every algorithm), CHROME_FLAGS, SHOTS (folder for screenshots)
// Drives headless Chrome through the attribution toggle: new (edge-site) graph, previous (node-site) graph, back; collects every console error and failed request.
import { spawn } from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
const SHOTS = process.env.SHOTS || os.tmpdir();
const URLB = process.env.URL || 'http://127.0.0.1:8792/viewer3.html', ALGOS = (process.env.ALGOS || 'bfs').split(',');
const sleep = ms => new Promise(r => setTimeout(r, ms)); const port = 9200 + Math.floor(Math.random() * 300);
const p = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless=new', '--disable-gpu', ...(process.env.CHROME_FLAGS || '').split(' ').filter(Boolean), `--remote-debugging-port=${port}`, `--user-data-dir=${fs.mkdtempSync(path.join(os.tmpdir(), 'eu-'))}`, 'about:blank'], { stdio: 'ignore' });
let ws; for (let i = 0; i < 100 && !ws; i++) { try { ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl; } catch { await sleep(150); } }
const sock = new WebSocket(ws); await new Promise(r => sock.addEventListener('open', r)); let id = 0; const cb = new Map(), problems = [];
sock.addEventListener('message', m => { const d = JSON.parse(m.data); if (d.id && cb.has(d.id)) { cb.get(d.id)(d.result); cb.delete(d.id); }
  else if (d.method === 'Runtime.exceptionThrown') problems.push('page error: ' + (d.params.exceptionDetails.exception?.description || '').split('\n').slice(0, 2).join(' | ').slice(0, 220));
  else if (d.method === 'Log.entryAdded' && d.params.entry.level === 'error') problems.push('console: ' + d.params.entry.text.slice(0, 160) + ' ' + (d.params.entry.url || ''));
  else if (d.method === 'Network.responseReceived' && d.params.response.status >= 400) problems.push(`HTTP ${d.params.response.status} ${d.params.response.url}`); });
const send = (method, params = {}, sessionId) => new Promise(r => { const i = ++id; cb.set(i, r); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });
const { targetId } = await send('Target.createTarget', { url: 'about:blank' }); const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true }); const S = (m, q) => send(m, q, sessionId);
for (const d of ['Page', 'Runtime', 'Log', 'Network']) await S(d + '.enable');
const ev = async e => { const r = await S('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error('eval: ' + (r.exceptionDetails.exception?.description || '').slice(0, 200)); return r.result.value; };
const waitFor = async (e, ms = 60000) => { const t = Date.now(); while (Date.now() - t < ms) { try { if (await ev(e)) return Date.now() - t; } catch {} await sleep(50); } return null; };
const shot = async n => { const r = await S('Page.captureScreenshot', { format: 'png' }); fs.writeFileSync(SHOTS + '/' + n + '.png', Buffer.from(r.data, 'base64')); };
await S('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
let fails = 0; const fail = m => { fails++; console.log('  FAIL', m); };
await S('Page.navigate', { url: URLB + (process.env.QS || '') });
for (const a of ALGOS) {
  if (a !== 'bfs') { await ev(`(()=>{const s=document.getElementById('inAlgo');s.value='${a}';s.dispatchEvent(new Event('change'))})()`); }
  const t0 = Date.now();
  const tEdge = await waitFor(`typeof AV!=='undefined' && AV.data && AV.data.mode==='edge' && curAlgo==='${a}' && document.querySelectorAll('#attribSvg circle, #attribSvg rect').length>5`, 90000);
  console.log(`${a}: edge-site graph`, tEdge === null ? 'NOT SHOWN' : tEdge + ' ms'); if (tEdge === null) { fail(a + ' edge graph'); continue; }
  const info = await ev(`({ms: AV.data.ms, nodes: AV.data.nAll, edges: AV.data.eAll, pruned: AV.data.pruned.nodes.length + '/' + AV.data.pruned.edges.length, kept: AV.data.keptInfluence, chk: AV.data.checks, svgNodes: document.querySelectorAll('#attribSvg g.nd').length, svgEdges: document.querySelectorAll('#attribSvg path.edge').length, note: document.getElementById('edgeNote').innerText.slice(0,330), q: document.getElementById('question').innerText.slice(0,160), stat: document.getElementById('gstat').innerText.slice(0,160)})`);
  console.log('   ', JSON.stringify(info).slice(0, 900)); if (!(info.chk.maxAbs < 1e-9)) fail('sum check ' + info.chk.maxAbs);
  if (a === ALGOS[0] || process.env.PANELS_ALL) {
    await shot('edge_' + a);
    // pin the most influential latent, check the inspector and an intervention
    const n = await ev(`(() => { const g = [...document.querySelectorAll('#attribSvg g.nd')].find(g => AV.data.nodesById[(g.getAttribute('data-tip')||'').slice(3)]?.type === 'feature'); if (!g) return 0; g.dispatchEvent(new MouseEvent('click', {bubbles:true})); return 1; })()`);
    await sleep(500); const insp = await ev(`document.getElementById('featDetail').innerText.slice(0,500)`); console.log('    inspector:', insp.replace(/\n+/g, ' | ').slice(0, 420)); if (!/latent f\d+/.test(insp)) fail('inspector');
    await ev(`document.getElementById('ivZero') && document.getElementById('ivZero').click()`); const tiv = await waitFor(`!!document.querySelector('#ivResult table')`, 20000);
    console.log('    intervention result:', tiv === null ? 'NONE' : tiv + ' ms', (await ev(`document.getElementById('ivResult').innerText.replace(/\\n+/g,' | ').slice(0,420)`))); if (tiv === null) fail('intervention');
    // supernodes + all views still render
    await ev(`document.querySelector('#viewSeg button[data-v=supernodes]').click()`); await sleep(500);
    console.log('    supernodes:', await ev(`document.querySelectorAll('#attribSvg g.nd').length + ' nodes'`)); await shot('edge_super_' + a);
    await ev(`document.querySelector('#viewSeg button[data-v=bynode]').click()`);
    await ev(`document.querySelectorAll('details.sec').forEach(d => d.open = true)`);
    // the old pieces must be gone from the main page
    const gone = await ev(`({ toggle: !!document.getElementById('attrSeg'), featTable: !!document.getElementById('featTable'), prevnote: !!document.querySelector('.prevnote'), mode: ATTR.mode, nodeSite: typeof AV !== 'undefined' && AV.data && AV.data.mode !== 'edge', link: (document.querySelector('a.navlink')||{}).getAttribute && document.querySelector('a.navlink').getAttribute('href'), facts: document.getElementById('etcFacts').innerText.slice(0,260) })`);
    console.log('    old pieces:', JSON.stringify(gone)); if (gone.toggle || gone.featTable || gone.prevnote || gone.mode !== 'edge' || gone.nodeSite || gone.link !== '/demo' || !/edge site/.test(gone.facts)) fail('old pieces still on the main page or facts missing');
    // meanings, top latents, dictionary, message map, by-meaning view, example loader
    const lt = await waitFor(`typeof LATS !== 'undefined' && !!LATS[curAlgo]`, 30000); console.log('    latent table loaded:', lt === null ? 'NO' : lt + ' ms'); if (lt === null) fail('latent table');
    await sleep(600);
    const ui = await ev(`({ top: document.querySelectorAll('#topLatents tr[data-id]').length, topSteps: document.querySelectorAll('#topLatents h3').length, dict: document.querySelectorAll('#latDict tr').length, dictNote: document.querySelector('#latDict p') && document.querySelector('#latDict p').innerText.slice(0,200), cells: document.querySelectorAll('#msgMap rect').length, N: G.n, tabs: document.querySelectorAll('#msgMap .msgtabs button').length, sampleRow: (document.querySelector('#topLatents tr[data-id]')||{}).innerText })`);
    console.log('    panels:', JSON.stringify(ui)); if (!(ui.top > 0 && ui.topSteps >= 1 && ui.dict >= 1 && ui.cells === ui.N * ui.N && ui.tabs >= 1)) fail('new panels incomplete');
    await ev(`document.querySelector('#topLatents tr[data-id]').click()`); await sleep(400);
    const mi = await ev(`document.getElementById('featDetail').innerText`); console.log('    meaning in inspector:', mi.replace(/\n+/g,' | ').slice(60, 520)); if (!/Meaning:|Unnamed:|never fired/.test(mi) || !/effect/.test(mi)) fail('meaning block missing');
    await ev(`document.querySelector('#viewSeg button[data-v=meaning]').click()`); await sleep(500);
    console.log('    by meaning boxes:', await ev(`[...document.querySelectorAll('#attribSvg text.nlab')].map(x => x.textContent).slice(0, 8).join(' | ')`)); await ev(`document.querySelector('#viewSeg button[data-v=bynode]').click()`); await sleep(300);
    const seed0 = await ev(`document.getElementById('inSeed').value`);
    await ev(`(() => { const b = document.querySelector('#featDetail button.avex'); if (b) b.click(); })()`);
    const ex = await waitFor(`document.getElementById('inSeed').value !== '${'$'}{0}' && AV.data && AV.data.mode === 'edge' && String(AV.data.key).includes('|' + document.getElementById('inSeed').value + '|')`.replace('${0}', seed0), 60000);
    console.log('    example loader:', ex === null ? 'NOT LOADED' : ex + ' ms -> seed ' + await ev(`document.getElementById('inSeed').value`) + ', explained node ' + await ev(`AV.data.i`) + ' at step ' + await ev(`AV.data.t`)); if (ex === null) fail('example loader');
    const reqs = await ev(`performance.getEntriesByType('resource').map(r => r.name).filter(n => /api\\/(weights|feature_labels|contexts)/.test(n)).length`); console.log('    node-site downloads on the main page:', reqs); if (reqs) fail('the main page still downloads node-site weights / labels');
  }
}
console.log(problems.length ? 'browser problems:\n  ' + [...new Set(problems)].slice(0, 12).join('\n  ') : 'no console errors, no failed requests'); if (problems.length) fails++;
console.log(fails ? `RESULT: ${fails} PROBLEM(S)` : 'RESULT: PASS'); p.kill(); process.exit(fails ? 1 : 0);
