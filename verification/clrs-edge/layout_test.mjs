// Collapsible sections of the lower part of the page: defaults, opening and closing, memory across reloads, inspector sub-sections, no overflow at three widths.
import { spawn } from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
const URLB = process.env.URL || 'http://127.0.0.1:8792/viewer3.html';
const sleep = ms => new Promise(r => setTimeout(r, ms)), port = 9600 + Math.floor(Math.random() * 300), dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lt-'));
const proc = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless=new', '--disable-gpu', ...(process.env.CHROME_FLAGS || '').split(' ').filter(Boolean), `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, 'about:blank'], { stdio: 'ignore' });
let ws; for (let i = 0; i < 100 && !ws; i++) { try { ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl; } catch { await sleep(150); } }
const sock = new WebSocket(ws); await new Promise(r => sock.addEventListener('open', r)); let id = 0; const cb = new Map(), problems = [];
sock.addEventListener('message', m => { const d = JSON.parse(m.data); if (d.id && cb.has(d.id)) { cb.get(d.id)(d.result); cb.delete(d.id); } else if (d.method === 'Runtime.exceptionThrown') problems.push((d.params.exceptionDetails.exception?.description || '').split('\n')[0]); });
const send = (method, params = {}, sessionId) => new Promise(r => { const i = ++id; cb.set(i, r); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });
const { targetId } = await send('Target.createTarget', { url: 'about:blank' }); const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true }); const S = (m, q) => send(m, q, sessionId);
await S('Runtime.enable'); await S('Page.enable');
const ev = async e => { const r = await S('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error('eval: ' + (r.exceptionDetails.exception?.description || '').slice(0, 200)); return r.result.value; };
const waitFor = async (e, ms = 60000) => { const t = Date.now(); while (Date.now() - t < ms) { try { if (await ev(e)) return true; } catch {} await sleep(60); } return false; };
let fails = 0; const ok = (c, m) => { console.log(c ? '  ok  ' : '  FAIL', m); if (!c) fails++; };
const ready = `typeof AV!=='undefined' && AV.data && AV.data.mode==='edge' && LATS && LATS[curAlgo] && document.querySelectorAll('#attribSvg g.nd').length>3`;
const state = () => ev(`Object.fromEntries([...document.querySelectorAll('details.sec')].map(d => [d.id, d.open]))`);
await S('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
await S('Page.navigate', { url: URLB }); ok(await waitFor(ready), 'page ready'); await sleep(600);
let st = await state(); console.log('  default state', JSON.stringify(st));
ok(st.secTop && st.secInsp && st.secIv && !st.secDict && !st.secNodes && !st.secModel && !st.secRead, 'defaults: top latents, inspector and intervention open; the rest collapsed');
ok((await ev(`document.querySelectorAll('details.sec').length`)) === 7, 'seven sections');
const meta = await ev(`Object.fromEntries([...document.querySelectorAll('.secmeta')].map(e => [e.id, e.textContent]))`); console.log('  headers:', JSON.stringify(meta));
ok(meta.meta_secTop && meta.meta_secDict && meta.meta_secNodes && meta.meta_secModel && meta.meta_secInsp, 'every collapsed or open header carries a live summary');
// open the dictionary by clicking its summary, reload, it must stay open; then close it again
await ev(`document.querySelector('#secDict > summary').click()`); await sleep(300); ok((await state()).secDict, 'clicking the header opens a section');
ok((await ev(`document.querySelectorAll('#latDict tr').length`)) > 1, 'the opened dictionary has rows');
await S('Page.navigate', { url: URLB }); await waitFor(ready); await sleep(500); st = await state(); ok(st.secDict === true, 'an opened section stays open after a reload');
await ev(`document.querySelector('#secDict > summary').click()`); await sleep(200); await S('Page.navigate', { url: URLB }); await waitFor(ready); await sleep(500); st = await state(); ok(st.secDict === false, 'a closed section stays closed after a reload');
// inspector sub-sections: open "Outgoing edges", pin another latent, it must still be open
await ev(`[...document.querySelectorAll('#featDetail details.isecd')].find(d => d.querySelector('summary').textContent.startsWith('Outgoing')).querySelector('summary').click()`); await sleep(200);
await ev(`document.querySelectorAll('#topLatents tr[data-id]')[1].click()`); await sleep(500);
ok(await ev(`[...document.querySelectorAll('#featDetail details.isecd')].find(d => d.querySelector('summary').textContent.startsWith('Outgoing')).open`), 'an inspector sub-section keeps its state when another latent is pinned');
ok(await ev(`[...document.querySelectorAll('#featDetail details.isecd')].map(d => d.querySelector('summary').textContent.split(' ')[0]).join(',')`) === 'What,Best-matching,Strongest,Total,Incoming,Outgoing', 'inspector sub-sections in order: meaning, conditions, examples, total effect, incoming, outgoing');
// layout at three widths
for (const [w, h, m, name] of [[1440, 1000, false, 'wide'], [1024, 800, false, 'tablet'], [390, 844, true, 'phone']]) {
  await S('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: m }); await sleep(500);
  await ev(`document.querySelectorAll('details.sec').forEach(d => d.open = true)`); await sleep(300);
  const r = await ev(`({ over: document.documentElement.scrollWidth - innerWidth, cols: getComputedStyle(document.querySelector('.lower')).gridTemplateColumns.split(' ').length, mainW: Math.round(document.querySelector('.lcol.main').getBoundingClientRect().width), sideW: Math.round(document.querySelector('.lcol.side').getBoundingClientRect().width), tableOver: [...document.querySelectorAll('details.sec')].filter(d => d.scrollWidth > d.clientWidth + 1).map(d => d.id) })`);
  console.log('  ', name, JSON.stringify(r)); ok(r.over <= 0 && r.tableOver.length === 0, `${name}: no horizontal overflow`);
  if (name === 'phone') { const H = await ev(`document.documentElement.scrollHeight`); const sh = await S('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: await ev(`Math.round(document.querySelector('.lower').getBoundingClientRect().top + scrollY)`), width: 390, height: 2400, scale: 1 } }); if (process.env.SHOTS) fs.writeFileSync(path.join(process.env.SHOTS, 'lower_phone.png'), Buffer.from(sh.data, 'base64')); }
}
ok(problems.length === 0, 'no page errors ' + problems.join(' | ').slice(0, 200)); console.log(fails ? `RESULT: ${fails} PROBLEM(S)` : 'RESULT: PASS'); proc.kill(); process.exit(fails ? 1 : 0);
