// Every number the page shows must be real.  This drives headless Chrome on the page, reads the numbers off the DOM, and compares each one with a fresh recomputation in Node
// (same instance, same model bundle, same transcoder) and with the ORIGINAL training files (report.json, summary_v3.json, the latent tables).
//   env: URL (page), MODELS (model bundles), REPORTS (clrs-transcoder-out), SUMMARY (results_v3/summary_v3.json), ALGOS, CHROME_FLAGS
import { spawn } from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path'; import { createRequire } from 'node:module';
const require = createRequire(import.meta.url), ROOT = path.join(path.dirname(new URL(import.meta.url).pathname), '..', '..');
const E = require(ROOT + '/clrs-edge-attrib.js'), MP = require(ROOT + '/mpnn.js'), AT = require(ROOT + '/attrib.js');
const URLB = process.env.URL || 'http://127.0.0.1:8792/viewer3.html', MODELS = process.env.MODELS, REPORTS = process.env.REPORTS, SUMMARY = JSON.parse(fs.readFileSync(process.env.SUMMARY)), ALGOS = (process.env.ALGOS || 'bfs,dijkstra,dfs,mst_kruskal').split(',');
const sleep = ms => new Promise(r => setTimeout(r, ms)), port = 9400 + Math.floor(Math.random() * 300);
const proc = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless=new', '--disable-gpu', ...(process.env.CHROME_FLAGS || '').split(' ').filter(Boolean), `--remote-debugging-port=${port}`, `--user-data-dir=${fs.mkdtempSync(path.join(os.tmpdir(), 'ut-'))}`, 'about:blank'], { stdio: 'ignore' });
let ws; for (let i = 0; i < 100 && !ws; i++) { try { ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl; } catch { await sleep(150); } }
const sock = new WebSocket(ws); await new Promise(r => sock.addEventListener('open', r)); let id = 0; const cb = new Map();
sock.addEventListener('message', m => { const d = JSON.parse(m.data); if (d.id && cb.has(d.id)) { cb.get(d.id)(d.result); cb.delete(d.id); } });
const send = (method, params = {}, sessionId) => new Promise(r => { const i = ++id; cb.set(i, r); sock.send(JSON.stringify({ id: i, method, params, sessionId })); });
const { targetId } = await send('Target.createTarget', { url: 'about:blank' }); const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true }); const S = (m, q) => send(m, q, sessionId);
await S('Runtime.enable');
const ev = async e => { const r = await S('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error('eval: ' + (r.exceptionDetails.exception?.description || '').slice(0, 300)); return r.result.value; };
const waitFor = async (e, ms = 90000) => { const t = Date.now(); while (Date.now() - t < ms) { try { if (await ev(e)) return true; } catch {} await sleep(60); } return false; };
await S('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
let checks = 0, bad = 0; const eq = (what, got, want) => { checks++; if (String(got) !== String(want)) { bad++; console.log('  MISMATCH', what, '| page:', got, '| truth:', want); } };
const pc1 = x => (x * 100).toFixed(1) + '%', f3 = x => (+x).toFixed(3);
await S('Page.navigate', { url: URLB });
for (const a of ALGOS) {
  if (a !== ALGOS[0]) await ev(`(()=>{const s=document.getElementById('inAlgo');s.value='${a}';s.dispatchEvent(new Event('change'))})()`);
  if (!await waitFor(`typeof AV!=='undefined' && AV.data && AV.data.mode==='edge' && curAlgo==='${a}' && LATS && LATS['${a}'] && document.querySelectorAll('#attribSvg g.nd').length>3 && ATTR.summary`)) { bad++; console.log('FAIL: page did not settle for', a); continue; }
  await sleep(700); await ev(`document.querySelectorAll('details.sec').forEach(d => d.open = true)`);
  // pin the most influential latent of the explained step and clamp it
  await ev(`document.querySelector('#topLatents tr[data-id]').click()`); await sleep(500);
  await ev(`document.querySelectorAll('#featDetail details').forEach(d => d.open = true)`);
  await ev(`document.getElementById('ivZero').click()`); await waitFor(`!!document.querySelector('#ivResult table')`, 30000);
  const P = await ev(`(() => { const txt = e => e.innerText.trim(), d = AV.data, nd = d.nodesById[AV.pinned];
    const bars = sec => { const el = [...document.querySelectorAll('#featDetail details.isecd')].find(x => x.querySelector('summary').textContent.startsWith(sec)); return el ? [...el.querySelectorAll('.bars')].flatMap(b => { const k = [...b.children]; const o = []; for (let i = 0; i + 2 < k.length; i += 3) o.push([txt(k[i]), k[i + 2].innerText.trim()]); return o; }) : null; };
    const ikv = () => { const o = {}; document.querySelectorAll('#featDetail dl.ikv').forEach(dl => { const c = [...dl.children]; for (let i = 0; i + 1 < c.length; i += 2) o[txt(c[i])] = txt(c[i + 1]); }); return o; };
    return { algo: curAlgo, inst: G._inst, t: d.t, i: d.i, main: W.target, ttype: W.target_type, pinned: AV.pinned,
      top: [...document.querySelectorAll('#topLatents h3')].map(h => { const tbl = h.nextElementSibling; return { step: +/Step (\\d+)/.exec(h.innerText)[1], rows: tbl && tbl.tagName === 'TABLE' ? [...tbl.querySelectorAll('tr[data-id]')].map(tr => ({ id: tr.dataset.id, cells: [...tr.cells].map(txt) })) : [] }; }),
      nodes: [...document.querySelectorAll('#nodeTableBody tr')].map(tr => [...tr.cells].map(txt)),
      f1s: txt(document.getElementById('f1s')), target: txt(document.getElementById('targetName')),
      facts: Object.fromEntries([...document.querySelectorAll('#etcFacts .kv')].map(k => [txt(k.children[0]), txt(k.children[1])])),
      dict: [...document.querySelectorAll('#latDict tr')].slice(1).map(tr => [...tr.cells].map(txt)), dictNote: txt(document.querySelector('#latDict p')),
      insp: { title: txt(document.querySelector('#featDetail .ititle')), kv: ikv(), total: bars('Total effect'), inc: bars('Incoming'), out: bars('Outgoing'),
        cond: [...document.querySelectorAll('#featDetail .condtab tr')].slice(1).map(tr => [...tr.cells].map(txt)), ex: [...document.querySelectorAll('#featDetail button.avex')].map(b => [b.dataset.seed, b.dataset.t, b.dataset.r]) },
      map: [...document.querySelectorAll('#msgMap rect')].map(r => r.querySelector('title').textContent), mapStep: MSG_STEP, shownSteps: [...document.querySelectorAll('#msgMap .msgtabs button')].map(b => b.innerText),
      iv: [...document.querySelectorAll('#ivResult table tr')].slice(1).map(tr => [...tr.cells].map(txt)), ivNote: txt(document.querySelector('#ivResult .note')) }; })()`);
  // ---------- recompute in Node ----------
  const man = JSON.parse(fs.readFileSync(path.join(MODELS, a + '.json'))), b = fs.readFileSync(path.join(MODELS, a + '.bin')), M = MP.loadModel(man, new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  const T = E.prepTC(JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.json'))), fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.bin'))), J = JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.latents.json')));
  const ptr = M.fields.some(f => f.name === 'pi_h' && f.stage === 'hint' && f.loc === 'node' && f.type === 'pointer') ? 'pi_h' : null, inst = P.inst, N = inst.n;
  const ex = E.explain(MP, AT, M, T, inst, { t: P.t, i: P.i, main: P.main, pointerField: ptr, window: 3 }), pr = E.prune(ex), pk = E.pack(ex, pr, inst), byId = Object.fromEntries(pk.nodes.map(n => [n.id, n]));
  const rep = JSON.parse(fs.readFileSync(path.join(REPORTS, a, 'report.json'))), sm = SUMMARY.results.find(r => r.algo === a);
  const name = (v) => v && v.tier !== '-' ? J.conditions[v.best[0][0]].id.replace(/_/g, ' ') : null;
  const effShort = v => v && v.eff ? `${v.eff[0] >= 0 ? '+' : ''}${v.eff[0].toFixed(3)}${v.eff[3] >= 0.9 ? ' (raises)' : v.eff[3] >= 0.65 ? ' (mostly raises)' : v.eff[3] <= 0.1 ? ' (lowers)' : v.eff[3] <= 0.35 ? ' (mostly lowers)' : ' (mixed)'}` : '-';
  console.log(`== ${a}: explaining node ${P.i} at step ${P.t}; pinned ${P.pinned}`);
  // 1. top latents at each step
  for (const blk of P.top) {
    const want = pk.pruned.nodes.map(i => byId[i]).filter(n => n && n.type === 'feature' && n.t === blk.step).sort((x, y) => pk.influence[y.id] - pk.influence[x.id]).slice(0, 6);
    eq(`top latents step ${blk.step}: row count`, blk.rows.length, want.length);
    want.forEach((n, k) => { const r = blk.rows[k]; if (!r) return; const v = J.latents[n.f];
      eq(`top[${blk.step}][${k}] id`, r.id, n.id); eq(`top[${blk.step}][${k}] latent`, r.cells[0], 'f' + n.f); eq(`top[${blk.step}][${k}] pair`, r.cells[1], `${n.s} → ${n.i}`);
      eq(`top[${blk.step}][${k}] activation`, r.cells[2], n.act.toFixed(2)); eq(`top[${blk.step}][${k}] influence`, r.cells[3], pk.influence[n.id].toFixed(4));
      eq(`top[${blk.step}][${k}] meaning`, r.cells[4].replace(/\s+[AB]$/, ''), v && v.tier !== '-' ? name(v) : 'unnamed'); eq(`top[${blk.step}][${k}] effect`, r.cells[5], effShort(v)); });
  }
  eq('steps listed', P.top.map(x => x.step).join(','), Array.from({ length: P.t - ex.tLo + 1 }, (_, k) => P.t - k).join(','));
  // 2. per-node predictions: the real model, never a transcoder approximation
  const L = ex.run.rec[ex.run.rec.length - 1].logits[P.main], C = L.length / N, tt = inst.target_true, cat = P.ttype === 'categorical';
  const predOf = node => P.ttype === 'categorical' ? (() => { let k = 0; for (let c = 1; c < C; c++) if (L[node * C + c] > L[node * C + k]) k = c; return 'class ' + k; })() : P.ttype === 'mask' ? (1 / (1 + Math.exp(-L[node]))).toFixed(2)
    : P.ttype === 'mask_one' ? (() => { const m = Math.max(...L), e = Array.from(L, v => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0); return (e[node] / s).toFixed(2); })() : L[node].toFixed(2);
  eq('node table rows', P.nodes.length, N);
  P.nodes.forEach((row, node) => { eq(`node ${node} pred`, row[2], predOf(node)); eq(`node ${node} state now`, row[1], tt[P.t][node].toFixed(cat ? 0 : 2)); eq(`node ${node} true next`, row[3], tt[P.t + 1][node].toFixed(cat ? 0 : 2)); });
  // 3. model and transcoder facts against the ORIGINAL training files
  eq('val / test score', P.f1s.split('[')[0].trim(), `${f3(sm.best_val_f1)} / ${f3(sm.test_f1)}`);
  const e1 = rep.E1, e2 = rep.E2, F = P.facts, steps = e1.val.nmse_per_step, hm = Object.values(rep.direct.hidden_rel_error.direct);
  eq('facts: site', F['transcoder'], `edge site · ${rep.config.D} latents · ${rep.config.k} active per pair`); eq('facts: tier', F['tier'], rep.tier);
  eq('facts: FVU', F['reconstruction error (FVU)'], `val ${e1.val.nmse.toFixed(4)} · train ${e1.train.nmse.toFixed(4)} · 64 nodes ${e1.test_n64 ? e1.test_n64.nmse.toFixed(4) : 'not measured'}`);
  eq('facts: dead latents', F['latents that never fire'], `${pc1(e1.train.dead_frac)} of ${rep.config.D}`); eq('facts: L0', F['active latents per pair'], e1.val.l0.toFixed(2));
  eq('facts: replacement', F['replacement keeps (F1)'], `16 nodes ${pc1(e2.val.retention_full)} (${e2.val.full.score.toFixed(3)} of ${e2.val.off.score.toFixed(3)})${e2.test ? ` · 64 nodes ${pc1(e2.test.retention_full)}` : ''}`);
  eq('facts: winner flips', F['winner flips (replacement alone)'], `${pc1(rep.direct.winner_flips.flip_rate)} of channels`); eq('facts: hidden error', F['one-step hidden-state error'], `${pc1(hm.reduce((x, y) => x + y, 0) / hm.length)} (mean over steps)`);
  eq('facts: trained', F['trained'], `${rep.config.steps.toLocaleString()} steps on ${rep.rows.train.toLocaleString()} rows, ${rep.minutes} min`);
  eq('facts: per-step error endpoints', F['error by step (val)'].replace(/\s+/g, ' ').trim(), `${steps['0'].toFixed(4)} → ${steps[String(Object.keys(steps).length - 1)].toFixed(4)}`);
  // 4. dictionary
  const named = Object.entries(J.latents).filter(([f, v]) => v.tier !== '-').sort((x, y) => y[1].best[0][3] - x[1].best[0][3] || y[1].n - x[1].n).slice(0, 20);
  eq('dictionary rows', P.dict.length, named.length);
  named.forEach(([f, v], k) => { const r = P.dict[k]; if (!r) return; eq(`dict[${k}] latent`, r[0], 'f' + f); eq(`dict[${k}] meaning`, r[1], J.conditions[v.best[0][0]].text); eq(`dict[${k}] tier`, r[2], v.tier);
    eq(`dict[${k}] P R F1`, r.slice(3, 6).join(' '), v.best[0].slice(1, 4).map(x => x.toFixed(2)).join(' ')); eq(`dict[${k}] fires`, r[6], (v.rate * 100).toFixed(1) + '%'); eq(`dict[${k}] effect`, r[7], v.eff ? (v.eff[0] >= 0 ? '+' : '') + v.eff[0].toFixed(3) : '-'); });
  eq('dictionary note counts', P.dictNote.replace(/\s+/g, ' ').slice(0, 90), `${J.fired} of ${J.fired + J.never_fired} latents fired in ${J.graphs} graphs (${J.rows_fit + J.rows_held} pair rows); ${J.named} can be named (tier A: ${J.tier_a}); ${J.never_fired} never fired.`.slice(0, 90).replace('; ', '; ').replace(/\*/g, ''));
  // 5. inspector of the pinned latent
  const nd = byId[P.pinned], v = J.latents[nd.f], I = P.insp, fi = nd.fires;
  eq('inspector title', I.title, `latent f${nd.f} at step ${nd.t}, pair ${nd.s} → ${nd.i}`); eq('inspector activation', I.kv['activation'], nd.act.toFixed(3)); eq('inspector influence', I.kv['influence on targets'], pk.influence[nd.id].toFixed(4));
  eq('inspector fires in graph', I.kv['active in this graph'].split(' (e.g.')[0], `${fi.n} of ${N * N} pairs at step ${nd.t}: ${fi.real} real edges, ${fi.self} self pairs`);
  eq('inspector fires rate', I.kv['fires on'], `${(v.rate * 100).toFixed(1)}% of pairs (mean activation ${v.act}, max ${v.max})`);
  const pcs = x => (x * 100).toFixed(0) + '%', base = c => J.conditions.find(x => x.id === c).base_rate_held;
  eq('inspector pair kinds', I.kv['on which pairs'], `${pcs(v.kind[0])} real edges (${pcs(base('edge'))} of all pairs), ${pcs(v.kind[1])} non-edges (${pcs(base('nonedge'))}), ${pcs(v.kind[2])} self pairs (${pcs(base('self'))})`);
  eq('inspector conditions', I.cond.map(r => r.join('|')).join(';'), v.best.map(x => [J.conditions[x[0]].text, x[1].toFixed(2), x[2].toFixed(2), x[3].toFixed(2)].join('|')).join(';'));
  eq('inspector examples', I.ex.map(x => x.join(',')).join(';'), v.ex.map(x => [x[0], x[1], x[3]].join(',')).join(';'));
  const te = pk.totalEffect[nd.id] || {}, teRows = Object.entries(te).filter(([i]) => pk.targets.some(x => x.id === i && (x.prob >= 0.01 || x.always))).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).slice(0, 6);
  eq('inspector total effect', (I.total || []).map(r => r[1]).join(' '), teRows.map(([i, w]) => (w >= 0 ? '+' : '') + w.toFixed(2)).join(' '));
  const edgesOf = dir => pk.pruned.edges.filter(e => dir === 'in' ? e.d === nd.id : e.s === nd.id).filter(e => byId[e.s] && byId[e.d]).sort((x, y) => Math.abs(y.w) - Math.abs(x.w)).slice(0, 10);
  eq('inspector incoming edges', (I.inc || []).map(r => r[1]).join(' '), edgesOf('in').map(e => (e.w >= 0 ? '+' : '') + e.w.toFixed(2)).join(' '));
  eq('inspector outgoing edges', (I.out || []).map(r => r[1]).join(' '), edgesOf('out').map(e => (e.w >= 0 ? '+' : '') + e.w.toFixed(2)).join(' '));
  // 6. message map (shade = summed influence of the pruned latents of each pair at the explained step)
  const sums = {}; pk.pruned.nodes.map(i => byId[i]).filter(n => n && n.type === 'feature' && n.t === P.mapStep).forEach(n => { const k = n.s + '>' + n.i; sums[k] = (sums[k] || 0) + pk.influence[n.id]; });
  eq('message map step', P.mapStep, P.t); eq('message map cells', P.map.length, N * N);
  P.map.forEach((txt, q) => { const s = Math.floor(q / N), r = q % N, real = s !== r && inst.adj[s][r] > 0; eq(`map ${s}->${r}`, txt, `${s} → ${r}: influence ${(sums[s + '>' + r] || 0).toFixed(4)}${real ? ' (real edge)' : ''}`); });
  // 7. the clamp: the real model re-run, against the graph's prediction
  const iv = E.intervene(MP, M, T, inst, ex, pr, P.pinned, 0), d3 = x => (x >= 0 ? '+' : '') + x.toFixed(3);
  eq('clamp rows', P.iv.length, iv.rows.length);
  iv.rows.forEach((r, k) => { const row = P.iv[k]; if (!row) return; eq(`clamp ${r.id} base`, row[1], r.base.toFixed(2)); eq(`clamp ${r.id} actual`, row[2], d3(r.actual - r.base)); eq(`clamp ${r.id} graph`, row[3], d3(r.predicted - r.base)); });
  const dir = iv.rows.filter(r => Math.abs(r.actual - r.base) > 1e-3).every(r => Math.sign(r.actual - r.base) === Math.sign(r.predicted - r.base));
  eq('clamp verdict', /Direction confirmed/.test(P.ivNote) ? 'confirmed' : 'not confirmed', dir ? 'confirmed' : 'not confirmed');
  eq('clamp decision counts', (/(\d[\d,]*) of ([\d,]+) downstream/.exec(P.ivNote) || []).slice(1).join('/'), `${iv.flips.toLocaleString()}/${iv.decisions.toLocaleString()}`);
}
console.log(bad ? `UI TRUTH TEST FAILED: ${bad} of ${checks} numbers differ from the recomputation` : `UI TRUTH TEST OK: all ${checks} numbers shown on the page match the recomputation and the original training files`);
proc.kill(); process.exit(bad ? 1 : 0);
