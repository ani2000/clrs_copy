/* Attribution-graph panel: a grid of graph nodes (columns) x algorithm steps (rows: features, then leaves),
 * following the layout of Anthropic-style attribution graphs. Graphs are computed by attrib.js in a dedicated
 * Web Worker from the real trained 128-hidden model; this file only requests, lays out and inspects them.
 * Relies on viewer globals: G, W, FL, curAlgo, curStep, selected, pruneMode, viewMode, minW, labelByFeat,
 * LOCAL, ALGO_CACHE, ensureLocalModel, loadContexts, fmt. */
const AV = { worker: null, ready: {}, waiters: {}, busy: false, pending: null, seq: 0, want: null, data: null,
             pinned: null, hovered: null, edgeEls: [], nodeEls: {}, gr: null, ivSeq: 0 };
const AVNS = 'http://www.w3.org/2000/svg';
const avEl = (tag, attrs, parent) => { const e = document.createElementNS(AVNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const avCss = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#999';
const avEsc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------- feature groups (from this project's own predicate labels) ----------
function avGroup(label) {
  const l = (label || '').toLowerCase();
  if (!l || l === '(unlabeled)' || l === '(rare)') return 'none';
  if (l.includes('source')) return 'source';
  if (l.startsWith('not ')) return 'neg';
  if (l.includes('class')) return 'class';
  return 'state';
}
const AV_GROUP = { state: ['--g-state', 'target state'], class: ['--g-class', 'class change'], source: ['--g-source', 'source'],
                   neg: ['--g-neg', 'negated state (“not …”)'], none: ['--g-none', 'unlabelled'],
                   pe: ['--g-state', 'latent on a real edge'], pn: ['--g-none', 'latent on a non-edge pair'], ps: ['--g-source', 'latent on a self pair'] };   // pe / pn / ps: edge-site latents, by what the pair is in this graph
const avEdge = () => !!(AV.data ? AV.data.mode === 'edge' : (typeof ATTR !== 'undefined' && ATTR.mode === 'edge'));
function avReset() {                         // the graph model changed: drop every trace of the old one at once
  AV.data = null; AV.gr = null; AV.pinned = null; AV.hovered = null; AV.edgeEls = []; AV.nodeEls = {}; AV.want = null;
  const g = id => document.getElementById(id); if (g('attribSvg')) g('attribSvg').innerHTML = ''; if (g('avLegend')) g('avLegend').innerHTML = '';
  if (g('featDetail')) g('featDetail').innerHTML = '<p class="empty">Hover a node in the attribution graph to see its edges; click to pin it here.</p>'; if (g('ivPanel')) g('ivPanel').innerHTML = '';
}
const avLabel = f => (labelByFeat[f] && labelByFeat[f].label) || '(unlabeled)';

// ---------- worker plumbing ----------
function avWorker() {
  if (AV.worker || typeof Worker === 'undefined') return AV.worker;
  AV.worker = new Worker('mpnn_worker.js' + VQ);
  AV.worker.onmessage = ev => {
    const m = ev.data;
    if (m.type === 'loaded' || m.type === 'tcloaded' || m.type === 'edgetcloaded') { const k = m.type + ':' + m.algo; AV.ready[k] = true; (AV.waiters[k] || []).forEach(f => f()); delete AV.waiters[k]; return; }
    if (m.type === 'attrib' || (m.type === 'error' && m.id && m.id.startsWith('a'))) {
      AV.busy = false;
      if (AV.want && m.id === AV.want.id) { if (m.type === 'error') { console.warn(m.error); avStatus('Could not trace this prediction just now. Click the node again to retry.'); } else avReceive(m); }
      avPump(); return;
    }
    if (m.type === 'intervened' || (m.type === 'error' && m.id && m.id.startsWith('i'))) avIvResult(m);
  };
  return AV.worker;
}
function avWait(key) { return AV.ready[key] ? Promise.resolve() : new Promise(res => (AV.waiters[key] = AV.waiters[key] || []).push(res)); }
async function avEnsure(algo) {
  const w = avWorker(); if (!w) throw new Error('Web Workers unavailable');
  ensureLocalModel(algo);
  const entry = LOCAL.models[algo];
  if (!entry) throw new Error('model bundle unavailable');
  const edge = ATTR.mode === 'edge';
  const wp = edge ? null : ensureWeights(algo), lp = edge ? null : ensureLabels(algo), ep = edge ? ensureEdgeTC(algo) : null;      // node-site: weights and labels; edge-site: its transcoder (1.3 MB)
  if (wp) wp.catch(() => {}); if (lp) lp.catch(() => {}); if (ep) ep.catch(() => {});
  await entry.promise;
  if (entry.state === 'aborted' || algo !== curAlgo) { const e = new Error('aborted'); e.name = 'AbortError'; throw e; }
  if (entry.stale) throw new StaleError('stale');
  if (entry.state !== 'ready' || !entry.man) throw new Error('model bundle failed to load');
  const Wfull = edge ? null : await wp, ET = edge ? await ep : null; if (lp) await lp;
  if (algo !== curAlgo) { const e = new Error('aborted'); e.name = 'AbortError'; throw e; }
  if (!AV.ready['loaded:' + algo] && !AV['sent:' + algo]) { AV['sent:' + algo] = true; w.postMessage({ type: 'load', algo, manifest: entry.man, buffer: entry.buf.slice(0) }); }
  if (edge) {
    if (!AV.ready['edgetcloaded:' + algo] && !AV['etcsent:' + algo]) { AV['etcsent:' + algo] = true; w.postMessage({ type: 'edgetc', algo, manifest: ET.man, buffer: ET.buf.slice(0) }); }
    await Promise.all([avWait('loaded:' + algo), avWait('edgetcloaded:' + algo)]); return;
  }
  if (!AV.ready['tcloaded:' + algo] && !AV['tcsent:' + algo]) { AV['tcsent:' + algo] = true; w.postMessage({ type: 'tc', algo, W: Wfull }); }
  await Promise.all([avWait('loaded:' + algo), avWait('tcloaded:' + algo)]);
}
function avPump() {
  if (AV.busy || !AV.pending) return;
  const req = AV.pending; AV.pending = null;
  if (!AV.want || req.id !== AV.want.id) return avPump();          // superseded before it was sent
  AV.busy = true; AV.worker.postMessage(req);
}
function avSameGraph() { return !!(AV.data && G && AV.data.key.startsWith([curAlgo, G.seed, G.n, G.length].join('|') + '|')); }
function avKey(step, node) { return [curAlgo, G.seed, G.n, G.length, step, node, ATTR.mode].join('|'); }
async function avRequest(step, node) {
  const key = avKey(step, node);
  const req = { type: ATTR.mode === 'edge' ? 'attribEdge' : 'attrib', id: 'a' + (++AV.seq), key, algo: curAlgo, t: step, i: node, main: W.target, window: 3 };
  AV.want = req;
  avStatus(AV.data ? 'tracing…' : (LOCAL.models[curAlgo] && LOCAL.models[curAlgo].state === 'ready' ? 'tracing the circuit…' : 'loading the model into your browser…'), true);
  try {
    await avEnsure(curAlgo);
    let inst = G._inst;
    if (!inst) { inst = await getInstance(curAlgo, G.n, +document.getElementById('inP').value, G.seed); G._inst = inst; }
    if (AV.want !== req) return;
    req.inst = inst; AV.pending = req; avPump();
  } catch (e) {
    if (e && e.name === 'AbortError') return;
    if (e instanceof StaleError) { showStale(); return; }
    console.warn('attribution', e && e.message);
    if (AV.want === req) { avStatus('Could not load what is needed for this view. Retrying\u2026'); setTimeout(() => { if (AV.want === req && G && !G._preview) { AV.want = null; drawAttribution(); } }, 2500); }
  }
}
function avStatus(text, busy) {
  const s = document.getElementById('gstat'); if (s) s.textContent = text;
  const svg = document.getElementById('attribSvg'); if (svg) svg.style.opacity = busy && AV.data ? '0.45' : '';
}

// ---------- receive & index ----------
function avReceive(m) {
  if (!G || !AV.want || !AV.want.key.startsWith([curAlgo, G.seed, G.n, G.length].join('|') + '|')) return;   // traced for a graph no longer shown
  const nodes = {}; m.nodes.forEach(nd => { nodes[nd.id] = nd; if (nd.type === 'feature') { if (m.mode === 'edge') { nd.label = `${nd.s}→${nd.i}`; nd.group = nd.ptype === 'edge' ? 'pe' : nd.ptype === 'self' ? 'ps' : 'pn'; } else { nd.label = avLabel(nd.f); nd.group = avGroup(nd.label); } } });
  AV.data = Object.assign(m, { nodesById: nodes, t: AV.want.t, i: AV.want.i, key: AV.want.key, mode: m.mode || 'node' });
  AV.pinned = null; AV.hovered = null; AV.scrolled = null;
  if (m.mode === 'edge') ensureLatents(curAlgo).catch(() => {});
  avRender();
  const top = m.pruned.nodes.filter(id => nodes[id] && nodes[id].type === 'feature').sort((a, b) => (m.influence[b] || 0) - (m.influence[a] || 0))[0];
  if (top) avInspect(nodes[top]);
  else avInspectEmpty();
  if (typeof drawGraph === 'function') { drawGraph(); }   // feature dots + real predictions in the simulation panel
}

// ---------- entry point (called by the viewer on every relevant change) ----------
function drawAttribution() {
  if (typeof renderEdgeNote === 'function') renderEdgeNote();          // the quality note follows the algorithm and graph size
  if (AV.data && (!avSameGraph() || (AV.data.mode === 'edge') !== (ATTR.mode === 'edge'))) {                 // a different algorithm or graph: drop every trace of the old one at once
    AV.data = null; AV.gr = null; AV.pinned = null; AV.hovered = null; AV.edgeEls = []; AV.nodeEls = {};
    document.getElementById('attribSvg').innerHTML = ''; document.getElementById('avLegend').innerHTML = '';
    document.getElementById('featDetail').innerHTML = '<p class="empty">Hover a node in the attribution graph to see its edges; click to pin it here.</p>';
    document.getElementById('ivPanel').innerHTML = '';
  }
  if (!G || !W || G._preview) return;
  if (!selected) avAutoSelect();
  if (!selected) return;
  const step = Math.min(curStep, G.length - 2);
  if (step < 0) { avStatus('This graph has no explainable step. Click “New random graph”.'); return; }
  selected = { node: selected.node, step };
  const key = avKey(step, selected.node);
  if (AV.data && AV.data.key === key) { avRender(); return; }
  avQuestionPending(step, selected.node);
  avRequest(step, selected.node);
}
function avAutoSelect() {           // the first node whose target hint changes, preferring a step with history
  const tt = G.target_true; if (!tt) { selected = { node: G.source !== null ? G.source : 0, step: 0 }; return; }
  let pick = null;
  for (let s = 1; s < G.length - 1 && !pick; s++) for (let v = 0; v < G.n; v++) if (JSON.stringify(tt[s][v]) !== JSON.stringify(tt[s + 1][v])) { pick = { node: v, step: s }; break; }
  if (!pick) for (let v = 0; v < G.n; v++) if (JSON.stringify(tt[0][v]) !== JSON.stringify(tt[1][v])) { pick = { node: v, step: 0 }; break; }
  pick = pick || { node: G.source !== null ? G.source : 0, step: Math.min(1, G.length - 2) };
  selected = pick; curStep = pick.step;
  const r = document.getElementById('stepRange'); if (r) r.value = curStep;
  if (typeof drawGraph === 'function') drawGraph();
}

// ---------- question & output chips ----------
function avTrue(step, i) { return G.target_true && G.target_true[step + 1] ? G.target_true[step + 1][i] : null; }
function avTargetText(tg, step) {
  const f = W.target, s1 = step + 1;
  if (tg.kind === 'pointer') return `${tg.field}[${s1}][${tg.i}] = ${tg.j}`;
  if (tg.ftype === 'mask') return `${f}[${s1}][${tg.i}] = ${tg.p1 > 0.5 ? 1 : 0}`;
  if (tg.ftype === 'categorical') return `${f}[${s1}][${tg.i}] = ${tg.c}`;
  if (tg.ftype === 'mask_one') return `${f}[${s1}] ${tg.p1 > 0.5 ? '=' : '≠'} node ${tg.i}`;
  return `${f}[${s1}][${tg.i}] = ${Number(tg.value).toFixed(2)}`;
}
function avChipText(tg) {
  if (tg.kind === 'pointer') return `${tg.i}→${tg.j} ${tg.prob.toFixed(2)}`;
  if (tg.ftype === 'mask' || tg.ftype === 'mask_one') return `${W.target}[${tg.i}] ${tg.p1.toFixed(2)}`;
  if (tg.ftype === 'categorical') return `${W.target}[${tg.i}]=${tg.c} ${tg.prob.toFixed(2)}`;
  return `${W.target}[${tg.i}] ${Number(tg.value).toFixed(2)}`;
}
function avQuestionPending(step, i) {
  document.getElementById('question').innerHTML = `Step ${step}: why does the model predict <b>${avEsc(W.target)}[${step + 1}][${i}]</b> for node ${i}?`;
  document.getElementById('outprobs').innerHTML = '';
}
function avQuestion() {
  const d = AV.data, step = d.t, i = d.i, main = d.targets.find(x => x.kind === 'hint'), ptrs = d.targets.filter(x => x.kind === 'pointer');
  const top = ptrs.find(x => x.always), truth = avTrue(step, i);
  const pOf = tg => tg.ftype === 'scalar' ? '' : ` (p = ${(tg.ftype === 'mask' || tg.ftype === 'mask_one' ? (tg.p1 > 0.5 ? tg.p1 : 1 - tg.p1) : tg.prob).toFixed(3)})`;
  const truthTxt = truth === null ? '' : ` Truth: ${avEsc(W.target)}[${step + 1}][${i}] = ${W.target_type === 'scalar' ? Number(truth).toFixed(2) : Math.round(truth)}.`;
  document.getElementById('question').innerHTML = `Step ${step}: why does the model predict ` + (top ? `<b>${avEsc(avTargetText(top, step))}</b>, and ` : '') +
    `<b>${avEsc(avTargetText(main, step))}</b>${pOf(main)}?<span class="qtruth">${truthTxt}</span>`;
  const chips = [...ptrs.filter(x => x.prob >= 0.01 || x.always), main];
  document.getElementById('outprobs').innerHTML = chips.map(tg => { const p = tg.ftype === 'scalar' ? 1 : (tg.kind === 'pointer' || tg.ftype === 'categorical' ? tg.prob : tg.p1);
    return `<span class="p ${tg.always ? 'true' : ''}" data-tip="an:${avEsc(tg.id)}">${avEsc(tg.kind === 'pointer' ? `${tg.field}[${tg.i}]→${tg.j}` : avTargetText(tg, step))} <i style="width:${Math.max(2, Math.round(p * 50))}px"></i> ${tg.ftype === 'scalar' ? '' : p.toFixed(3)}</span>`; }).join('');
}

// ---------- current view: pruned / all, by node / supernodes ----------
function avCurrent() {
  const d = AV.data, N = d.nodesById;
  let nodes, edges;
  if (pruneMode === 'pruned') { nodes = d.pruned.nodes.map(id => N[id]).filter(Boolean); edges = d.pruned.edges; }
  else { edges = d.top; const ids = new Set(); edges.forEach(e => { ids.add(e.s); ids.add(e.d); }); d.targets.forEach(tg => ids.add(tg.id)); nodes = [...ids].map(id => N[id]).filter(Boolean); }
  nodes = nodes.filter(nd => nd.type !== 'logit' || nd.prob >= 0.01 || nd.always);
  edges = edges.filter(e => Math.abs(e.w) >= minW);
  if (viewMode === 'supernodes' || viewMode === 'meaning') {
    const flab = nd => (viewMode === 'meaning' && d.mode === 'edge') ? avMeaningOf(nd) : nd.label;                 // 'By meaning' groups latents by what they were found to mean
    const key = nd => nd.type === 'feature' ? `S:F${nd.t}:${flab(nd)}` : nd.type === 'leaf' ? `S:L${nd.t}:${nd.probe}` : nd.type === 'error' ? `S:E${nd.t}` : nd.id;
    const sup = {}, memberOf = {};
    nodes.forEach(nd => { const k = key(nd); memberOf[nd.id] = k;
      if (!sup[k]) sup[k] = Object.assign({}, nd, { id: k, members: [], act: 0, label: nd.type === 'feature' ? flab(nd) : nd.type === 'leaf' ? nd.probe : nd.type === 'error' ? 'error' : nd.label });
      sup[k].members.push(nd.id); sup[k].act += (nd.act || 0); });
    const agg = {}; edges.forEach(e => { const s = memberOf[e.s], dd = memberOf[e.d]; if (!s || !dd || s === dd) return; agg[s + '\u0001' + dd] = (agg[s + '\u0001' + dd] || 0) + e.w; });
    return { nodes: Object.values(sup), edges: Object.entries(agg).map(([k, w]) => { const [s, dd] = k.split('\u0001'); return { s, d: dd, w }; }), super: true };
  }
  const present = new Set(nodes.map(nd => nd.id));
  return { nodes, edges: edges.filter(e => present.has(e.s) && present.has(e.d)), super: false };
}

// ---------- layout: columns = bias + graph nodes (or labels), rows = predictions, then features / leaves per step ----------
function avPlace(Gr) {
  const d = AV.data, n = G.n, t = d.t;
  const rowKey = nd => nd.type === 'logit' || (nd.type === 'bias' && nd.t > t) ? 'O' : nd.type === 'leaf' ? 'L' + nd.t : 'F' + nd.t;
  const rows = ['O']; for (let tt = t; tt >= d.tLo; tt--) { rows.push('F' + tt); rows.push('L' + tt); }
  const left = 128;
  let cols;
  if (!Gr.super) { cols = ['bias']; for (let i = 0; i < n; i++) cols.push(String(i)); }
  else { const labs = [...new Set(Gr.nodes.filter(nd => nd.type === 'feature').map(nd => nd.label))].sort();
         const probes = [...new Set(Gr.nodes.filter(nd => nd.type === 'leaf').map(nd => nd.label))].sort(); cols = ['bias', 'error', ...probes, ...labs]; }
  const host = document.getElementById('attribScroll');
  const avail = (host ? host.clientWidth : 800) - left - 16;
  const colW = Math.max(Gr.super ? 92 : 64, Math.min(108, Math.floor(avail / cols.length)));
  const colX = c => left + (cols.indexOf(c) + 0.5) * colW;
  const colOf = nd => {
    if (Gr.super) return nd.type === 'feature' || nd.type === 'leaf' ? nd.label : nd.type === 'error' ? 'error' : 'bias';
    if (nd.type === 'bias') return 'bias';
    if (nd.type === 'logit') return String(nd.kind === 'pointer' ? nd.j : nd.i);
    if (nd.type === 'leaf' && (nd.i === null || nd.i === undefined)) return 'bias';
    return String(nd.i);
  };
  const cells = {};
  Gr.nodes.forEach(nd => { const k = rowKey(nd) + '|' + colOf(nd); (cells[k] = cells[k] || []).push(nd); });
  const perRow = Math.max(1, Math.floor((colW - 12) / 14));
  const need = {}; rows.forEach(r => need[r] = 1);
  Object.entries(cells).forEach(([k, list]) => { const r = k.split('|')[0]; const m = list.filter(x => x.type !== 'error').length; need[r] = Math.max(need[r], m <= 5 ? 1 : Math.ceil(m / perRow)); });
  const rowH = {}; rows.forEach(r => rowH[r] = r === 'O' ? 64 : r[0] === 'F' ? Math.max(76, 48 + need[r] * 16) : Math.max(58, 30 + need[r] * 16));
  let y = 36; const rowY = {}, rowTop = {}; rows.forEach(r => { rowTop[r] = y; rowY[r] = y + (r === 'O' ? rowH[r] / 2 : 26); y += rowH[r]; });
  const pos = {};
  Object.entries(cells).forEach(([k, list]) => {
    const [r, c] = k.split('|');
    const errs = list.filter(x => x.type === 'error'), rest = list.filter(x => x.type !== 'error');
    rest.sort((a, b) => a.type === b.type ? ((b.act || 0) - (a.act || 0) || String(a.probe || '').localeCompare(String(b.probe || '')) || (a.j ?? -1) - (b.j ?? -1)) : a.type.localeCompare(b.type));
    if (Gr.super && r === 'O') { rest.forEach((nd, q) => { pos[nd.id] = { x: left + 60 + q * 120, y: rowY[r] }; }); }
    else {
      const m = rest.length, span = Math.min(colW - 14, m * 20), dx = m > 1 ? span / (m - 1) : 0;
      rest.forEach((nd, q) => { let x, yy;
        if (m <= 5) { x = colX(c) - span / 2 + q * dx; yy = rowY[r]; }
        else { const rr = Math.floor(q / perRow), cc = q % perRow, cnt = Math.min(perRow, m - rr * perRow); x = colX(c) - (cnt - 1) * 7 + cc * 14; yy = rowY[r] - 8 + rr * 16; }
        pos[nd.id] = { x, y: yy }; nd._q = q; nd._m = m; });
    }
    errs.forEach(nd => { pos[nd.id] = { x: colX(c) - colW / 2 + 9, y: rowY[r] + 24 }; });
  });
  const chips = Gr.nodes.filter(nd => nd.type === 'logit' && pos[nd.id]).sort((a, b) => pos[a.id].x - pos[b.id].x);
  const cw = nd => Math.max(64, avChipText(nd).length * 6.6 + 16);
  for (let k = 1; k < chips.length; k++) { const a = pos[chips[k - 1].id], b = pos[chips[k].id], gap = cw(chips[k - 1]) / 2 + cw(chips[k]) / 2 + 6; if (b.x - a.x < gap) b.x = a.x + gap; }
  let right = left + cols.length * colW + 20;
  if (chips.length) { const last = chips[chips.length - 1]; right = Math.max(right, pos[last.id].x + cw(last) / 2 + 12); }
  return { pos, rows, rowY, rowTop, rowH, cols, colX, width: right, height: y + 12, colW };
}

// ---------- draw ----------
function avNodeColor(nd) {
  if (nd.type === 'feature') return avCss(AV_GROUP[nd.group || avGroup(nd.label)][0]);
  if (nd.type === 'leaf') return avCss('--leaf');
  if (nd.type === 'logit') return avCss('--logit');
  return avCss('--ink-3');
}
function avNodeText(nd, Gr) {
  if (nd.type === 'feature') return Gr.super ? `${nd.label} (${nd.members.length})` : (AV.data && AV.data.mode === 'edge' ? `${nd.s}:f${nd.f}` : `f${nd.f}`);
  if (nd.type === 'leaf') { if (Gr.super) return `${nd.label} (${nd.members.length})`;
    if (nd.probe === 'state') return `h[${nd.i}]`; return nd.i === null || nd.i === undefined ? nd.probe : (nd.j === null || nd.j === undefined ? `${nd.probe}[${nd.i}]` : `${nd.probe}[${nd.i},${nd.j}]`); }
  if (nd.type === 'logit') return avChipText(nd);
  if (nd.type === 'error') return Gr.super ? `err (${nd.members.length})` : (AV.data && AV.data.mode === 'edge' ? `err ${nd.s}→${nd.i}` : 'err');
  return 'bias';
}
function avRender() {
  if (!AV.data) return;
  const svg = document.getElementById('attribSvg');
  svg.style.opacity = '';
  avQuestion();
  const Gr = avCurrent(), L = avPlace(Gr); AV.gr = Gr; AV.layout = L;
  svg.innerHTML = ''; AV.edgeEls = []; AV.nodeEls = {}; AV.hovered = null;
  svg.setAttribute('width', L.width); svg.setAttribute('height', L.height); svg.setAttribute('viewBox', `0 0 ${L.width} ${L.height}`);
  svg.style.width = L.width + 'px'; svg.style.height = L.height + 'px';
  const defs = avEl('defs', {}, svg), pat = avEl('pattern', { id: 'avhatch', width: 4, height: 4, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
  avEl('rect', { width: 4, height: 4, fill: avCss('--surface') }, pat); avEl('rect', { width: 1.5, height: 4, fill: avCss('--err') }, pat);
  L.rows.forEach((r, k) => {
    if (k % 2 === 1) avEl('rect', { x: 0, y: L.rowTop[r], width: L.width, height: L.rowH[r], fill: avCss('--surface-2'), opacity: .5 }, svg);
    const lab = r === 'O' ? `PREDICTIONS OF STEP ${AV.data.t}` : (r[0] === 'F' ? `STEP ${r.slice(1)} · FEATURES` : `STEP ${r.slice(1)} · LEAVES`);
    const tx = avEl('text', { x: 8, y: L.rowY[r] + 3, class: 'rowlab' }, svg); tx.textContent = lab;
    const tip = r === 'O' ? 'g:avRowO' : r[0] === 'F' ? 'g:avRowF' : 'g:avRowL'; tx.setAttribute('data-tip', tip);
  });
  L.cols.forEach(c => { const tx = avEl('text', { x: L.colX(c), y: 22, class: 'collab', 'text-anchor': 'middle' }, svg);
    tx.textContent = Gr.super ? c : (c === 'bias' ? 'bias' : (AV.data.mode === 'edge' ? `→ ${c}${+c === G.source ? ' src' : ''}${+c === AV.data.i ? ' ★' : ''}` : `node ${c}${+c === G.source ? ' (src)' : ''}${+c === AV.data.i ? ' ★' : ''}`));
    if (!Gr.super && c !== 'bias') tx.setAttribute('data-tip', 'node:' + c); });
  const wmax = Math.max(1e-6, ...Gr.edges.map(e => Math.abs(e.w)));
  const eg = avEl('g', {}, svg), hitg = avEl('g', {}, svg);
  const hits = Gr.edges.length <= 700;
  Gr.edges.forEach((e, idx) => {
    const a = L.pos[e.s], b = L.pos[e.d]; if (!a || !b) return;
    const sw = 0.5 + 3.5 * Math.sqrt(Math.abs(e.w) / wmax), dy = b.y - a.y, fromBias = String(e.s).startsWith('B');
    const dstr = `M${a.x},${a.y} C${a.x},${a.y + dy * 0.5} ${b.x},${b.y - dy * 0.5} ${b.x},${b.y}`;
    const p = avEl('path', { d: dstr, class: 'edge', stroke: e.w >= 0 ? avCss('--pos') : avCss('--neg'), 'stroke-width': sw.toFixed(2), opacity: fromBias ? .15 : .42 }, eg);
    p._e = e; AV.edgeEls.push(p);
    if (hits) avEl('path', { d: dstr, fill: 'none', stroke: 'transparent', 'stroke-width': 8, 'data-tip': 'ae:' + idx, class: 'ehit' }, hitg);
  });
  const ng = avEl('g', {}, svg);
  Gr.nodes.forEach(nd => {
    const p = L.pos[nd.id]; if (!p) return;
    const g = avEl('g', { class: 'nd', transform: `translate(${p.x},${p.y})`, tabindex: 0, role: 'button', 'data-tip': 'an:' + nd.id, 'aria-label': avNodeText(nd, Gr) }, ng);
    const col = avNodeColor(nd);
    if (nd.type === 'feature') { const r = Gr.super ? Math.min(14, 5 + 2.5 * Math.sqrt(nd.members.length)) : Math.min(11, 4 + 2.2 * Math.sqrt(nd.act)); avEl('circle', { r, fill: col, stroke: avCss('--surface'), 'stroke-width': 1.5 }, g); }
    else if (nd.type === 'leaf') { const s = Gr.super ? Math.min(14, 8 + 1.5 * Math.sqrt(nd.members.length)) : 8; avEl('rect', { x: -s / 2, y: -s / 2, width: s, height: s, rx: 1.5, fill: nd.probe === 'state' ? avCss('--g-none') : col }, g); }
    else if (nd.type === 'error') avEl('rect', { x: -4.5, y: -4.5, width: 9, height: 9, rx: 1, fill: 'url(#avhatch)', stroke: avCss('--err'), 'stroke-width': 1 }, g);
    else if (nd.type === 'bias') avEl('rect', { x: -4.5, y: -4.5, width: 9, height: 9, fill: col, transform: 'rotate(45)' }, g);
    else { const txt = avChipText(nd), w = Math.max(64, txt.length * 6.6 + 16), fill = nd.always ? avCss('--logit') : avCss('--surface');
      avEl('rect', { x: -w / 2, y: -11, width: w, height: 22, rx: 11, fill, stroke: avCss('--logit'), 'stroke-width': 1.2 }, g);
      const tx = avEl('text', { y: 3.5, 'text-anchor': 'middle', fill: nd.always ? avCss('--surface') : avCss('--logit'), 'font-size': '10.5px' }, g); tx.textContent = txt; }
    if (nd.type !== 'logit' && (Gr.super || (nd.type === 'feature' && (AV.data.mode !== 'edge' || (nd._m || 1) <= 4)) || (nd.type === 'leaf' && pruneMode === 'pruned' && nd._m <= 4))) {   // edge-site cells can hold many latents: their labels would overlap, so they show on hover instead
      const yoff = nd.type === 'feature' ? (Gr.super ? Math.min(14, 5 + 2.5 * Math.sqrt(nd.members.length)) : Math.min(11, 4 + 2.2 * Math.sqrt(nd.act))) + 10 + ((nd._m || 1) > 2 ? ((nd._q || 0) % 2) * 9 : 0) : nd.type === 'error' ? 15 : 15 + ((nd._q || 0) % 3) * 9;
      const tx = avEl('text', { y: yoff, 'text-anchor': 'middle', class: 'nlab' }, g); tx.textContent = avNodeText(nd, Gr);
    }
    g.addEventListener('mouseenter', () => { AV.hovered = nd.id; avHighlight(); });
    g.addEventListener('mouseleave', () => { AV.hovered = null; avHighlight(); });
    g.addEventListener('click', ev => { ev.stopPropagation(); AV.pinned = AV.pinned === nd.id ? null : nd.id; avHighlight(); if (AV.pinned) avInspect(nd, Gr); });
    g.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); g.dispatchEvent(new Event('click')); } });
    AV.nodeEls[nd.id] = g;
  });
  avHighlight(); avLegend(Gr); avStat(Gr);
  if (AV.data.mode === 'edge') { renderTopLatents(); renderMessageMap(); renderDictionary(); }
  if (AV.data.mode === 'edge' && !Gr.super && !AV.scrolled) {          // edge-site graphs are wide: bring the explained node's column into view the first time
    const host = document.getElementById('attribScroll'), c = String(AV.data.i); if (host && L.cols.includes(c)) host.scrollLeft = Math.max(0, L.colX(c) - host.clientWidth / 2); AV.scrolled = AV.data.key;
  }
  if (window.HoverCard) HoverCard.refresh();
}
function avHighlight() {
  if (AV.hovered && !AV.nodeEls[AV.hovered]) AV.hovered = null;          // node vanished in a re-render
  if (AV.pinned && !AV.nodeEls[AV.pinned]) AV.pinned = null;
  const focus = AV.hovered || AV.pinned;
  if (!focus) { AV.edgeEls.forEach(p => p.classList.remove('dim', 'hi')); Object.values(AV.nodeEls).forEach(g => g.classList.remove('dim')); return; }
  const keep = new Set([focus]);
  AV.edgeEls.forEach(p => { const on = p._e.s === focus || p._e.d === focus; p.classList.toggle('hi', on); p.classList.toggle('dim', !on); if (on) { keep.add(p._e.s); keep.add(p._e.d); } });
  Object.entries(AV.nodeEls).forEach(([id, g]) => g.classList.toggle('dim', !keep.has(id)));
}
function avLegend(Gr) {
  const groups = new Set(Gr.nodes.filter(nd => nd.type === 'feature').map(nd => nd.group || avGroup(nd.label)));
  const g = ['state', 'class', 'source', 'neg', 'none', 'pe', 'pn', 'ps'].filter(k => groups.has(k)).map(k => `<span data-tip="g:avg_${k}"><i class="sw" style="background:var(${AV_GROUP[k][0]})"></i>${AV_GROUP[k][1]}</span>`).join('');
  document.getElementById('avLegend').innerHTML = g +
    `<span data-tip="g:legendLeaf"><i class="sw sq" style="background:var(--leaf)"></i>encoder leaf</span>` +
    `<span data-tip="g:avError"><i class="sw sq" style="background:repeating-linear-gradient(45deg,transparent 0 2px,var(--err) 2px 3px)"></i>error</span>` +
    `<span data-tip="g:bias"><i class="sw dia" style="background:var(--ink-3)"></i>bias</span>` +
    `<span data-tip="g:legendPos"><i class="sw ln" style="background:var(--pos)"></i>positive</span>` +
    `<span data-tip="g:legendNeg"><i class="sw ln" style="background:var(--neg)"></i>negative</span>`;
}
function avStat(Gr) {
  const d = AV.data;
  const shown = pruneMode === 'pruned' ? '' : ` Showing the ${d.top.length.toLocaleString()} strongest of ${d.eAll.toLocaleString()} edges${minW > 0 ? `, |w| ≥ ${minW.toFixed(2)}` : ''}.`;
  const ek = d.mode === 'edge', ck = d.checks;
  document.getElementById('gstat').textContent = ek ? `Attribution graph (edge-site transcoder): ${d.nAll.toLocaleString()} nodes / ${d.eAll.toLocaleString()} edges at steps ${d.tLo}–${d.t}; pruned to ${d.pruned.nodes.length} / ${d.pruned.edges.length}, keeping ${(d.keptInfluence * 100).toFixed(0)}% of the influence on the targets.${shown} Error nodes supply ${((d.errorShare || 0) * 100).toFixed(d.errorShare < 0.1 ? 1 : 0)}% of the absolute weight arriving at the predictions (the part the latents do not explain). Every latent's pre-activation and every target logit equals the sum of its incoming edges plus a bias computed from the constant terms (max-routing winners, LayerNorm σ, the node update's ReLU pattern and the pointer max frozen). Checked on this graph: ${ck.n.toLocaleString()} nodes, largest difference ${ck.maxAbs.toExponential(1)}. Edge weights and total effects were checked against finite differences in the test suite (verification/clrs-edge). Traced in your browser in ${d.ms} ms.` : `Attribution graph: ${d.nAll.toLocaleString()} nodes / ${d.eAll.toLocaleString()} edges at steps ${d.tLo}–${d.t}; pruned to ${d.pruned.nodes.length} / ${d.pruned.edges.length}, keeping ${(d.keptInfluence * 100).toFixed(0)}% of the influence on the targets.${shown} Every feature's pre-activation and every target logit equals the sum of its incoming edges (ReLU masks, max-routing, top-k and pointer max frozen; edge weights verified against finite differences). Traced in your browser in ${d.ms} ms.`;
}


// ---------- what the edge-site latents mean (edge_tc/<algo>.latents.json, measured by verification/clrs-edge/clrs_edge_latents.js) ----------
// Per latent: how often it fires and on which pairs, its effect on the receiver's main target logit, and its best-matching conditions scored on held-out graphs.
const LATS = {}, LAT_REQ = {};
function ensureLatents(a) {
  if (LATS[a]) return Promise.resolve(LATS[a]);
  if (!LAT_REQ[a]) LAT_REQ[a] = withRetry(() => getJSON('/edge_tc/' + a + '.latents.json?v=' + BUILD)).then(j => { LATS[a] = j; delete LAT_REQ[a]; if (a === curAlgo) avLatentsReady(); return j; }, e => { delete LAT_REQ[a]; throw e; });
  return LAT_REQ[a];
}
const avJ = () => (typeof LATS !== 'undefined' && LATS[curAlgo]) || null;
const avMeaning = f => { const J = avJ(); return J && J.latents[f] ? J.latents[f] : null; };
const avShortName = (J, v) => v && v.tier !== '-' ? J.conditions[v.best[0][0]].id.replace(/_/g, ' ') : null;
function avMeaningOf(nd) { const J = avJ(); if (!J) return 'loading…'; return avShortName(J, J.latents[nd.f]) || 'unnamed'; }
const pct0 = x => (x * 100).toFixed(0) + '%';
function avEffectText(J, v) {                 // effect on the receiver's main target logit per unit of activation (the graph's own edge weight, averaged over the activations measured)
  if (!v || !v.eff) return 'not measured (it never fired at a step that was used for effects)';
  const [m, sd, n, pos] = v.eff, word = pos >= 0.9 ? 'raises' : pos >= 0.65 ? 'mostly raises' : pos <= 0.1 ? 'lowers' : pos <= 0.35 ? 'mostly lowers' : 'has a mixed effect on';
  return `${word} the ${avEsc(J.effect_of || 'receiver’s ' + J.target + ' logit')}: ${m >= 0 ? '+' : ''}${m.toFixed(3)} per unit of activation on average (sd ${sd.toFixed(3)}, positive in ${pct0(pos)} of ${n.toLocaleString()} activations)`;
}
function avMeaningHtml(nd) {                  // the "what does this latent mean" block of the inspector
  const J = avJ(); if (!J) return '<p class="note">Loading what this latent means…</p>';
  const v = J.latents[nd.f];
  if (!v) return `<p class="note">This latent never fired in the ${J.graphs} measurement graphs, so nothing is known about what it does.</p>`;
  const base = c => (J.conditions.find(x => x.id === c) || {}).base_rate_held, e = v.kind;
  let h = '';
  if (v.tier !== '-') { const b = v.best[0]; h += `<p class="note"><b>Meaning:</b> ${avEsc(J.conditions[b[0]].text)}. <span class="mono">tier ${v.tier}</span>: held-out precision ${b[1].toFixed(2)}, recall ${b[2].toFixed(2)}, F1 ${b[3].toFixed(2)}.</p>`; }
  else h += `<p class="note"><b>Unnamed:</b> no simple condition matches this latent well (best held-out F1 ${v.best.length ? v.best[0][3].toFixed(2) : '0.00'}). The facts below are all that is known about it.</p>`;
  h += `<dl class="ikv"><dt>effect</dt><dd>${avEffectText(J, v)}</dd><dt>fires on</dt><dd>${(v.rate * 100).toFixed(1)}% of pairs (mean activation ${v.act}, max ${v.max})</dd>
    <dt>on which pairs</dt><dd>${pct0(e[0])} real edges (${pct0(base('edge') || 0)} of all pairs), ${pct0(e[1])} non-edges (${pct0(base('nonedge') || 0)}), ${pct0(e[2])} self pairs (${pct0(base('self') || 0)})</dd></dl>`;
  if (v.best.length) h += `<div class="isec"><h3>Best-matching conditions (held-out graphs)</h3><table class="ivtab condtab"><tr><th>condition</th><th>P</th><th>R</th><th>F1</th></tr>${v.best.map(b => `<tr><td>${avEsc(J.conditions[b[0]].text)}</td><td>${b[1].toFixed(2)}</td><td>${b[2].toFixed(2)}</td><td>${b[3].toFixed(2)}</td></tr>`).join('')}</table></div>`;
  if (v.ex && v.ex.length) h += `<div class="isec"><h3>Strongest activations (sampler n = 16, p = 0.4)</h3>${v.ex.map(x => `<div class="small"><span class="mono">seed ${x[0]}, step ${x[1]}, ${x[2]} → ${x[3]}</span> activation ${x[4]} <button class="avex" data-seed="${x[0]}" data-t="${x[1]}" data-r="${x[3]}">show</button></div>`).join('')}</div>`;
  h += `<p class="small" style="color:var(--muted)">Measured on ${J.graphs} graphs (${J.rows_fit + J.rows_held} pair rows${J.max_steps_per_graph ? ', up to ' + J.max_steps_per_graph + ' evenly spaced steps per graph' : ''}): conditions chosen on ${J.graphs_fit} graphs, scored on the other ${J.graphs - J.graphs_fit}.</p>`;
  return h;
}
async function avOpenExample(seed, t, r) {      // load one of the measured graphs and explain the receiver at that step
  document.getElementById('inN').value = 16; document.getElementById('inP').value = 0.4; document.getElementById('inSeed').value = seed;
  await loadGraph(seed); if (!G || G._preview) return;
  curStep = Math.min(t, G.length - 2); const rg = document.getElementById('stepRange'); if (rg) rg.value = curStep;
  selected = { node: r, step: curStep }; drawGraph(); drawAttribution();
}
document.addEventListener('click', ev => { const b = ev.target.closest && ev.target.closest('button.avex'); if (b) avOpenExample(+b.dataset.seed, +b.dataset.t, +b.dataset.r); });
function avLatentsReady() { renderDictionary(); if (!AV.data || AV.data.mode !== 'edge') return; avRender(); if (AV.pinned && AV.data.nodesById[AV.pinned]) avInspect(AV.data.nodesById[AV.pinned], AV.gr); }

// ---------- panels that summarise the current graph (edge-site) ----------
function avInfl(id) { return AV.data.influence[id] || 0; }
function renderTopLatents() {                 // top latents of the pruned graph at every step of the window, with what they mean
  const box = document.getElementById('topLatents'); if (!box) return;
  const d = AV.data; if (!d || d.mode !== 'edge') { box.innerHTML = '<p class="small">Click a node in the simulation to trace a prediction.</p>'; return; }
  const J = avJ(); let h = '';
  for (let ts = d.t; ts >= d.tLo; ts--) {
    const feats = d.pruned.nodes.map(id => d.nodesById[id]).filter(nd => nd && nd.type === 'feature' && nd.t === ts).sort((a, b) => avInfl(b.id) - avInfl(a.id)).slice(0, 6);
    h += `<h3>Step ${ts}${ts === d.t ? ' (the step being explained)' : ' (feeds the next step)'}</h3>`;
    if (!feats.length) { h += '<p class="small">No latent of this step is in the pruned graph.</p>'; continue; }
    h += `<table class="toptab"><tr><th>latent</th><th>pair</th><th>activation</th><th>influence</th><th>meaning</th><th>effect on the receiver</th></tr>` + feats.map(nd => {
      const v = J && J.latents[nd.f], nm = J ? (avShortName(J, v) || 'unnamed') : '…', ef = v && v.eff ? `${v.eff[0] >= 0 ? '+' : ''}${v.eff[0].toFixed(3)}${v.eff[3] >= 0.9 ? ' (raises)' : v.eff[3] >= 0.65 ? ' (mostly raises)' : v.eff[3] <= 0.1 ? ' (lowers)' : v.eff[3] <= 0.35 ? ' (mostly lowers)' : ' (mixed)'}` : '-';
      return `<tr data-id="${avEsc(nd.id)}" tabindex="0"><td>f${nd.f}</td><td>${nd.s} → ${nd.i}</td><td>${nd.act.toFixed(2)}</td><td>${avInfl(nd.id).toFixed(4)}</td><td>${v && v.tier !== '-' ? `<b>${avEsc(nm)}</b> <span class="mono">${v.tier}</span>` : '<span style="color:var(--muted)">unnamed</span>'}</td><td>${ef}</td></tr>`; }).join('') + '</table>';
  }
  box.innerHTML = h;
}
document.addEventListener('click', ev => { const tr = ev.target.closest && ev.target.closest('#topLatents tr[data-id]'); if (!tr || !AV.data) return; const nd = AV.data.nodesById[tr.dataset.id]; if (!nd || !AV.nodeEls[nd.id]) return; AV.pinned = nd.id; avHighlight(); avInspect(nd, AV.gr); });
function renderDictionary() {                 // the best-named latents of the algorithm, with the measured evidence
  const box = document.getElementById('latDict'); if (!box) return; const J = avJ();
  if (!J) { box.innerHTML = '<p class="small">Loading…</p>'; return; }
  const rows = Object.entries(J.latents).filter(([f, v]) => v.tier !== '-').sort((a, b) => b[1].best[0][3] - a[1].best[0][3] || b[1].n - a[1].n).slice(0, 20);
  box.innerHTML = `<p class="small">${J.fired} of ${J.fired + J.never_fired} latents fired in ${J.graphs} graphs (${J.rows_fit + J.rows_held} pair rows); <b>${J.named}</b> can be named (tier A: ${J.tier_a}); ${J.never_fired} never fired. The rest have no simple description that holds on held-out graphs.</p>` +
    (rows.length ? `<div class="tscroll"><table class="toptab"><tr><th>latent</th><th>meaning</th><th>tier</th><th>P</th><th>R</th><th>F1</th><th>fires on</th><th>effect on the receiver</th></tr>${rows.map(([f, v]) => { const b = v.best[0];
      return `<tr><td>f${f}</td><td>${avEsc(J.conditions[b[0]].text)}</td><td>${v.tier}</td><td>${b[1].toFixed(2)}</td><td>${b[2].toFixed(2)}</td><td>${b[3].toFixed(2)}</td><td>${(v.rate * 100).toFixed(1)}%</td><td>${v.eff ? (v.eff[0] >= 0 ? '+' : '') + v.eff[0].toFixed(3) : '-'}</td></tr>`; }).join('')}</table></div>` : '<p class="small">No latent reached a held-out F1 of 0.5 for any condition.</p>');
}
let MSG_STEP = null;
function renderMessageMap() {                 // sender x receiver: how much influence the pruned latents of each message carry at one step
  const box = document.getElementById('msgMap'); if (!box) return; const d = AV.data;
  if (!d || d.mode !== 'edge') { box.innerHTML = ''; return; }
  if (MSG_STEP === null || MSG_STEP < d.tLo || MSG_STEP > d.t || AV.msgKey !== d.key) { MSG_STEP = d.t; AV.msgKey = d.key; }
  const N = G.n, cell = Math.max(8, Math.min(16, Math.floor(260 / N))), m = Array.from({ length: N }, () => new Float64Array(N)), best = {};
  d.pruned.nodes.map(id => d.nodesById[id]).filter(nd => nd && nd.type === 'feature' && nd.t === MSG_STEP).forEach(nd => { const v = avInfl(nd.id); m[nd.s][nd.i] += v; const k = nd.s + ',' + nd.i; if (!best[k] || v > avInfl(best[k])) best[k] = nd.id; });
  let mx = 1e-12; for (const r of m) for (const v of r) if (v > mx) mx = v;
  const o = 22; let svg = `<svg width="${o + N * cell + 4}" height="${o + N * cell + 4}" role="img" aria-label="influence of each sender to receiver message">`;
  for (let i = 0; i < N; i++) { svg += `<text x="${o + i * cell + cell / 2}" y="${o - 8}" font-size="8" text-anchor="middle" fill="#6b7280">${i}</text><text x="${o - 4}" y="${o + i * cell + cell * 0.7}" font-size="8" text-anchor="end" fill="#6b7280">${i}</text>`; }
  for (let s2 = 0; s2 < N; s2++) for (let r = 0; r < N; r++) { const v = m[s2][r], real = s2 !== r && G.adj[s2][r] > 0, a = v > 0 ? 0.12 + 0.88 * Math.sqrt(v / mx) : 0;
    svg += `<rect x="${o + r * cell}" y="${o + s2 * cell}" width="${cell - 1}" height="${cell - 1}" fill="${v > 0 ? `rgba(37,99,235,${a.toFixed(2)})` : '#f3f4f6'}" ${real ? 'stroke="#111827" stroke-width="0.6"' : ''} ${best[s2 + ',' + r] ? `data-id="${avEsc(best[s2 + ',' + r])}" style="cursor:pointer"` : ''}><title>${s2} → ${r}: influence ${v.toFixed(4)}${real ? ' (real edge)' : ''}</title></rect>`; }
  svg += '</svg>';
  const tabs = []; for (let ts = d.t; ts >= d.tLo; ts--) tabs.push(`<button data-t="${ts}" class="${ts === MSG_STEP ? 'active' : ''}">step ${ts}</button>`);
  box.innerHTML = `<h3>Message map <span class="small" style="font-weight:400">rows = senders, columns = receivers; shade = influence of the pruned latents on that message; outlined = real edge</span></h3><div class="seg msgtabs">${tabs.join('')}</div>${svg}`;
}
document.addEventListener('click', ev => {
  const b = ev.target.closest && ev.target.closest('#msgMap .msgtabs button'); if (b) { MSG_STEP = +b.dataset.t; renderMessageMap(); return; }
  const r = ev.target.closest && ev.target.closest('#msgMap rect[data-id]'); if (r && AV.data) { const nd = AV.data.nodesById[r.dataset.id]; if (nd && AV.nodeEls[nd.id]) { AV.pinned = nd.id; avHighlight(); avInspect(nd, AV.gr); } }
});

// ---------- inspector ----------
function avName(id, Gr) {
  const nd = (Gr && Gr.nodes.find(x => x.id === id)) || AV.data.nodesById[id]; if (!nd) return id;
  if (nd.type === 'feature') return Gr && Gr.super ? nd.label : (AV.data.mode === 'edge' ? `f${nd.f} @${nd.t}:${nd.s}→${nd.i}` : `f${nd.f} @${nd.t}:${nd.i} · ${nd.label}`);
  if (nd.type === 'leaf' && !(Gr && Gr.super)) return avNodeText(nd, Gr || { super: false }) + ` @${nd.t}`;
  if (nd.type === 'bias') return nd.t > AV.data.t ? 'bias (outputs)' : `bias @${nd.t}`;
  if (nd.type === 'error') return (Gr && Gr.super) ? `errors @${nd.t}` : (AV.data.mode === 'edge' ? `err @${nd.t}:${nd.s}→${nd.i}` : `err @${nd.t}:${nd.i}`);
  return avNodeText(nd, Gr || { super: false });
}
function avTargetName(id) { const tg = AV.data.targets.find(x => x.id === id); return tg ? (tg.kind === 'pointer' ? `${tg.field}[${tg.i}]→${tg.j}` : `${W.target}[${tg.i}]${tg.ftype === 'categorical' ? '=' + tg.c : ''}`) : id; }
function avBars(list, labelFn) {
  if (!list.length) return '<p class="empty">none in the current view</p>';
  const m = Math.max(...list.map(x => Math.abs(x.w)), 1e-9);
  return `<div class="bars">${list.map(x => `<span data-tip="${x.tip || ''}">${avEsc(labelFn(x))}</span><span class="bar"><i class="${x.w >= 0 ? 'pos' : 'neg'}" style="width:${(Math.abs(x.w) / m * 100).toFixed(0)}%"></i></span><span>${x.w >= 0 ? '+' : ''}${x.w.toFixed(2)}</span>`).join('')}</div>`;
}
function avEdgeList(id, dir, Gr) {
  const list = (dir === 'in' ? Gr.edges.filter(e => e.d === id) : Gr.edges.filter(e => e.s === id)).sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 10);
  return avBars(list.map(e => ({ w: e.w, other: dir === 'in' ? e.s : e.d, tip: 'an:' + (dir === 'in' ? e.s : e.d) })), x => avName(x.other, Gr));
}
function avInspectEmpty() { document.getElementById('featDetail').innerHTML = '<p class="empty">No feature survives pruning for this target.</p>'; document.getElementById('ivPanel').innerHTML = ''; }
function avInspect(nd, Gr) {
  Gr = Gr || AV.gr; const d = AV.data; let h = '';
  const iv = document.getElementById('ivPanel'); iv.innerHTML = '<p class="empty small">Pin a feature to clamp it.</p>';
  if (nd.type === 'feature' && !Gr.super && d.mode === 'edge') {
    const fi = nd.fires || { n: 0, real: 0, self: 0, pairs: [] };
    h += `<div class="ititle">latent f${nd.f} at step ${nd.t}, pair ${nd.s} → ${nd.i}</div>
      <div class="chips"><span class="chip grp" style="background:var(${AV_GROUP[nd.group][0]})">${AV_GROUP[nd.group][1]}</span></div>
      <dl class="ikv"><dt data-tip="g:avAct">activation</dt><dd>${nd.act.toFixed(3)}</dd><dt data-tip="g:avInfluence">influence on targets</dt><dd>${(d.influence[nd.id] || 0).toFixed(4)}</dd>
      <dt>active in this graph</dt><dd>${fi.n} of ${G.n * G.n} pairs at step ${nd.t}: ${fi.real} real edges, ${fi.self} self pairs${fi.pairs.length ? ' (e.g. ' + fi.pairs.slice(0, 5).map(p => p[0] + '→' + p[1]).join(', ') + (fi.n > 5 ? ', …' : '') + ')' : ''}</dd></dl>
      ${avMeaningHtml(nd)}`;
    const te = d.totalEffect[nd.id] || {};
    const rows = Object.entries(te).filter(([id]) => d.targets.some(x => x.id === id && (x.prob >= 0.01 || x.always))).map(([id, w]) => ({ id, w, tip: 'an:' + id })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 6);
    h += `<div class="isec"><h3 data-tip="g:avTotalEffect">Total effect on the targets (per unit activation)</h3>${avBars(rows, x => avTargetName(x.id))}</div>`;
    h += `<div class="isec"><h3>Incoming edges</h3>${avEdgeList(nd.id, 'in', Gr)}</div><div class="isec"><h3>Outgoing edges</h3>${avEdgeList(nd.id, 'out', Gr)}</div>`;
    iv.innerHTML = `<div class="small">Clamp <b>f${nd.f}@${nd.t}:${nd.s}→${nd.i}</b> (its message changes by (factor − 1) × activation × decoder vector) and re-run the real model (its own hint feedback included):</div>
      <div class="ivbtns"><button id="ivNeg" data-tip="g:ivNeg">Clamp to −1×</button> <button id="ivZero" data-tip="g:ivZero">Ablate (×0)</button> <button id="ivDbl" data-tip="g:ivDbl">Amplify (×2)</button></div><div id="ivResult"></div>`;
    [['ivNeg', -1], ['ivZero', 0], ['ivDbl', 2]].forEach(([id, f]) => document.getElementById(id).addEventListener('click', () => avIntervene(nd, f)));
    document.getElementById('featDetail').innerHTML = h; return;
  }
  if (nd.type === 'feature' && !Gr.super) {
    const L = labelByFeat[nd.f] || {};
    h += `<div class="ititle">f${nd.f} at step ${nd.t}, node ${nd.i}</div>
      <div class="chips"><span class="chip grp" style="background:var(${AV_GROUP[nd.group][0]})" data-tip="fl:${nd.f}">${avEsc(nd.label)}</span>
      <span class="chip" data-tip="g:precision">P ${fmt(L.precision, 2)}</span><span class="chip" data-tip="g:recall">R ${fmt(L.recall, 2)}</span><span class="chip" data-tip="g:f1">F1 ${fmt(L.f1, 2)}</span></div>
      <dl class="ikv"><dt data-tip="g:avAct">activation</dt><dd>${nd.act.toFixed(3)}</dd><dt data-tip="g:avInfluence">influence on targets</dt><dd>${(d.influence[nd.id] || 0).toFixed(4)}</dd>
      <dt data-tip="g:firesOn">fires on</dt><dd>${L.n_fire || 0} of ${FL.total_sites} dataset sites</dd></dl>`;
    const te = d.totalEffect[nd.id] || {};
    const rows = Object.entries(te).filter(([id]) => d.targets.some(x => x.id === id && (x.prob >= 0.01 || x.always))).map(([id, w]) => ({ id, w, tip: 'an:' + id })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 6);
    h += `<div class="isec"><h3 data-tip="g:avTotalEffect">Total effect on the targets (per unit activation)</h3>${avBars(rows, x => avTargetName(x.id))}</div>`;
    h += `<div class="isec"><h3 data-tip="g:featdetail">Top activating contexts (dataset)</h3><div id="ctxBox" class="small">loading real examples…</div></div>`;
    h += `<div class="isec"><h3>Incoming edges</h3>${avEdgeList(nd.id, 'in', Gr)}</div><div class="isec"><h3>Outgoing edges</h3>${avEdgeList(nd.id, 'out', Gr)}</div>`;
    iv.innerHTML = `<div class="small">Clamp <b>f${nd.f}@${nd.t}:${nd.i}</b> and re-run the real model (its own hint feedback included):</div>
      <div class="ivbtns"><button id="ivNeg" data-tip="g:ivNeg">Clamp to −1×</button> <button id="ivZero" data-tip="g:ivZero">Ablate (×0)</button> <button id="ivDbl" data-tip="g:ivDbl">Amplify (×2)</button></div><div id="ivResult"></div>`;
    [['ivNeg', -1], ['ivZero', 0], ['ivDbl', 2]].forEach(([id, f]) => document.getElementById(id).addEventListener('click', () => avIntervene(nd, f)));
  } else if (nd.type === 'feature') {
    h += `<div class="ititle">Supernode · ${avEsc(nd.label)} · step ${nd.t}</div><div class="chips"><span class="chip grp" style="background:var(${AV_GROUP[nd.group || avGroup(nd.label)][0]})">${nd.members.length} features</span></div>
      <p class="note">Members: ${nd.members.map(id => { const m = d.nodesById[id]; return d.mode === 'edge' ? `f${m.f}@${m.s}→${m.i}` : `f${m.f}@${m.i}`; }).join(', ')}</p>
      <div class="isec"><h3>Incoming edges (summed)</h3>${avEdgeList(nd.id, 'in', Gr)}</div><div class="isec"><h3>Outgoing edges (summed)</h3>${avEdgeList(nd.id, 'out', Gr)}</div>`;
  } else if (nd.type === 'leaf') {
    const desc = nd.probe === 'state' ? `The node's hidden state from before step ${d.tLo} (the window shown starts at step ${d.tLo}); its effect is exact but not traced further back.`
      : `${nd.stage === 'hint' ? 'Hint' : 'Input'} <span class="mono">${avEsc(nd.probe)}</span> (${nd.stage} · ${nd.loc} · ${nd.ftype}), read by its linear encoder at step ${nd.t}.` + (nd.stage === 'hint' ? ' Hints are the model’s own previous prediction, fed back in: here it is an observed value (a leaf).' : '') + ' Edge weight = value × (encoder direction · frozen path · destination).';
    h += `<div class="ititle">${Gr.super ? avEsc(nd.label) + ' leaves' : avEsc(avNodeText(nd, Gr)) + (nd.value !== null && nd.value !== undefined ? ' = ' + (typeof nd.value === 'number' ? +nd.value.toFixed(3) : nd.value) : '')}</div>
      <p class="note">${Gr.super ? `${nd.members.length} values encoded at step ${nd.t}.` : desc}</p><div class="isec"><h3>Outgoing edges</h3>${avEdgeList(nd.id, 'out', Gr)}</div>`;
  } else if (nd.type === 'logit') {
    h += `<div class="ititle">${nd.kind === 'pointer' ? 'Pointer hint logit' : 'Hint logit'} · ${avEsc(avTargetText(nd, d.t))}</div>
      <dl class="ikv"><dt>logit</dt><dd>${nd.logit.toFixed(3)}</dd><dt>probability</dt><dd>${nd.ftype === 'scalar' ? 'n/a (scalar)' : (nd.kind === 'pointer' || nd.ftype === 'categorical' ? nd.prob : nd.p1).toFixed(4)}</dd></dl>
      <div class="isec"><h3>Incoming edges</h3>${avEdgeList(nd.id, 'in', Gr)}</div>`;
  } else if (nd.type === 'error') {
    h += `<div class="ititle">Error node · ${Gr.super ? `step ${nd.t}` : `step ${nd.t}, node ${nd.i}`}</div><p class="note">${d.mode === 'edge' ? 'The part of the pair’s message that the transcoder did not reconstruct' : 'The part of the new hidden state the transcoder did not reconstruct'}${Gr.super ? '' : ` (${d.mode === 'edge' ? '‖y − ŷ‖' : '‖h − ĥ‖'} = ${nd.norm.toFixed(3)})`}. Included so the replacement model reproduces the real network exactly; its outgoing weight is how much of the explanation is not captured by features.</p><div class="isec"><h3>Outgoing edges</h3>${avEdgeList(nd.id, 'out', Gr)}</div>`;
  } else {
    h += `<div class="ititle">Bias · ${nd.t > d.t ? 'into the predictions' : 'into step ' + nd.t}</div><p class="note">${d.mode === 'edge' ? 'All constant terms on the frozen path (encoder, message-site, node-update, hidden-state-decoder and transcoder biases and offsets), added up from their parts. Every value equals the sum of its incoming edges plus this term; the page checks that for every node.' : 'All constant terms on the frozen path (encoder, message, node-update, transcoder and decoder biases), summed, so every value is exactly the sum of its incoming edges.'}</p><div class="isec"><h3>Outgoing edges</h3>${avEdgeList(nd.id, 'out', Gr)}</div>`;
  }
  document.getElementById('featDetail').innerHTML = h;
  if (nd.type === 'feature' && !Gr.super) loadContexts(nd.f);
}

// ---------- interventions ----------
function avIntervene(nd, factor) {
  const res = document.getElementById('ivResult'); res.innerHTML = '<span class="small">re-running the model…</span>';
  const id = 'i' + (++AV.ivSeq); AV.ivWant = { id, nd, factor };
  if (AV.data.mode === 'edge') AV.worker.postMessage({ type: 'interveneEdge', id, ref: AV.data.id, node: nd.id, factor });
  else AV.worker.postMessage({ type: 'intervene', id, ref: AV.data.id, t0: nd.t, i0: nd.i, q: nd.f, factor });
}
function avIvResult(m) {
  if (!AV.ivWant || m.id !== AV.ivWant.id) return;
  const res = document.getElementById('ivResult'); if (!res) return;
  if (m.type === 'error') { res.innerHTML = `<p class="note">${avEsc(m.error.split('\n')[0])}</p>`; return; }
  const iv = m.iv; LAST_IV = { name: iv.s0 !== undefined ? `f${iv.f}@${iv.t0}:${iv.s0}→${iv.i0}` : `f${iv.q}@${iv.t0}:${iv.i0}`, factor: iv.factor, base: iv.rows[0].base, after: iv.rows[0].actual,
    predicted: `${(iv.rows[0].predicted - iv.rows[0].base) >= 0 ? '+' : ''}${(iv.rows[0].predicted - iv.rows[0].base).toFixed(3)} (graph)` };
  const agree = iv.rows.filter(r => Math.abs(r.actual - r.base) > 1e-3).every(r => Math.sign(r.actual - r.base) === Math.sign(r.predicted - r.base));
  res.innerHTML = `<table class="ivtab" data-tip="dyn:iv"><tr><th>target</th><th>logit</th><th>Δ actual</th><th>Δ graph</th></tr>${iv.rows.map(r => `<tr><td>${avEsc(avTargetName(r.id))}</td><td>${r.base.toFixed(2)}</td><td>${(r.actual - r.base >= 0 ? '+' : '') + (r.actual - r.base).toFixed(3)}</td><td>${(r.predicted - r.base >= 0 ? '+' : '') + (r.predicted - r.base).toFixed(3)}</td></tr>`).join('')}</table>
    <p class="note">${agree ? '<b>Direction confirmed:</b> the real model moved the way the graph predicted.' : '<b>Not confirmed:</b> the real model moved differently from the linear prediction.'} ${iv.flips} of ${iv.decisions.toLocaleString()} downstream ${iv.s0 !== undefined ? 'max-routing winner / TopK decisions' : 'ReLU / max-routing decisions'} changed (${iv.decisions ? (100 * iv.flips / iv.decisions).toFixed(2) : '0.00'}%)${iv.flips ? '; where they flip, the frozen-linear graph can only approximate.' : '.'} “Δ graph” = (factor − 1) × activation × total effect.${iv.s0 !== undefined ? ' The graph freezes the LayerNorm σ, the max-routing winners and the TopK sets, so this is only a first-order guide: in our tests (ablating the most influential latents on all 12 algorithms, 325 comparisons) its direction matched the real model 69% of the time, 44–77% depending on the algorithm. “Δ actual” is the real model with its own hint feedback.' : ''}</p>`;
}

// ---------- simulation panel helpers ----------
function avDotsFor(step) {                 // active features per graph node at this step, in the current (pruned) graph
  if (!avSameGraph() || AV.data.t !== step) return null;
  const inView = new Set(pruneMode === 'pruned' ? AV.data.pruned.nodes : AV.data.nodes.map(n => n.id));
  const out = {}; AV.data.nodes.forEach(nd => { if (nd.type === 'feature' && nd.t === step && inView.has(nd.id)) (out[nd.i] = out[nd.i] || []).push(nd); });
  Object.values(out).forEach(l => l.sort((a, b) => b.act - a.act)); return out;
}
function avRealPred(step, node) {           // the real model's prediction for any node at this step (from the traced run)
  if (!avSameGraph() || AV.data.t !== step || !AV.data.rowLogits) return null;
  const L = AV.data.rowLogits, tt = W.target_type;
  if (tt === 'categorical') { const C = L.length / G.n; let b = 0; for (let c = 1; c < C; c++) if (L[node * C + c] > L[node * C + b]) b = c; return 'class ' + b; }
  if (tt === 'mask') return (1 / (1 + Math.exp(-L[node]))).toFixed(2);
  if (tt === 'mask_one') { const m = Math.max(...L); const ex = L.map(v => Math.exp(v - m)); const s = ex.reduce((a, b) => a + b, 0); return (ex[node] / s).toFixed(2); }
  return L[node].toFixed(2);
}

// ---------- hover cards (explain.js delegates attribution nodes / edges here) ----------
window.avCard = function (id) {
  if (!avSameGraph()) return null; const d = AV.data, Gr = AV.gr || { super: false, nodes: [], edges: [] };
  const nd = Gr.nodes.find(x => x.id === id) || d.nodesById[id]; if (!nd) return null;
  const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  const inW = Gr.edges.filter(e => e.d === id).length, outW = Gr.edges.filter(e => e.s === id).length;
  if (nd.type === 'feature' && !Gr.super && d.mode === 'edge') {
    const te = d.totalEffect[id] || {};
    return `<div class="hc-title">Latent ${nd.f} · step ${nd.t} · pair ${nd.s} → ${nd.i}</div><div class="hc-body"><b>${avEsc(AV_GROUP[nd.group][1])}</b> <span class="hc-muted">unnamed</span></div>
      <table class="hc-kv">${row('activation', nd.act.toFixed(3))}${row('influence on targets', (d.influence[id] || 0).toFixed(4))}${d.targets.filter(t => t.always).map(t => row('total effect → ' + avEsc(avTargetName(t.id)), ((te[t.id] || 0) >= 0 ? '+' : '') + (te[t.id] || 0).toFixed(3))).join('')}${row('edges in / out (view)', `${inW} / ${outW}`)}</table>
      ${(() => { const J = avJ(), v = J && J.latents[nd.f]; return J ? `<div class="hc-how"><b>${v && v.tier !== '-' ? avEsc(J.conditions[v.best[0][0]].text) + ' (tier ' + v.tier + ')' : 'Unnamed'}</b>${v && v.eff ? '<br>' + avEffectText(J, v) : ''}</div>` : ''; })()}
      <div class="hc-why"><b>Why:</b> one of the 8 transcoder latents active on the pair ${nd.s} → ${nd.i} in step ${nd.t}. It writes into the message that node ${nd.s} sends to node ${nd.i}; that message reaches node ${nd.i}’s new state only through the channels where ${nd.s} is the max-routing winner.</div><div class="hc-how">Click to pin it: highlights its edges, opens the inspector and lets you clamp it.</div>`;
  }
  if (nd.type === 'feature' && !Gr.super) {
    const L = labelByFeat[nd.f] || {}, te = d.totalEffect[id] || {};
    return `<div class="hc-title">Feature ${nd.f} · step ${nd.t} · node ${nd.i}</div><div class="hc-body"><b>${avEsc(nd.label)}</b> <span class="hc-muted">P ${fmt(L.precision, 2)} · R ${fmt(L.recall, 2)} · F1 ${fmt(L.f1, 2)}</span></div>
      <table class="hc-kv">${row('activation', nd.act.toFixed(3))}${row('influence on targets', (d.influence[id] || 0).toFixed(4))}${d.targets.filter(t => t.always).map(t => row('total effect → ' + avEsc(avTargetName(t.id)), ((te[t.id] || 0) >= 0 ? '+' : '') + (te[t.id] || 0).toFixed(3))).join('')}${row('edges in / out (view)', `${inW} / ${outW}`)}</table>
      <div class="hc-why"><b>Why:</b> one of the ≤8 transcoder features active at node ${nd.i} in step ${nd.t}. ${nd.t === d.t ? 'It writes directly into the new state the decoders read.' : 'It acts through the next step’s hidden state and the messages this node sends to others.'}</div><div class="hc-how">Click to pin it: highlights its edges, opens the inspector and lets you clamp it.</div>`;
  }
  if (nd.type === 'feature') return `<div class="hc-title">Supernode · ${avEsc(nd.label)}</div><div class="hc-body">${nd.members.length} feature instance(s) at step ${nd.t} that share this label, merged; edges are exact sums.</div>`;
  if (nd.type === 'leaf') { const lab = avNodeText(nd, Gr);
    return `<div class="hc-title">${Gr.super ? avEsc(nd.label) + ' leaves' : 'Encoder leaf · ' + avEsc(lab)}</div><div class="hc-body">${nd.probe === 'state' ? `Hidden state of node ${nd.i} from before step ${d.tLo}.` : `${avEsc(nd.probe)}: ${nd.stage} · ${nd.loc} · ${nd.ftype}${nd.value !== null && nd.value !== undefined ? `, value ${typeof nd.value === 'number' ? +nd.value.toFixed(3) : nd.value}` : ''}, encoded at step ${nd.t}.`}</div>
      <div class="hc-why"><b>Why:</b> leaves are where the explanation stops: observed inputs, hints the model fed back to itself, or older state outside the window.</div>`; }
  if (nd.type === 'error') return `<div class="hc-title">Error node · ${Gr.super ? 'step ' + nd.t : `step ${nd.t}, node ${nd.i}`}</div><div class="hc-body">What the transcoder did not reconstruct${Gr.super ? '' : ` (‖h − ĥ‖ = ${nd.norm.toFixed(3)})`}.</div><div class="hc-why"><b>Why:</b> keeps the replacement model exact (as in the Circuit Tracing paper). Large outgoing weight = part of the mechanism the features miss.</div>`;
  if (nd.type === 'bias') return `<div class="hc-title">Bias</div><div class="hc-body">All constant terms on the frozen path into ${nd.t > d.t ? 'the predictions' : 'step ' + nd.t}, summed.</div>`;
  if (nd.type === 'logit') return `<div class="hc-title">${nd.kind === 'pointer' ? 'Pointer hint' : 'Target hint'} · ${avEsc(avTargetText(nd, d.t))}</div><table class="hc-kv">${row('logit', nd.logit.toFixed(3))}${nd.ftype !== 'scalar' ? row('probability', (nd.kind === 'pointer' || nd.ftype === 'categorical' ? nd.prob : nd.p1).toFixed(4)) : ''}${row('incoming edges (view)', inW)}</table><div class="hc-why"><b>Why:</b> the prediction being explained. Every path in the graph ends here.</div>`;
  return null;
};
window.avEdgeCard = function (idx) {
  if (!AV.gr || !avSameGraph()) return null; const e = AV.gr.edges[+idx]; if (!e) return null;
  const into = AV.gr.edges.filter(x => x.d === e.d).reduce((a, x) => a + Math.abs(x.w), 0) || 1;
  const sn = AV.data.nodesById[e.s] || {};
  const how = sn.type === 'feature' ? 'activation × (decoder vector · frozen path · destination encoder)' : sn.type === 'leaf' ? 'value × (encoder direction · frozen path · destination)' : sn.type === 'error' ? 'reconstruction error · frozen path · destination' : 'sum of constant terms';
  return `<div class="hc-title">${avEsc(avName(e.s, AV.gr))} → ${avEsc(avName(e.d, AV.gr))}</div><div class="hc-body">${e.w >= 0 ? 'Pushes the destination <b>up</b>.' : 'Pushes the destination <b>down</b>.'}</div>
    <table class="hc-kv"><tr><td>weight</td><td>${e.w >= 0 ? '+' : ''}${e.w.toFixed(3)}</td></tr><tr><td>share of |inputs| to destination (view)</td><td>${(100 * Math.abs(e.w) / into).toFixed(1)}%</td></tr></table><div class="hc-how">Edge = ${how}. Direct effect only; indirect effects flow through the intermediate nodes.</div>`;
};
