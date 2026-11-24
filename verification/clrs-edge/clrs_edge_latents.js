// What does each edge-site latent do?  node clrs_edge_latents.js <algo>   (env: MODELS, INST = folder with <algo>_<seed>.json, OUT = folder to write <algo>.latents.json)
//
// For one algorithm, over many fresh n = 16 graphs from the tracer's own sampler (seeds the transcoder never saw), this measures for every latent that fires:
//   - how often it fires, how strongly, and on which kind of pair (real edge / non-edge / self pair);
//   - what it does: its EFFECT on the main target logit of the receiver, per unit of activation (the graph's own edge weight, summed over the step's winners; checked
//     by the sum test on every call), mean +- sd over all its activations;
//   - how well it matches simple, checkable conditions on the pair (is the sender already reached, does the receiver change next step, ...): precision / recall / F1.
//     The conditions are fitted on graphs with even seeds and SCORED ON THE ODD SEEDS the fit never saw; a latent is only named when its held-out F1 >= 0.5
//     (tier B) or >= 0.8 (tier A). Everything else stays unnamed, with its best match shown as a fact, not as a name.
//   - its three strongest activations (seed, step, sender, receiver) so they can be looked up in the page.
// Conditions use the TRUE hints of the reference algorithm (instance.target_true) and the input graph; "sender wins" uses the model's own max-routing.
const fs = require('fs'), path = require('path'), ROOT = path.join(__dirname, '..', '..');
const E = require(ROOT + '/clrs-edge-attrib.js'), MP = require(ROOT + '/mpnn.js'), AT = require(ROOT + '/attrib.js');
const algo = process.argv[2], MODELS = process.env.MODELS, INST = process.env.INST, OUT = process.env.OUT || path.join(ROOT, 'edge_tc');
const MAXSTEPS = +(process.env.MAXSTEPS || 20);                      // steps per graph used (evenly spaced) so the long algorithms stay affordable
const meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'algos', algo, 'meta.json'))), TYPE = meta.target_type, TARGET = meta.target;
const man = JSON.parse(fs.readFileSync(path.join(MODELS, algo + '.json'))), buf = fs.readFileSync(path.join(MODELS, algo + '.bin'));
const M = MP.loadModel(man, new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
const T = E.prepTC(JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', algo + '.json'))), fs.readFileSync(path.join(ROOT, 'edge_tc', algo + '.bin')));
const H = M.H, D = T.D, pre = new Float64Array(D), tmp = new Float64Array(D);

// ---------- the conditions ----------
const hasVal = (v, node, src) => node === src || v > 1e-6;
function catalog(K) {
  const c = [['edge', 'the pair is a real edge of the input graph'], ['nonedge', 'the pair is not an edge of the input graph'], ['self', 'sender and receiver are the same node'],
             ['src_sender', 'the sender is the source node'], ['src_receiver', 'the receiver is the source node'], ['sender_wins', 'the sender wins at least one channel of the max at the receiver'],
             ['receiver_changes', `${TARGET} of the receiver changes at the next step`]];
  if (TYPE === 'mask') c.push(['sender_on', `${TARGET} of the sender is 1`], ['sender_off', `${TARGET} of the sender is 0`], ['receiver_on', `${TARGET} of the receiver is 1`], ['receiver_off', `${TARGET} of the receiver is 0`],
    ['receiver_turns_on', `${TARGET} of the receiver turns from 0 to 1`], ['receiver_stays_on', `${TARGET} of the receiver stays 1`], ['receiver_stays_off', `${TARGET} of the receiver stays 0`],
    ['frontier', `real edge from a sender with ${TARGET} = 1 to a receiver with ${TARGET} = 0`], ['discovers', `real edge from a sender with ${TARGET} = 1 to a receiver whose ${TARGET} turns 1`]);
  if (TYPE === 'mask_one') c.push(['sender_marked', `the sender is the node marked by ${TARGET} now`], ['receiver_marked_now', `the receiver is the node marked by ${TARGET} now`],
    ['receiver_marked_next', `the receiver is the node marked by ${TARGET} at the next step`], ['receiver_newly_marked', `the receiver becomes the marked node of ${TARGET}`]);
  if (TYPE === 'categorical') { c.push(['receiver_changes_class', `the class of ${TARGET} of the receiver changes`]);
    for (let k = 0; k < K; k++) c.push([`sender_class_${k}`, `${TARGET} of the sender is class ${k}`], [`receiver_class_${k}`, `${TARGET} of the receiver is class ${k}`], [`receiver_becomes_${k}`, `the receiver turns into class ${k} of ${TARGET}`]); }
  if (TYPE === 'scalar') c.push(['sender_has_value', `the sender already has a value of ${TARGET}`], ['sender_no_value', `the sender has no value of ${TARGET} yet`], ['receiver_has_value', `the receiver already has a value of ${TARGET}`],
    ['receiver_no_value', `the receiver has no value of ${TARGET} yet`], ['receiver_gets_value', `the receiver gets its first value of ${TARGET}`], ['receiver_value_down', `${TARGET} of the receiver decreases`],
    ['receiver_value_up', `${TARGET} of the receiver increases`], ['sender_value_lower', `${TARGET} of the sender is lower than that of the receiver (both have values)`]);
  return c;
}
// truth of every condition for the pair (s -> r) at step t
function truth(ids, inst, tt, t, s, r, wins, N) {
  const out = [], src = inst.source, A = inst.adj, a = tt[t], b = tt[t + 1], real = s !== r && A[s][r] > 0, on = v => v > 0.5, K = ids.K;
  const add = id => { const q = ids.index[id]; if (q !== undefined) out.push(q); };
  if (real) add('edge'); else if (s === r) add('self'); else add('nonedge');
  if (s === src) add('src_sender'); if (r === src) add('src_receiver'); if (wins > 0) add('sender_wins');
  const changed = Math.abs(a[r] - b[r]) > 1e-6; if (changed) add('receiver_changes');
  if (TYPE === 'mask') { const so = on(a[s]), ro = on(a[r]), rn = on(b[r]);
    add(so ? 'sender_on' : 'sender_off'); add(ro ? 'receiver_on' : 'receiver_off');
    if (!ro && rn) add('receiver_turns_on'); if (ro && rn) add('receiver_stays_on'); if (!ro && !rn) add('receiver_stays_off');
    if (real && so && !ro) add('frontier'); if (real && so && !ro && rn) add('discovers'); }
  if (TYPE === 'mask_one') { if (on(a[s])) add('sender_marked'); if (on(a[r])) add('receiver_marked_now'); if (on(b[r])) add('receiver_marked_next'); if (!on(a[r]) && on(b[r])) add('receiver_newly_marked'); }
  if (TYPE === 'categorical') { if (Math.round(a[r]) !== Math.round(b[r])) add('receiver_changes_class'); add('sender_class_' + Math.round(a[s])); add('receiver_class_' + Math.round(a[r])); if (Math.round(a[r]) !== Math.round(b[r])) add('receiver_becomes_' + Math.round(b[r])); }
  if (TYPE === 'scalar') { const sh = hasVal(a[s], s, src), rh = hasVal(a[r], r, src), rn = hasVal(b[r], r, src);
    add(sh ? 'sender_has_value' : 'sender_no_value'); add(rh ? 'receiver_has_value' : 'receiver_no_value'); if (!rh && rn) add('receiver_gets_value');
    if (rh && b[r] < a[r] - 1e-6) add('receiver_value_down'); if (rh && b[r] > a[r] + 1e-6) add('receiver_value_up'); if (sh && rh && a[s] < a[r]) add('sender_value_lower'); }
  return out;
}

// ---------- pass over the graphs ----------
const files = fs.readdirSync(INST).filter(f => f.startsWith(algo + '_') && f.endsWith('.json')).sort();
if (!files.length) throw new Error('no instances for ' + algo);
let K = 1; if (TYPE === 'categorical') for (const f of files) { const tt = JSON.parse(fs.readFileSync(path.join(INST, f))).target_true; for (const row of tt) for (const v of row) K = Math.max(K, Math.round(v) + 1); }
const CAT = catalog(Math.min(K, 8)), ids = { index: Object.fromEntries(CAT.map((c, q) => [c[0], q])), K }, NP = CAT.length;
const lat = new Map();       // latent -> stats
const L = f => { let v = lat.get(f); if (!v) { v = { nf: 0, nh: 0, sum: 0, mx: 0, kind: [0, 0, 0], tpF: new Int32Array(NP), tpH: new Int32Array(NP), ex: [], eN: 0, eS: 0, eQ: 0, eP: 0 }; lat.set(f, v); } return v; };
const predN = { F: new Int32Array(NP), H: new Int32Array(NP) }; let rowsF = 0, rowsH = 0, maxCheck = 0, effCalls = 0, nGraphs = 0;
const t0 = Date.now();
for (const file of files) {
  const seed = +/_(\d+)\.json$/.exec(file)[1], fit = seed % 2 === 0, inst = JSON.parse(fs.readFileSync(path.join(INST, file))), N = inst.n, steps = inst.length - 1, tt = inst.target_true;
  const run = MP.runRecord(M, inst, { record: { from: 0, to: steps - 1 } }); nGraphs++;
  const stride = Math.max(1, Math.ceil(steps / MAXSTEPS));
  for (let t = 0; t < steps; t += stride) {
    const rec = run.rec[t], wins = new Int32Array(N * N);                         // wins[s*N+r] = channels the sender wins at the receiver
    for (let r = 0; r < N; r++) for (let c = 0; c < H; c++) wins[rec.route[r * H + c] * N + r]++;
    for (let s = 0; s < N; s++) for (let r = 0; r < N; r++) {
      const p = s * N + r, idx = E.encodeRow(T, rec.s, p * H, pre, tmp), tr = truth(ids, inst, tt, t, s, r, wins[p], N), kind = s === r ? 2 : (inst.adj[s][r] > 0 ? 0 : 1);
      if (fit) rowsF++; else rowsH++; for (const q of tr) (fit ? predN.F : predN.H)[q]++;
      for (const f of idx) { const v = L(f), a = pre[f]; if (fit) v.nf++; else v.nh++; v.sum += a; if (a > v.mx) v.mx = a; v.kind[kind]++;
        const tp = fit ? v.tpF : v.tpH; for (const q of tr) tp[q]++;
        if (v.ex.length < 3 || a > v.ex[2][4]) { v.ex.push([seed, t, s, r, a]); v.ex.sort((x, y) => y[4] - x[4]); if (v.ex.length > 3) v.ex.pop(); } }
    }
  }
  // effect of every latent on the receiver's main target logit, per unit of activation (every other step of the used ones)
  for (let t = 0; t < steps; t += stride * 2) for (let r = 0; r < N; r++) {
    const ex = E.explain(MP, AT, M, T, inst, { t, i: r, main: TARGET, pointerField: null, window: 1, run, lite: true }); effCalls++;
    maxCheck = Math.max(maxCheck, ex.checks.maxAbs);
    const lg = ex.nodes.findIndex(n => n.type === 'logit');
    for (let e = 0; e < ex.E; e++) if (ex.eD[e] === lg) { const nd = ex.nodes[ex.eS[e]]; if (nd.type !== 'feature') continue; const eff = ex.eW[e] / nd.act, v = L(nd.f); v.eN++; v.eS += eff; v.eQ += eff * eff; if (eff > 0) v.eP++; }
  }
  if (nGraphs % 10 === 0) console.error(algo, nGraphs, 'graphs', ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
if (maxCheck > 1e-9) throw new Error('sum test failed during labelling: ' + maxCheck);

// ---------- scoring: pick on the fit half, report on the held-out half ----------
const f1 = (tp, nAct, nTrue) => { const P = nAct ? tp / nAct : 0, R = nTrue ? tp / nTrue : 0; return [P, R, P + R ? 2 * P * R / (P + R) : 0]; };
const r3 = x => +x.toPrecision(3), out = { algo, target: TARGET, target_type: TYPE, graphs: nGraphs, graphs_fit: files.filter(f => +/_(\d+)\.json$/.exec(f)[1] % 2 === 0).length, rows_fit: rowsF, rows_held: rowsH,
  split: 'conditions are chosen on graphs with even seeds and scored on graphs with odd seeds', sampler: 'n=16, p=0.4', max_steps_per_graph: MAXSTEPS,
  effect_of: TYPE === 'categorical' ? `logit of the class the model predicts for the receiver` : TYPE === 'mask_one' ? `logit of "the receiver is the node marked by ${TARGET}"` : `logit of ${TARGET} at the receiver`, max_sum_check: maxCheck, effect_calls: effCalls,
  conditions: CAT.map(c => ({ id: c[0], text: c[1], base_rate_held: r3(predN.H[ids.index[c[0]]] / Math.max(1, rowsH)) })), latents: {} };
let named = 0, tierA = 0;
for (const [f, v] of [...lat.entries()].sort((a, b) => a[0] - b[0])) {
  const n = v.nf + v.nh, rank = [];
  for (let q = 0; q < NP; q++) if (predN.F[q] > 0 && v.nf > 0) { const [P, R, F] = f1(v.tpF[q], v.nf, predN.F[q]); rank.push([q, F, P, R]); }
  rank.sort((a, b) => b[1] - a[1]);
  const top = rank.slice(0, 3).map(([q, Ff]) => { const [P, R, F] = f1(v.tpH[q], v.nh, predN.H[q]); return [q, r3(P), r3(R), r3(F), r3(Ff)]; });   // [condition, held-out P, R, F1, fit F1]
  let tier = '-', label = null;
  if (v.nf >= 20 && v.nh >= 10 && top.length) { const F = top[0][3]; if (F >= 0.8) tier = 'A'; else if (F >= 0.5) tier = 'B'; if (tier !== '-') { label = CAT[top[0][0]][1]; named++; if (tier === 'A') tierA++; } }
  out.latents[f] = { n, rate: r3(n / (rowsF + rowsH)), act: r3(v.sum / n), max: r3(v.mx), kind: v.kind.map(k => r3(k / n)), eff: v.eN ? [r3(v.eS / v.eN), r3(Math.sqrt(Math.max(0, v.eQ / v.eN - (v.eS / v.eN) ** 2))), v.eN, r3(v.eP / v.eN)] : null,
    best: top, tier, label, ex: v.ex.map(e => [e[0], e[1], e[2], e[3], r3(e[4])]) };
}
out.fired = lat.size; out.named = named; out.tier_a = tierA; out.never_fired = D - lat.size;
fs.mkdirSync(OUT, { recursive: true }); fs.writeFileSync(path.join(OUT, algo + '.latents.json'), JSON.stringify(out));
console.log(`${algo}: ${nGraphs} graphs, ${rowsF + rowsH} pair rows, ${lat.size}/${D} latents fired, ${named} named (tier A ${tierA}), effects over ${effCalls} calls, worst sum check ${maxCheck.toExponential(1)}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
