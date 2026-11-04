/* Attribution graphs for the CLRS-30 MPNN (128 hidden), following Anthropic's Circuit Tracing recipe.
 *
 * Replacement model: at every node-update site (step t, node i) the transcoder reads
 * x = [z_i, agg_i] (z_i = [node_fts_i, h_prev_i], agg_i = max-aggregated message) and writes the new hidden
 * state h'_i = W_dec·f + b_dec + err (the error node makes the replacement exact). With the MPNN's
 * nonlinearities frozen at their values on this input (message ReLU masks, max-routing per channel, the
 * transcoder's top-k set, pointer-decoder max selection), every feature pre-activation and every target logit
 * is an exact linear function of: features and error nodes one step earlier (through h_prev and through the
 * messages every other node sends), the encoded inputs and hints at that step (leaves), and constants (bias).
 * Hints fed back by the model (its own soft predictions) are leaves: values the encoders actually read.
 * Graphs span a window of steps [tLo, t]; state older than tLo enters as one "earlier state" leaf per node.
 * Works in browsers, Web Workers and Node. Needs mpnn.js (model + recorded forward pass). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Attrib = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const EPS = 1e-7;
  const zeros = n => new Float64Array(n);
  const dot = (a, b, ao = 0, bo = 0, n = a.length) => { let s = 0; for (let k = 0; k < n; k++) s += a[ao + k] * b[bo + k]; return s; };
  const nz = v => { if (!v) return false; for (let k = 0; k < v.length; k++) if (v[k] !== 0) return true; return false; };
  function linT(p, g) {                       // cotangent on outputs -> inputs (no bias); p.w row-major din*dout
    const { din, dout, w } = p, out = zeros(din);
    for (let i = 0; i < din; i++) { let s = 0; const r = i * dout; for (let o = 0; o < dout; o++) s += w[r + o] * g[o]; out[i] = s; }
    return out;
  }
  const addInto = (a, b, off = 0) => { for (let k = 0; k < b.length; k++) a[off + k] += b[k]; };

  // ---------- transcoder ----------
  function prepTranscoder(W) {
    const flat = (m) => { if (!Array.isArray(m[0])) return Float64Array.from(m); const r = m.length, c = m[0].length, o = zeros(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) o[i * c + j] = m[i][j]; return o; };
    const din = W.w_enc.length, F = W.F;
    return { F, K: W.K, din, H: W.hidden, enc: flat(W.w_enc), benc: flat(W.b_enc), dec: flat(W.w_dec), bdec: flat(W.b_dec), mu: flat(W.mu_x), sd: flat(W.sd_x) };
  }
  function transcode(T, x, hNew) {            // x: Float64Array(din); returns pre, f (top-k ReLU), err
    const { F, K, din, H, enc, benc, dec, bdec, mu, sd } = T, pre = Float64Array.from(benc);
    for (let c = 0; c < din; c++) { const v = (x[c] - mu[c]) / sd[c]; if (v === 0) continue; const r = c * F; for (let q = 0; q < F; q++) pre[q] += v * enc[r + q]; }
    const order = Array.from({ length: F }, (_, q) => q).sort((a, b) => Math.max(0, pre[b]) - Math.max(0, pre[a]));
    const f = zeros(F); for (let k = 0; k < K; k++) { const q = order[k]; if (pre[q] > 0) f[q] = pre[q]; }
    const err = Float64Array.from(hNew); for (let c = 0; c < H; c++) err[c] -= bdec[c];
    for (let q = 0; q < F; q++) if (f[q] > 0) { const r = q * H; for (let c = 0; c < H; c++) err[c] -= f[q] * dec[r + c]; }
    return { pre, f, err };
  }
  function siteInput(r, i, H) { const x = zeros(3 * H); x.set(r.z.subarray(i * 2 * H, (i + 1) * 2 * H), 0); x.set(r.agg.subarray(i * H, (i + 1) * H), 2 * H); return x; }

  // ---------- reverse mode through one frozen processor step ----------
  // cot: gz[k] (2H, cotangent on z_k = [node_fts_k, h_prev_k]), gagg[k] (H), ge Map (a*N+b -> H), gg (H)
  function backprop(M, N, r, cot) {
    const H = M.H, P = M.proc, gz = cot.gz, ge = cot.ge || new Map(), gg = cot.gg || zeros(H);
    const gmsg1 = new Array(N).fill(null), gmsg2 = new Array(N).fill(null);
    for (let b = 0; b < N; b++) {
      const gA = cot.gagg && cot.gagg[b]; if (!nz(gA)) continue;
      const gm2 = new Array(N).fill(null);
      for (let c = 0; c < H; c++) { if (gA[c] === 0) continue; const a = r.route[b * H + c]; (gm2[a] = gm2[a] || zeros(H))[c] += gA[c]; }
      for (let a = 0; a < N; a++) {
        if (!gm2[a]) continue;
        const o = (a * N + b) * H;
        const gm1 = linT(P.mlp1, gm2[a]); for (let c = 0; c < H; c++) gm1[c] *= r.mask1[o + c];
        const gm0 = linT(P.mlp0, gm1); for (let c = 0; c < H; c++) gm0[c] *= r.mask0[o + c];
        addInto(gmsg1[b] = gmsg1[b] || zeros(H), gm0);
        addInto(gmsg2[a] = gmsg2[a] || zeros(H), gm0);
        const key = a * N + b; addInto(ge.get(key) || (ge.set(key, zeros(H)), ge.get(key)), linT(P.me, gm0));
        addInto(gg, linT(P.mg, gm0));
      }
    }
    for (let k = 0; k < N; k++) {
      if (gmsg1[k]) addInto(gz[k] = gz[k] || zeros(2 * H), linT(P.m1, gmsg1[k]));
      if (gmsg2[k]) addInto(gz[k] = gz[k] || zeros(2 * H), linT(P.m2, gmsg2[k]));
    }
    return { gz, ge, gg };
  }

  // ---------- the graph ----------
  function build(M, T, inst, run, opts) {
    const N = inst.n, H = M.H, t = opts.t, tLo = opts.tLo;
    const nodes = {}, eS = [], eD = [], eW = [];
    const push = (s, d, w) => { if (Math.abs(w) > EPS) { eS.push(s); eD.push(d); eW.push(w); return w; } return 0; };
    const fieldsAt = M.fields.filter(f => f.stage === 'input' || f.stage === 'hint');
    const valuesAt = ts => { const v = {}; for (const f of fieldsAt) v[f.name] = f.stage === 'input' ? run.inputs[f.name] : run.hintsRead[ts][f.name]; return v; };
    // transcoder at every site in the window
    const TC = {};
    for (let ts = tLo; ts <= t; ts++) {
      const r = run.rec[ts]; TC[ts] = [];
      for (let i = 0; i < N; i++) {
        const tc = transcode(T, siteInput(r, i, H), r.hNew.subarray(i * H, (i + 1) * H)); TC[ts].push(tc);
        for (let q = 0; q < T.F; q++) if (tc.f[q] > 0) nodes[`F${ts}:${i}:${q}`] = { id: `F${ts}:${i}:${q}`, type: 'feature', t: ts, i, f: q, act: tc.f[q], pre: tc.pre[q] };
        let en = 0; for (let c = 0; c < H; c++) en += tc.err[c] ** 2;
        nodes[`E${ts}:${i}`] = { id: `E${ts}:${i}`, type: 'error', t: ts, i, norm: Math.sqrt(en) };
      }
      nodes[`B${ts}`] = { id: `B${ts}`, type: 'bias', t: ts };
    }
    const leaf = (id, info) => { if (!nodes[id]) nodes[id] = Object.assign({ id, type: 'leaf' }, info); return id; };
    const fromY = (ts, k, g, dst) => {       // effect through a new hidden state h'_{ts,k} = W_dec f + b_dec + err
      let acc = 0; const tc = TC[ts][k];
      for (let q = 0; q < T.F; q++) if (tc.f[q] > 0) acc += push(`F${ts}:${k}:${q}`, dst, tc.f[q] * dot(T.dec, g, q * H, 0, H));
      acc += push(`E${ts}:${k}`, dst, dot(tc.err, g));
      return acc;
    };
    function collect(ts, grads, dst, value) {
      const r = run.rec[ts], vals = valuesAt(ts); let acc = 0;
      const { gz, ge, gg } = grads;
      for (const f of fieldsAt) {
        const e0 = f.enc[0], v = vals[f.name], Kd = e0.din, cat = f.type === 'categorical';
        const encDot = (lay, g, off = 0) => { const out = zeros(lay.din); for (let c = 0; c < lay.din; c++) out[c] = dot(lay.w, g, c * H, off, H); return out; };
        if (f.loc === 'node' && f.type !== 'pointer') {
          for (let k = 0; k < N; k++) {
            if (!gz[k]) continue; const wd = encDot(e0, gz[k]);           // gz[k][0:H] is the node_fts cotangent
            let w = 0; if (cat) for (let c = 0; c < Kd; c++) w += v[k * Kd + c] * wd[c]; else w = v[k] * wd[0];
            if (Math.abs(w) > EPS) acc += push(leaf(`L${ts}:${f.name}:${k}`, { t: ts, probe: f.name, i: k, j: null, stage: f.stage, ftype: f.type, loc: f.loc, value: cat ? argmaxAt(v, k, Kd) : v[k] }), dst, w);
          }
        } else if (f.loc === 'node' && f.type === 'pointer') {                 // node pointers are encoded as edge features
          for (const [key, g] of ge) { const a = Math.floor(key / N), b = key % N, p = v[a * N + b]; if (!p) continue;
            const w = p * dot(e0.w, g, 0, 0, H);
            if (Math.abs(w) > EPS) acc += push(leaf(`L${ts}:${f.name}:${a}-${b}`, { t: ts, probe: f.name, i: a, j: b, stage: f.stage, ftype: f.type, loc: f.loc, value: p }), dst, w); }
        } else if (f.loc === 'edge' && f.type !== 'pointer') {
          for (const [key, g] of ge) { const a = Math.floor(key / N), b = key % N; let w = 0, val;
            if (cat) { const wd = encDot(e0, g); for (let c = 0; c < Kd; c++) w += v[key * Kd + c] * wd[c]; val = argmaxAt(v, key, Kd); } else { val = v[key]; w = val * dot(e0.w, g, 0, 0, H); }
            if (Math.abs(w) > EPS) acc += push(leaf(`L${ts}:${f.name}:${a}-${b}`, { t: ts, probe: f.name, i: a, j: b, stage: f.stage, ftype: f.type, loc: f.loc, value: val }), dst, w); }
        } else if (f.loc === 'edge' && f.type === 'pointer') {
          const e1 = f.enc[1];
          for (const [key, g] of ge) { const u = Math.floor(key / N), w2 = key % N; let m1 = 0, m2 = 0;
            for (let x = 0; x < N; x++) { m1 += v[(x * N + u) * N + w2]; m2 += v[(u * N + x) * N + w2]; } m1 /= N; m2 /= N;
            const w = m1 * dot(e0.w, g, 0, 0, H) + m2 * dot(e1.w, g, 0, 0, H);
            if (Math.abs(w) > EPS) acc += push(leaf(`L${ts}:${f.name}:${u}-${w2}`, { t: ts, probe: f.name, i: u, j: w2, stage: f.stage, ftype: f.type, loc: f.loc, value: +m1.toFixed(3) }), dst, w); }
        } else if (f.loc === 'graph' && nz(gg)) {
          const wd = encDot(e0, gg); let w = 0; if (cat) for (let c = 0; c < Kd; c++) w += v[c] * wd[c]; else w = v[0] * wd[0];
          if (Math.abs(w) > EPS) acc += push(leaf(`L${ts}:${f.name}:g`, { t: ts, probe: f.name, i: null, j: null, stage: f.stage, ftype: f.type, loc: f.loc, value: cat ? argmaxAt(v, 0, Kd) : v[0] }), dst, w);
        }
      }
      for (let k = 0; k < N; k++) {                                            // previous hidden state
        if (!gz[k]) continue; const g = gz[k].subarray(H, 2 * H); if (!nz(g)) continue;
        if (ts - 1 >= tLo) acc += fromY(ts - 1, k, g, dst);
        else if (ts > 0) { const w = dot(r.hidden, g, k * H, 0, H);
          if (Math.abs(w) > EPS) acc += push(leaf(`S${ts}:${k}`, { t: ts, probe: 'state', i: k, j: null, stage: 'state', value: null }), dst, w); }
      }
      if (grads.gHnew) for (let k = 0; k < N; k++) if (grads.gHnew[k]) acc += fromY(ts, k, grads.gHnew[k], dst);
      return value - acc;                                                      // bias edge: every constant term on the frozen path
    }
    // features: incoming edges
    for (let ts = tLo; ts <= t; ts++) {
      const r = run.rec[ts];
      for (let i = 0; i < N; i++) { const tc = TC[ts][i];
        for (let q = 0; q < T.F; q++) if (tc.f[q] > 0) {
          const gx = zeros(3 * H); for (let c = 0; c < 3 * H; c++) gx[c] = T.enc[c * T.F + q] / T.sd[c];
          const gz = new Array(N).fill(null), gagg = new Array(N).fill(null);
          gz[i] = gx.slice(0, 2 * H); gagg[i] = gx.slice(2 * H);
          const grads = backprop(M, N, r, { gz, gagg });
          const dst = `F${ts}:${i}:${q}`;
          push(`B${ts}`, dst, collect(ts, grads, dst, tc.pre[q]));
        } }
    }
    // targets at step t
    const r = run.rec[t];
    nodes['Bout'] = { id: 'Bout', type: 'bias', t: t + 1 };
    for (const tg of opts.targets) {
      const f = M.fields.find(x => x.name === tg.field);
      const gz = new Array(N).fill(null), gHnew = new Array(N).fill(null), ge = new Map();
      let value;
      if (f.type === 'pointer') {
        const sel = tg.sel, p4 = f.dec[3], gp1 = zeros(H), gp2 = zeros(H);
        for (let c = 0; c < H; c++) { gp1[c] = p4.w[c] * sel[c]; gp2[c] = p4.w[c] * (1 - sel[c]); }
        const a1 = linT(f.dec[0], gp1), a2 = linT(f.dec[1], gp2);
        const put = (k, a) => { addInto(gz[k] = gz[k] || zeros(2 * H), a.subarray(0, 2 * H)); addInto(gHnew[k] = gHnew[k] || zeros(H), a.subarray(2 * H)); };
        put(tg.i, a1); put(tg.j, a2);
        ge.set(tg.j * N + tg.i, linT(f.dec[2], gp2));
        value = r.logits[f.name][tg.i * N + tg.j];
      } else {
        const d0 = f.dec[0], c = tg.c || 0, g = zeros(3 * H); for (let k = 0; k < 3 * H; k++) g[k] = d0.w[k * d0.dout + c];
        gz[tg.i] = g.slice(0, 2 * H); gHnew[tg.i] = g.slice(2 * H);
        value = r.logits[f.name][tg.i * d0.dout + c];
      }
      nodes[tg.id] = Object.assign({ id: tg.id, type: 'logit', t: t + 1, logit: value }, tg);
      push('Bout', tg.id, collect(t, { gz, ge, gg: zeros(H), gHnew }, tg.id, value));
    }
    return { nodes, eS, eD, eW, TC };
  }
  const argmaxAt = (v, k, K) => { let b = 0; for (let c = 1; c < K; c++) if (v[k * K + c] > v[k * K + b]) b = c; return b; };

  // ---------- pruning (Circuit Tracing) and signed total effects ----------
  function prune(G, nodeKeep = 0.8, edgeKeep = 0.98) {
    const ids = Object.keys(G.nodes), pos = {}; ids.forEach((id, k) => pos[id] = k); const N = ids.length, E = G.eW.length;
    const S = new Int32Array(E), D = new Int32Array(E), A = new Float64Array(E);
    for (let e = 0; e < E; e++) { S[e] = pos[G.eS[e]]; D[e] = pos[G.eD[e]]; A[e] = Math.abs(G.eW[e]); }
    const colsum = new Float64Array(N); for (let e = 0; e < E; e++) colsum[D[e]] += A[e];
    const isLogit = ids.map(id => G.nodes[id].type === 'logit'), p = Float64Array.from(ids.map(id => G.nodes[id].prob || 0));
    const depth = Math.max(...ids.map(id => G.nodes[id].t || 0)) - Math.min(...ids.map(id => G.nodes[id].t || 0)) + 4;
    let x = Float64Array.from(p);
    for (let it = 0; it < 2 * depth + 4; it++) { const nx = Float64Array.from(p); for (let e = 0; e < E; e++) { const d = D[e]; if (colsum[d] > 0 && x[d] !== 0) nx[S[e]] += A[e] * x[d] / colsum[d]; } x = nx; }
    const infl = x.map((v, k) => v - p[k]); const total = infl.reduce((a, b) => a + b, 0);
    const order = Array.from({ length: N }, (_, k) => k).sort((a, b) => infl[b] - infl[a]);
    const keep = new Uint8Array(N); ids.forEach((id, k) => { if (isLogit[k] && (p[k] > 0.01 || G.nodes[id].always)) keep[k] = 1; }); if (N) keep[order[0]] = 1;
    let cum = 0; for (const k of order) { if (cum >= nodeKeep * total) break; keep[k] = 1; cum += infl[k]; }
    const col2 = new Float64Array(N); const keptE = [];
    for (let e = 0; e < E; e++) if (keep[S[e]] && keep[D[e]]) { keptE.push(e); col2[D[e]] += A[e]; }
    const scored = keptE.map(e => [A[e] / col2[D[e]] * (infl[D[e]] + (isLogit[D[e]] ? p[D[e]] : 0)), e]).sort((a, b) => b[0] - a[0]);
    const tot = scored.reduce((a, s) => a + s[0], 0); let c2 = 0; const fin = [];
    for (const [sc, e] of scored) { if (c2 >= edgeKeep * tot) break; fin.push(e); c2 += sc; }
    const used = new Uint8Array(N); fin.forEach(e => { used[S[e]] = 1; used[D[e]] = 1; });
    const prunedNodes = ids.filter((id, k) => keep[k] && (used[k] || isLogit[k]));
    // signed total effect of each node on each logit: r = e_l + W r, W[s,d] = w / act(s) for features
    const actOf = ids.map(id => G.nodes[id].type === 'feature' ? G.nodes[id].act : 1);
    const totalEffect = {};
    ids.forEach((lid, l) => { if (!isLogit[l]) return;
      let rv = new Float64Array(N); rv[l] = 1;
      for (let it = 0; it < 2 * depth + 4; it++) { const nr = new Float64Array(N); nr[l] = 1; for (let e = 0; e < E; e++) if (rv[D[e]] !== 0) nr[S[e]] += G.eW[e] / actOf[S[e]] * rv[D[e]]; rv = nr; }
      for (let k = 0; k < N; k++) if (k !== l && Math.abs(rv[k]) > 1e-9) (totalEffect[ids[k]] = totalEffect[ids[k]] || {})[lid] = rv[k]; });
    const influence = {}; ids.forEach((id, k) => influence[id] = infl[k]);
    return { nodes: prunedNodes, edges: fin.map(e => ({ s: G.eS[e], d: G.eD[e], w: G.eW[e] })), influence, keptInfluence: total > 0 ? cum / total : 1, totalEffect, nAll: N, eAll: E };
  }

  const sigmoid = v => 1 / (1 + Math.exp(-v));
  const softmax = arr => { const m = Math.max(...arr); const ex = arr.map(v => Math.exp(v - m)); const s = ex.reduce((a, b) => a + b, 0); return ex.map(v => v / s); };

  // targets for "explain node i at step t": the algorithm's main hint at node i, plus the pointer hint (i -> j) if any
  function targetsFor(MP, M, run, t, i, main, pointerField) {
    const N = run.rec[t].hT.length / (3 * M.H), r = run.rec[t], list = [];
    const f = M.fields.find(x => x.name === main), L = r.logits[main];
    if (f.type === 'categorical') { const C = f.dec[0].dout, row = Array.from(L.subarray(i * C, (i + 1) * C)), pr = softmax(row); let c = 0; for (let k = 1; k < C; k++) if (row[k] > row[c]) c = k;
      list.push({ id: `T:${main}:${i}:${c}`, field: main, kind: 'hint', ftype: f.type, i, c, prob: pr[c], always: true }); }
    else if (f.type === 'mask') { const pz = sigmoid(L[i]); list.push({ id: `T:${main}:${i}`, field: main, kind: 'hint', ftype: f.type, i, prob: Math.max(pz, 1 - pz), p1: pz, always: true }); }
    else if (f.type === 'mask_one') { const pr = softmax(Array.from(L)); list.push({ id: `T:${main}:${i}`, field: main, kind: 'hint', ftype: f.type, i, prob: Math.max(pr[i], 0.02), p1: pr[i], always: true }); }
    else list.push({ id: `T:${main}:${i}`, field: main, kind: 'hint', ftype: f.type, i, prob: 1, value: L[i], always: true });
    if (pointerField) {
      const pf = M.fields.find(x => x.name === pointerField), row = Array.from(r.logits[pointerField].subarray(i * N, (i + 1) * N)), pr = softmax(row);
      let am = 0; for (let j = 1; j < N; j++) if (row[j] > row[am]) am = j;
      for (let j = 0; j < N; j++) if (pr[j] >= 0.01 || j === am)
        list.push({ id: `P:${pointerField}:${i}->${j}`, field: pointerField, kind: 'pointer', ftype: 'pointer', i, j, prob: pr[j], always: j === am, sel: MP.pointerSel(pf, r, i, j, N, M.H) });
    }
    return list;
  }

  function explain(MP, M, T, inst, opts) {
    const t = opts.t, tLo = Math.max(0, t - (opts.window || 3) + 1);
    const run = MP.runRecord(M, inst, { record: { from: tLo, to: t }, stopAt: t });
    const targets = targetsFor(MP, M, run, t, opts.i, opts.main, opts.pointerField);
    const G = build(M, T, inst, run, { t, tLo, targets });
    const Pr = prune(G, opts.nodeKeep, opts.edgeKeep);
    return { G, Pr, run, targets, tLo };
  }

  // clamp feature (t0, i0, q) to factor x its activation; re-run the real model (its own hint feedback) and report target changes
  function intervene(MP, M, T, inst, ex, t0, i0, q, factor) {
    const H = M.H, act = ex.G.nodes[`F${t0}:${i0}:${q}`].act, dy = zeros(H);
    for (let c = 0; c < H; c++) dy[c] = (factor - 1) * act * T.dec[q * H + c];
    const t = ex.run.rec.length - 1;
    const alt = MP.runRecord(M, inst, { patch: { t: t0, i: i0, dy }, record: { from: t0, to: t }, stopAt: t });
    let flips = 0, decisions = 0;
    for (let ts = t0 + 1; ts <= t; ts++) { const A = alt.rec[ts], B = ex.run.rec[ts];
      for (let k = 0; k < A.mask0.length; k++) { decisions += 2; if (A.mask0[k] !== B.mask0[k]) flips++; if (A.mask1[k] !== B.mask1[k]) flips++; }
      for (let k = 0; k < A.route.length; k++) { decisions++; if (A.route[k] !== B.route[k]) flips++; } }
    const val = (rr, tg) => { const L = rr.logits[tg.field]; if (tg.kind === 'pointer') return L[tg.i * inst.n + tg.j]; const f = M.fields.find(x => x.name === tg.field); return L[tg.i * f.dec[0].dout + (tg.c || 0)]; };
    const rows = ex.targets.map(tg => { const te = (ex.Pr.totalEffect[`F${t0}:${i0}:${q}`] || {})[tg.id] || 0;
      return { id: tg.id, base: val(ex.run.rec[t], tg), actual: val(alt.rec[t], tg), predicted: val(ex.run.rec[t], tg) + (factor - 1) * act * te }; });
    return { t0, i0, q, act, factor, rows, flips, decisions };
  }

  return { prepTranscoder, transcode, backprop, build, prune, explain, intervene, targetsFor, sigmoid, softmax };
});
