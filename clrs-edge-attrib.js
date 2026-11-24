/* Attribution graphs from the EDGE-SITE transcoders trained for the 12 CLRS-30 MPNNs (clrs-transcoder-out/<algo>/transcoder.npz).
 *
 * The site. One MPNN step reads, for every ordered pair (sender i -> receiver j) of the n^2 pairs (the MPNN overrides the adjacency
 * with all ones), the pre-ReLU message input  s_ij = M2 z_i + M1 z_j + Me e_ij + Mg g + b  and produces the message
 *   y_ij = L2 ReLU(L1 ReLU(s_ij) + b1) + b2.   The transcoder replaces that map by a sparse code (TopK, k = 8 of D = 1280):
 *   a = relu(topk_k((x - b_pre) W_enc + b_enc)),  x = (s - s_mu) / s_scale,   y_hat = (a W_dec + b_dec) * y_scale + y_mu.
 * The error node eps_ij = y_ij - y_hat_ij is added back, so the replacement model reproduces the real network exactly.
 *
 * The graph. With the max-aggregation winners, the LayerNorm sigma and the ReLU pattern of the node update frozen at their values
 * on this input, every quantity below is a linear function of the nodes one step earlier:
 *   m_j[c] = y_{i*,j}[c]  (i* = winner of channel c at receiver j)
 *   x_j    = O1 z_j + O2 m_j + bo,   h'_j = diag(gamma/sigma_j) P Du_j x_j + beta        (P = I - 11^T/H, Du = diag(x_j > 0))
 *   pre_f(i,j) = (1/s_scale) w_f . (M2 z_i + M1 z_j + Me e_ij + Mg g + b) + const_f
 * so the pre-activation of every latent, and every target logit, equals the sum of its incoming edges from
 *   - latents at pairs (i*, j) of the step before (through the winner mask and W_dec),
 *   - error nodes eps_ij, encoder leaves (the encoded inputs and hints), the "earlier state" leaf, and one bias node.
 * Hidden state also reaches later steps without passing through any latent (O1 z_j), so such skip edges exist and are exact.
 * The bias is computed from the constant terms and is NOT chosen to make the sums work: verify() checks every sum.
 * Works in browsers, Web Workers and Node. Needs mpnn.js (model + recorded forward pass) and attrib.js (targetsFor). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ClrsEdge = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------- the transcoder (maths of the training notebook: tc_encode / tc_decode) ----------
  function prepTC(man, buf) {
    const F = buf instanceof Float32Array ? buf : new Float32Array(buf.buffer ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) : buf);
    const o = man.offsets, take = n => Float64Array.from(F.subarray(o[n][0], o[n][0] + o[n][1]));
    const H = man.H, D = man.D, sc = take('scales'), W_enc = take('W_enc'), WencT = new Float64Array(D * H);
    for (let c = 0; c < H; c++) for (let d = 0; d < D; d++) WencT[d * H + c] = W_enc[c * D + d];       // column f of W_enc, contiguous
    return { H, D, k: man.k, s_mu: take('s_mu'), y_mu: take('y_mu'), b_pre: take('b_pre'), b_dec: take('b_dec'), b_enc: take('b_enc'),
             s_scale: sc[0], y_scale: sc[1], W_enc, WencT, W_dec: take('W_dec') };
  }
  // one row s (H numbers, starting at sOff) -> pre-activations (D) and the active set: the k largest if positive; ties at the threshold are all kept
  function encodeRow(T, s, sOff, pre, tmp) {
    const { H, D, W_enc, b_enc, b_pre, s_mu, s_scale, k } = T;
    pre.set(b_enc);
    for (let c = 0; c < H; c++) {
      const u = (s[sOff + c] - s_mu[c]) / s_scale - b_pre[c]; if (u === 0) continue;
      const r = c * D; for (let d = 0; d < D; d++) pre[d] += u * W_enc[r + d];
    }
    tmp.set(pre); tmp.sort(); const thr = tmp[D - k], idx = [];
    for (let d = 0; d < D; d++) if (pre[d] >= thr && pre[d] > 0) idx.push(d);
    return idx;                                                             // activation of latent d is pre[d] (relu(topk(pre)))
  }
  function decodeRow(T, idx, pre, out) {
    const { H, W_dec, b_dec, y_scale, y_mu } = T;
    for (let c = 0; c < H; c++) out[c] = b_dec[c];
    for (const f of idx) { const a = pre[f], r = f * H; for (let c = 0; c < H; c++) out[c] += a * W_dec[r + c]; }
    for (let c = 0; c < H; c++) out[c] = out[c] * y_scale + y_mu[c];
    return out;
  }


  // ---------- small linear algebra ----------
  const zeros = n => new Float64Array(n);
  const dot = (a, b, ao = 0, bo = 0, n = a.length) => { let s = 0; for (let k = 0; k < n; k++) s += a[ao + k] * b[bo + k]; return s; };
  function linT(p, g) {                       // cotangent on a layer's outputs -> its inputs (no bias); p.w is row-major din x dout
    const { din, dout, w } = p, out = zeros(din);
    for (let i = 0; i < din; i++) { let s = 0; const r = i * dout; for (let o = 0; o < dout; o++) s += w[r + o] * g[o]; out[i] = s; }
    return out;
  }
  function linTz(p, g) {                    // same as linT; zero entries of g are skipped
    const { din, dout, w } = p, out = zeros(din), nz = []; for (let o = 0; o < dout; o++) if (g[o] !== 0) nz.push(o);
    if (nz.length === dout) return linT(p, g);
    for (let i = 0; i < din; i++) { let s = 0; const r = i * dout; for (let q = 0; q < nz.length; q++) s += w[r + nz[q]] * g[nz[q]]; out[i] = s; }
    return out;
  }
  const EPS = 1e-12;

  // ---------- the graph ----------
  // Nodes (integer index internally; string ids only in the packed result): latents F{t}:{pair}:{f} (pair = sender*N + receiver), error nodes E{t}:{pair},
  // encoder leaves L..., earlier-state leaves S{t}:{node}, one bias node B{t} per step (+ Bout for the predictions) and the target logits.
  // opts: {t, i, main, pointerField, window}. AT = attrib.js (only targetsFor is used).
  function explain(MP, AT, M, T, inst, opts) {
    const t = opts.t, tLo = Math.max(0, t - (opts.window || 3) + 1), N = inst.n, H = M.H, P = M.proc, D = T.D, ys = T.y_scale;
    const run = MP.runRecord(M, inst, { record: { from: tLo, to: t }, stopAt: t });
    const targets = AT.targetsFor(MP, M, run, t, opts.i, opts.main, opts.pointerField);
    const fieldsAt = M.fields.filter(f => f.stage === 'input' || f.stage === 'hint');
    const valuesAt = ts => { const v = {}; for (const f of fieldsAt) v[f.name] = f.stage === 'input' ? run.inputs[f.name] : run.hintsRead[ts][f.name]; return v; };
    const beta = P.lnOffset, gamma = P.lnScale;

    // ---- per step: codes of every pair, error nodes, frozen winners and node-update pattern
    const St = {}, pre = zeros(D), tmp = zeros(D), yh = zeros(H);
    const targetNodes = new Set(); for (const tg of targets) { targetNodes.add(tg.i); if (tg.j !== undefined && tg.j !== null) targetNodes.add(tg.j); }
    for (let ts = tLo; ts <= t; ts++) {
      const r = run.rec[ts], S = { r, lat: new Array(N * N).fill(null), eps: zeros(N * N * H), Du: new Uint8Array(N * H), sig: zeros(N), winCh: new Array(N * N), hidden: r.hidden, vals: valuesAt(ts) };
      for (let p = 0; p < N * N; p++) {
        if (ts === t && !opts.full && !targetNodes.has(p % N)) continue;                 // at the last step only latents written to a target node can reach the targets
        const ids = encodeRow(T, r.s, p * H, pre, tmp); decodeRow(T, ids, pre, yh);
        S.lat[p] = { idx: ids, a: ids.map(f => pre[f]) };
        for (let c = 0; c < H; c++) S.eps[p * H + c] = r.y[p * H + c] - yh[c];
      }
      // frozen node update: x = O1 z + O2 m + bo, u = relu(x), sigma from u
      const h1 = MP.linRows(r.z, N, P.o1), h2 = MP.linRows(r.agg, N, P.o2);
      for (let k = 0; k < N; k++) {
        let mean = 0; for (let c = 0; c < H; c++) { const x = h1[k * H + c] + h2[k * H + c]; if (x > 0) { S.Du[k * H + c] = 1; mean += x; } }
        mean /= H; let vr = 0; for (let c = 0; c < H; c++) { const x = h1[k * H + c] + h2[k * H + c], u = x > 0 ? x : 0; vr += (u - mean) ** 2; }
        S.sig[k] = Math.sqrt(vr / H + 1e-5);
      }
      for (let k = 0; k < N; k++) { const lists = Array.from({ length: N }, () => []); for (let c = 0; c < H; c++) lists[r.route[k * H + c]].push(c); for (let i = 0; i < N; i++) S.winCh[k * N + i] = Int32Array.from(lists[i]); }
      St[ts] = S;
    }

    // ---- node and edge storage
    const nodes = [], nid = new Map();
    const nodeOf = (key, mk) => { let v = nid.get(key); if (v === undefined) { v = nodes.length; nodes.push(Object.assign({ id: key }, mk())); nid.set(key, v); } return v; };
    let cap = 1 << 20, eS = new Int32Array(cap), eD = new Int32Array(cap), eW = new Float64Array(cap), E = 0;
    const push = (s, d, w, acc) => { acc.sum += w; if (Math.abs(w) <= EPS) return; if (E === cap) { cap *= 2; const a = new Int32Array(cap), b = new Int32Array(cap), c = new Float64Array(cap); a.set(eS); b.set(eD); c.set(eW); eS = a; eD = b; eW = c; } eS[E] = s; eD[E] = d; eW[E] = w; E++; };
    const featId = {}, errId = {};                                                  // node index of every latent / error node, by array position (no string keys on the hot path)
    const featNode = (ts, p, q) => { const a = featId[ts] || (featId[ts] = new Array(N * N)); let v = a[p]; if (!v) v = a[p] = new Int32Array(St[ts].lat[p].idx.length).fill(-1);
      if (v[q] < 0) { const L = St[ts].lat[p], f = L.idx[q]; v[q] = nodeOf(`F${ts}:${p}:${f}`, () => ({ type: 'feature', t: ts, s: Math.floor(p / N), i: p % N, f, act: L.a[q], pre: L.a[q], pair: p })); } return v[q]; };
    const errNode = (ts, p) => { const a = errId[ts] || (errId[ts] = new Int32Array(N * N).fill(-1)); if (a[p] < 0) a[p] = nodeOf(`E${ts}:${p}`, () => { let n2 = 0; const e = St[ts].eps; for (let c = 0; c < H; c++) n2 += e[p * H + c] ** 2; return { type: 'error', t: ts, s: Math.floor(p / N), i: p % N, norm: Math.sqrt(n2), pair: p }; }); return a[p]; };
    const biasNode = ts => nodeOf(ts > t ? 'Bout' : `B${ts}`, () => ({ type: 'bias', t: ts }));
    const argmaxAt = (v, k, K) => { let b = 0; for (let c = 1; c < K; c++) if (v[k * K + c] > v[k * K + b]) b = c; return b; };
    const leaf = (key, info, w, dst, acc) => { if (Math.abs(w) > EPS) push(nodeOf(key, () => Object.assign({ type: 'leaf' }, info)), dst, w, acc); else acc.sum += w; };

    // ---- encoder leaves: cotangents on node features (gz: node -> 2H, first H used), on edge features (ge: pair -> H) and on the graph feature (gg)
    function leafEdges(ts, gz, ge, gg, dst, acc) {
      const vals = St[ts].vals;
      for (const f of fieldsAt) {
        const e0 = f.enc[0], v = vals[f.name], Kd = e0.din, cat = f.type === 'categorical';
        const encDot = (lay, g) => { const out = zeros(lay.din); for (let c = 0; c < lay.din; c++) out[c] = dot(lay.w, g, c * H, 0, H); return out; };
        if (f.loc === 'node' && f.type !== 'pointer') {
          if (!gz) continue;
          for (const [k, g] of gz) {
            acc.bias += dot(e0.b, g, 0, 0, H); const wd = encDot(e0, g); let w = 0; if (cat) for (let c = 0; c < Kd; c++) w += v[k * Kd + c] * wd[c]; else w = v[k] * wd[0];
            leaf(`L${ts}:${f.name}:${k}`, { t: ts, probe: f.name, i: k, j: null, stage: f.stage, ftype: f.type, loc: f.loc, value: cat ? argmaxAt(v, k, Kd) : v[k] }, w, dst, acc);
          }
        } else if (f.loc === 'node' && f.type === 'pointer') {                 // node pointers are encoded as edge features
          if (!ge) continue;
          for (const [key, g] of ge) { const a = Math.floor(key / N), b = key % N, pv = v[a * N + b]; acc.bias += dot(e0.b, g, 0, 0, H);
            leaf(`L${ts}:${f.name}:${a}-${b}`, { t: ts, probe: f.name, i: a, j: b, stage: f.stage, ftype: f.type, loc: f.loc, value: pv }, pv * dot(e0.w, g, 0, 0, H), dst, acc); }
        } else if (f.loc === 'edge' && f.type !== 'pointer') {
          if (!ge) continue;
          for (const [key, g] of ge) { const a = Math.floor(key / N), b = key % N; let w = 0, val; acc.bias += dot(e0.b, g, 0, 0, H);
            if (cat) { const wd = encDot(e0, g); for (let c = 0; c < Kd; c++) w += v[key * Kd + c] * wd[c]; val = argmaxAt(v, key, Kd); } else { val = v[key]; w = val * dot(e0.w, g, 0, 0, H); }
            leaf(`L${ts}:${f.name}:${a}-${b}`, { t: ts, probe: f.name, i: a, j: b, stage: f.stage, ftype: f.type, loc: f.loc, value: val }, w, dst, acc); }
        } else if (f.loc === 'edge' && f.type === 'pointer') {
          if (!ge) continue;
          const e1 = f.enc[1];
          for (const [key, g] of ge) { const u = Math.floor(key / N), w2 = key % N; let m1 = 0, m2 = 0;
            for (let x = 0; x < N; x++) { m1 += v[(x * N + u) * N + w2]; m2 += v[(u * N + x) * N + w2]; } m1 /= N; m2 /= N;
            acc.bias += dot(e0.b, g, 0, 0, H) + dot(e1.b, g, 0, 0, H);
            leaf(`L${ts}:${f.name}:${u}-${w2}`, { t: ts, probe: f.name, i: u, j: w2, stage: f.stage, ftype: f.type, loc: f.loc, value: +m1.toFixed(3) }, m1 * dot(e0.w, g, 0, 0, H) + m2 * dot(e1.w, g, 0, 0, H), dst, acc); }
        } else if (f.loc === 'graph' && gg) {
          acc.bias += dot(e0.b, gg, 0, 0, H); const wd = encDot(e0, gg); let w = 0; if (cat) for (let c = 0; c < Kd; c++) w += v[c] * wd[c]; else w = v[0] * wd[0];
          leaf(`L${ts}:${f.name}:g`, { t: ts, probe: f.name, i: null, j: null, stage: f.stage, ftype: f.type, loc: f.loc, value: cat ? argmaxAt(v, 0, Kd) : v[0] }, w, dst, acc);
        }
      }
    }

    // ---- hidden state entering step ts at node k (cotangent g, H): either produced by step ts-1 (expand) or the "earlier state" leaf
    function fromHidden(ts, k, g, dst, acc) {
      if (ts - 1 >= tLo) expandHidden(ts - 1, k, g, dst, acc);
      else if (ts > 0) leaf(`S${ts}:${k}`, { t: ts, probe: 'state', i: k, j: null, stage: 'state', value: null }, dot(St[ts].hidden, g, k * H, 0, H), dst, acc);
    }
    // cotangent gh on the hidden state h'_k produced by step ts:  h' = diag(gamma/sigma) P Du x + beta,  x = O1 z + O2 m + bo,  m[c] = y_{winner(c),k}[c]
    function expandHidden(ts, k, gh, dst, acc) {
      const S = St[ts], v = zeros(H); let mean = 0, bb = 0;
      for (let c = 0; c < H; c++) { v[c] = gh[c] * gamma[c] / S.sig[k]; mean += v[c]; bb += beta[c] * gh[c]; } mean /= H;
      const gx = zeros(H); for (let c = 0; c < H; c++) if (S.Du[k * H + c]) { gx[c] = v[c] - mean; bb += gx[c] * (P.o1.b[c] + P.o2.b[c]); }
      acc.bias += bb;
      const gz = linTz(P.o1, gx), gm = linTz(P.o2, gx);
      for (let i = 0; i < N; i++) {
        const ch = S.winCh[k * N + i]; if (!ch.length) continue;                 // a sender that wins no channel has no effect
        const p = i * N + k, L = S.lat[p]; let cst = 0, er = 0;
        for (const c of ch) { cst += gm[c] * (T.b_dec[c] * ys + T.y_mu[c]); er += gm[c] * S.eps[p * H + c]; }
        acc.bias += cst; push(errNode(ts, p), dst, er, acc);
        for (let q = 0; q < L.idx.length; q++) { const f = L.idx[q]; let w = 0; for (const c of ch) w += gm[c] * T.W_dec[f * H + c]; push(featNode(ts, p, q), dst, w * L.a[q] * ys, acc); }
      }
      leafEdges(ts, new Map([[k, gz]]), null, null, dst, acc);
      fromHidden(ts, k, gz.subarray(H, 2 * H), dst, acc);
    }

    // ---- latent destinations: pre_f = (1/s_scale) w_f . s_ij + const
    const checks = { n: 0, maxAbs: 0, maxRel: 0, worst: null };
    const closeOut = (dst, value, acc, ts, label) => {
      push(biasNode(ts), dst, acc.bias, acc); const err = Math.abs(value - acc.sum), rel = err / Math.max(1e-9, Math.abs(value));
      checks.n++; if (err > checks.maxAbs) { checks.maxAbs = err; checks.worst = label; } if (rel > checks.maxRel && Math.abs(value) > 1e-3) checks.maxRel = rel;
    };
    const latCache = new Map();
    const mb = new Float64Array(H); for (let c = 0; c < H; c++) mb[c] = P.m1.b[c] + P.m2.b[c] + P.me.b[c] + P.mg.b[c];
    for (let ts = tLo; ts <= t; ts++) {
      for (let p = 0; p < N * N; p++) {
        const j2 = p % N, i2 = Math.floor(p / N);
        const L = St[ts].lat[p]; if (!L) continue;                                    // at the last step only latents that can reach the targets exist
        for (let q = 0; q < L.idx.length; q++) {
          const f = L.idx[q], d = featNode(ts, p, q), acc = { sum: 0, bias: 0 };
          let cf = latCache.get(f);                                                       // everything that depends only on the latent, not on the pair
          if (!cf) {
            const qv = zeros(H); let c1 = T.b_enc[f], bq = 0;
            for (let c = 0; c < H; c++) { qv[c] = T.WencT[f * H + c] / T.s_scale; c1 += (-T.s_mu[c] / T.s_scale - T.b_pre[c]) * T.WencT[f * H + c]; bq += qv[c] * mb[c]; }
            cf = { bias: c1 + bq, gzi: linT(P.m2, qv), gzj: linT(P.m1, qv), ge: linT(P.me, qv), gg: linT(P.mg, qv) }; latCache.set(f, cf);
          }
          acc.bias += cf.bias;
          const gz = new Map();
          if (i2 === j2) { const g = new Float64Array(2 * H); for (let c = 0; c < 2 * H; c++) g[c] = cf.gzi[c] + cf.gzj[c]; gz.set(i2, g); } else { gz.set(i2, cf.gzi); gz.set(j2, cf.gzj); }
          leafEdges(ts, gz, new Map([[p, cf.ge]]), cf.gg, d, acc);
          for (const [k, g] of gz) fromHidden(ts, k, g.subarray(H, 2 * H), d, acc);
          closeOut(d, L.a[q], acc, ts, `F${ts}:${i2}->${j2}:f${f}`);
        }
      }
    }

    // ---- target logits (decoders read [node_fts, h, h'] of their nodes)
    const r = St[t].r; biasNode(t + 1);
    for (const tg of targets) {
      const f = M.fields.find(x => x.name === tg.field), gz = new Map(), gH = new Map(), ge = new Map(), acc = { sum: 0, bias: 0 }; let value;
      const add = (m, k, a, n) => { const cur = m.get(k) || zeros(n); for (let c = 0; c < n; c++) cur[c] += a[c]; m.set(k, cur); };
      const put = (k, a) => { add(gz, k, a.subarray(0, 2 * H), 2 * H); add(gH, k, a.subarray(2 * H), H); };
      if (f.type === 'pointer') {
        const sel = tg.sel, p4 = f.dec[3], gp1 = zeros(H), gp2 = zeros(H);
        for (let c = 0; c < H; c++) { gp1[c] = p4.w[c] * sel[c]; gp2[c] = p4.w[c] * (1 - sel[c]); acc.bias += gp1[c] * f.dec[0].b[c] + gp2[c] * (f.dec[1].b[c] + f.dec[2].b[c]); }
        acc.bias += p4.b[0];
        put(tg.i, linT(f.dec[0], gp1)); put(tg.j, linT(f.dec[1], gp2)); ge.set(tg.j * N + tg.i, linT(f.dec[2], gp2));
        value = r.logits[f.name][tg.i * N + tg.j];
      } else {
        const d0 = f.dec[0], c = tg.c || 0, g = zeros(3 * H); for (let k = 0; k < 3 * H; k++) g[k] = d0.w[k * d0.dout + c];
        put(tg.i, g); acc.bias += d0.b[c]; value = r.logits[f.name][tg.i * d0.dout + c];
      }
      const d = nodeOf(tg.id, () => Object.assign({ type: 'logit', t: t + 1, logit: value }, tg, { sel: undefined }));
      leafEdges(t, gz, ge, null, d, acc);
      for (const [k, g] of gz) fromHidden(t, k, g.subarray(H, 2 * H), d, acc);
      for (const [k, g] of gH) expandHidden(t, k, g, d, acc);
      closeOut(d, value, acc, t + 1, tg.id);
    }
    return { nodes, nid, eS: eS.subarray(0, E), eD: eD.subarray(0, E), eW: eW.subarray(0, E), E, targets, run, tLo, t, St, T, checks, N, inst };
  }


  // ---------- pruning (Circuit Tracing recipe) and signed total effects ----------
  // Every edge goes from a node of an earlier step to a node of a later step (leaves, errors and bias count as the step of their destination),
  // so one pass over the edges grouped by destination step, latest first, replaces the iteration of the node-site version.
  function prune(ex, nodeKeep = 0.8, edgeKeep = 0.98) {
    const { nodes, eS, eD, eW, E, tLo, t } = ex, n = nodes.length, nSteps = t + 2 - tLo;
    const A = new Float64Array(E), colsum = new Float64Array(n), start = new Int32Array(nSteps + 1);
    for (let e = 0; e < E; e++) { A[e] = Math.abs(eW[e]); colsum[eD[e]] += A[e]; start[nodes[eD[e]].t - tLo + 1]++; }
    for (let k = 0; k < nSteps; k++) start[k + 1] += start[k];
    const fill = start.slice(0, nSteps), order = new Int32Array(E);
    for (let e = 0; e < E; e++) order[fill[nodes[eD[e]].t - tLo]++] = e;
    const isLogit = nodes.map(nd => nd.type === 'logit'), pr = Float64Array.from(nodes.map(nd => nd.prob || 0));
    const x = Float64Array.from(pr);
    for (let st = nSteps - 1; st >= 0; st--) for (let q = start[st]; q < start[st + 1]; q++) { const e = order[q], d = eD[e]; if (colsum[d] > 0 && x[d] !== 0) x[eS[e]] += A[e] * x[d] / colsum[d]; }
    const infl = x.map((v, k) => v - pr[k]); let total = 0; for (const v of infl) total += v;
    const rank = Array.from({ length: n }, (_, k) => k).sort((a, b) => infl[b] - infl[a]);
    const keep = new Uint8Array(n); nodes.forEach((nd, k) => { if (isLogit[k] && (pr[k] > 0.01 || nd.always)) keep[k] = 1; }); if (n) keep[rank[0]] = 1;
    let cum = 0; for (const k of rank) { if (cum >= nodeKeep * total) break; keep[k] = 1; cum += infl[k]; }
    const col2 = new Float64Array(n), keptE = [];
    for (let e = 0; e < E; e++) if (keep[eS[e]] && keep[eD[e]]) { keptE.push(e); col2[eD[e]] += A[e]; }
    const scored = keptE.map(e => [A[e] / col2[eD[e]] * (infl[eD[e]] + (isLogit[eD[e]] ? pr[eD[e]] : 0)), e]).sort((a, b) => b[0] - a[0]);
    let tot = 0; for (const q of scored) tot += q[0]; let c2 = 0; const fin = [];
    for (const [sc, e] of scored) { if (c2 >= edgeKeep * tot) break; fin.push(e); c2 += sc; }
    const used = new Uint8Array(n); fin.forEach(e => { used[eS[e]] = 1; used[eD[e]] = 1; });
    const prunedNodes = []; for (let k = 0; k < n; k++) if (keep[k] && (used[k] || isLogit[k])) prunedNodes.push(k);
    // total effect of one unit of activation of every node on each logit: r = e_l + sum over edges (w / act(src)) r[dst]
    const act = Float64Array.from(nodes.map(nd => nd.type === 'feature' ? nd.act : 1)), effects = new Map();
    nodes.forEach((nd, l) => { if (!isLogit[l]) return; const r = new Float64Array(n); r[l] = 1;
      for (let st = nSteps - 1; st >= 0; st--) for (let q = start[st]; q < start[st + 1]; q++) { const e = order[q], d = eD[e]; if (r[d] !== 0) r[eS[e]] += eW[e] / act[eS[e]] * r[d]; }
      effects.set(l, r); });
    return { prunedNodes, fin, infl, keptInfluence: total > 0 ? cum / total : 1, effects, A };
  }

  // the part of the result the page needs: pruned graph, the strongest edges for the "all" view, and facts about each shown latent in THIS graph
  function pack(ex, pr, inst) {
    const { nodes, eS, eD, eW, E, St, N, tLo, t } = ex;
    const topN = 6000; let topE = [];
    if (E <= topN) { for (let e = 0; e < E; e++) topE.push(e); }
    else {                                                                              // threshold from a log-scale histogram (linear time), then sort only the candidates
      let mx = 0; for (let e = 0; e < E; e++) if (pr.A[e] > mx) mx = pr.A[e];
      const B = 4096, lo = Math.log(mx) - 40, hist = new Int32Array(B + 1), bin = a => a <= 0 ? 0 : Math.max(0, Math.min(B, Math.floor((Math.log(a) - lo) / 40 * B)));
      for (let e = 0; e < E; e++) hist[bin(pr.A[e])]++;
      let cnt = 0, b = B; while (b > 0 && cnt + hist[b] < topN) { cnt += hist[b]; b--; }
      for (let e = 0; e < E; e++) if (bin(pr.A[e]) >= b) topE.push(e);
      topE.sort((a, b2) => pr.A[b2] - pr.A[a]); topE = topE.slice(0, topN);
    }
    topE.sort((a, b) => pr.A[b] - pr.A[a]);
    const send = new Set(pr.prunedNodes); for (const e of topE) { send.add(eS[e]); send.add(eD[e]); }
    nodes.forEach((nd, k) => { if (nd.type === 'logit') send.add(k); });
    const where = {};                                                                  // step -> latent -> [pairs where it is active at that step]
    for (let ts = tLo; ts <= t; ts++) { const m = new Map(); St[ts].lat.forEach((L, p) => L && L.idx.forEach(f => { if (!m.has(f)) m.set(f, []); m.get(f).push(p); })); where[ts] = m; }
    const adjOf = (a, b) => (inst.adj && inst.adj[a] && inst.adj[a][b] > 0);
    const ids = [...send], out = [], influence = {}, totalEffect = {};
    for (const k of ids) {
      const nd = Object.assign({}, nodes[k]); delete nd.sel;
      if (nd.type === 'feature') {
        nd.ptype = nd.s === nd.i ? 'self' : (adjOf(nd.s, nd.i) ? 'edge' : 'nonedge');
        const ps = where[nd.t].get(nd.f) || [], real = ps.filter(p => Math.floor(p / N) !== p % N && adjOf(Math.floor(p / N), p % N)).length;
        nd.fires = { n: ps.length, real, self: ps.filter(p => Math.floor(p / N) === p % N).length, pairs: ps.slice(0, 8).map(p => [Math.floor(p / N), p % N]) };
      }
      out.push(nd); influence[nd.id] = pr.infl[k];
      const te = {}; for (const [l, r] of pr.effects) if (l !== k && Math.abs(r[k]) > 1e-9) te[nodes[l].id] = r[k]; if (Object.keys(te).length) totalEffect[nd.id] = te;
    }
    const id = k => nodes[k].id;
    return { nodes: out, pruned: { nodes: pr.prunedNodes.map(id), edges: pr.fin.map(e => ({ s: id(eS[e]), d: id(eD[e]), w: eW[e] })) },
             top: topE.map(e => ({ s: id(eS[e]), d: id(eD[e]), w: eW[e] })), influence, totalEffect, keptInfluence: pr.keptInfluence,
             nAll: nodes.length, eAll: E, targets: ex.targets.map(tg => { const o = Object.assign({}, tg); delete o.sel; return o; }), tLo, checks: ex.checks };
  }

  // clamp latent (t0, pair, f) to factor x its activation: add (factor-1) a W_dec[f] y_scale to that message, re-run the REAL model (its own hint feedback),
  // and compare the change of every target logit with the graph's prediction (total effect). Also counts downstream winner / TopK decisions that changed.
  function intervene(MP, M, T, inst, ex, pr, nodeId, factor) {
    const H = M.H, N = inst.n, k0 = ex.nid.get(nodeId), nd = ex.nodes[k0], t0 = nd.t, t = ex.t, dy = zeros(H);
    for (let c = 0; c < H; c++) dy[c] = (factor - 1) * nd.act * T.W_dec[nd.f * H + c] * T.y_scale;
    const alt = MP.runRecord(M, inst, { patchY: { t: t0, i: nd.s, j: nd.i, dy }, record: { from: t0, to: t }, stopAt: t });
    // the graph treats the hints that are fed back as fixed inputs; the real model re-decodes them after the clamp. Run it both ways so the two effects can be told apart.
    const fixed = MP.runRecord(M, inst, { patchY: { t: t0, i: nd.s, j: nd.i, dy }, hints: ex.run.hintsRead, record: { from: t0, to: t }, stopAt: t });
    let flips = 0, decisions = 0; const pre = zeros(T.D), tmp = zeros(T.D);
    for (let ts = t0; ts <= t; ts++) { const A = alt.rec[ts], B = ex.run.rec[ts];
      for (let q = 0; q < A.route.length; q++) { decisions++; if (A.route[q] !== B.route[q]) flips++; }
      if (ts > t0) for (let p = 0; p < N * N; p++) { const L = ex.St[ts].lat[p]; if (!L) continue; const a = encodeRow(T, A.s, p * H, pre, tmp), b = L.idx; decisions += T.k; if (a.length !== b.length || a.some((v, q) => v !== b[q])) flips++; } }
    const val = (rr, tg) => { const L = rr.logits[tg.field]; if (tg.kind === 'pointer') return L[tg.i * N + tg.j]; const f = M.fields.find(x => x.name === tg.field); return L[tg.i * f.dec[0].dout + (tg.c || 0)]; };
    const rows = ex.targets.map(tg => { const l = ex.nid.get(tg.id), te = pr.effects.get(l)[k0], base = val(ex.run.rec[t], tg);
      return { id: tg.id, base, actual: val(alt.rec[t], tg), actualFixed: val(fixed.rec[t], tg), predicted: base + (factor - 1) * nd.act * te }; });
    return { t0, i0: nd.i, s0: nd.s, f: nd.f, act: nd.act, factor, rows, flips, decisions };
  }

  return { prepTC, encodeRow, decodeRow, explain, prune, pack, intervene };
});
