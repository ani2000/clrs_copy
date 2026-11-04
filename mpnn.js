/* In-browser forward pass of the trained CLRS-30 MPNN (128 hidden), mirroring
 * capture_net.py / clrs encoders+decoders exactly (soft hint feedback, repred).
 * Consumes a model bundle (models/<algo>.json + .bin) and a compact instance
 * (see model_export.build_instance); returns the per-step (z, msgs) that the
 * transcoder / attribution graph read. Works in Node, browsers and Web Workers. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MPNN = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function loadModel(man, buf) {
    const F = buf instanceof Float32Array ? buf : new Float32Array(buf.buffer ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) : buf);
    const L = d => ({ w: F.subarray(d.w, d.w + d.din * d.dout), b: F.subarray(d.b, d.b + d.dout), din: d.din, dout: d.dout });
    const M = { H: man.H, algo: man.algo, proc: {}, fields: [] };
    for (const k of ['m1', 'm2', 'me', 'mg', 'o1', 'o2', 'mlp0', 'mlp1']) M.proc[k] = L(man.proc[k]);
    M.proc.lnScale = F.subarray(man.proc.ln.scale, man.proc.ln.scale + man.H);
    M.proc.lnOffset = F.subarray(man.proc.ln.offset, man.proc.ln.offset + man.H);
    for (const f of man.fields) {
      if (f.stage === 'output') continue;
      M.fields.push({ name: f.name, stage: f.stage, loc: f.loc, type: f.type, K: f.K,
                      enc: (f.enc || []).map(L), dec: (f.dec || []).map(L) });
    }
    return M;
  }

  // out[r*dout+o] = b[o] + sum_i X[r*din+i] * W[i*dout+o]
  function linRows(X, rows, lay, out) {
    const { w, b, din, dout } = lay;
    out = out || new Float64Array(rows * dout);
    for (let r = 0; r < rows; r++) {
      const xo = r * din, oo = r * dout;
      for (let o = 0; o < dout; o++) out[oo + o] = b[o];
      for (let i = 0; i < din; i++) {
        const xi = X[xo + i]; if (xi === 0) continue;
        const wo = i * dout;
        for (let o = 0; o < dout; o++) out[oo + o] += xi * w[wo + o];
      }
    }
    return out;
  }

  function flat(x, out) {
    out = out || [];
    if (Array.isArray(x)) for (const v of x) flat(v, out); else out.push(x);
    return out;
  }
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  function softmaxRows(a, rows, cols) {           // in place over the last axis
    for (let r = 0; r < rows; r++) {
      const o = r * cols; let m = -Infinity;
      for (let c = 0; c < cols; c++) if (a[o + c] > m) m = a[o + c];
      let s = 0;
      for (let c = 0; c < cols; c++) { a[o + c] = Math.exp(a[o + c] - m); s += a[o + c]; }
      for (let c = 0; c < cols; c++) a[o + c] /= s;
    }
    return a;
  }

  // ground-truth hint / input data -> flat Float64Array in the layout the encoders expect
  function prep(f, data, N) {
    if (f.type === 'pointer') {
      if (f.loc === 'node') { const P = new Float64Array(N * N); for (let i = 0; i < N; i++) P[i * N + (data[i] | 0)] = 1; return P; }
      const P = new Float64Array(N * N * N);                                   // edge pointer
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) P[(i * N + j) * N + (data[i][j] | 0)] = 1;
      return P;
    }
    return Float64Array.from(flat(data));
  }

  function encodeInto(f, d, N, H, nodeF, edgeF, graphF) {
    const lay = f.enc[0], w = lay.w, b = lay.b;
    const cat = f.type === 'categorical', K = lay.din;
    const addVec = (out, o, vecOff, arr) => {                                   // out[o..o+H) += enc(arr[vecOff..])
      if (cat) { for (let h = 0; h < H; h++) { let s = b[h]; for (let k = 0; k < K; k++) s += arr[vecOff + k] * w[k * H + h]; out[o + h] += s; } }
      else { const v = arr[vecOff]; for (let h = 0; h < H; h++) out[o + h] += v * w[h] + b[h]; }
    };
    if (f.loc === 'node') {
      if (f.type === 'pointer') { for (let i = 0; i < N * N; i++) addVec(edgeF, i * H, i, d); }
      else for (let i = 0; i < N; i++) addVec(nodeF, i * H, cat ? i * K : i, d);
    } else if (f.loc === 'edge') {
      if (f.type === 'pointer') {
        const w2 = f.enc[1].w, b2 = f.enc[1].b;
        for (let u = 0; u < N; u++) for (let v = 0; v < N; v++) {
          let m1 = 0, m2 = 0;
          for (let x = 0; x < N; x++) { m1 += d[(x * N + u) * N + v]; m2 += d[(u * N + x) * N + v]; }
          m1 /= N; m2 /= N;
          const o = (u * N + v) * H;
          for (let h = 0; h < H; h++) edgeF[o + h] += m1 * w[h] + b[h] + m2 * w2[h] + b2[h];
        }
      } else for (let i = 0; i < N * N; i++) addVec(edgeF, i * H, i, d);
    } else {                                                                    // graph
      addVec(graphF, 0, 0, d);
    }
  }

  // soft-postprocessed hint decode (repred=True, hint_repred_mode='soft')
  function decodeHint(f, hT, edgeF, graphF, N, H, raw) {
    const H3 = 3 * H, dec = f.dec;
    if (f.loc === 'node') {
      if (f.type === 'pointer') {
        const p1 = linRows(hT, N, dec[0]), p2 = linRows(hT, N, dec[1]), p3 = linRows(edgeF, N * N, dec[2]);
        const pm = new Float64Array(H), out = new Float64Array(N * N), l3 = dec[3];
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          for (let c = 0; c < H; c++) { const r = p2[j * H + c] + p3[(j * N + i) * H + c], l = p1[i * H + c]; pm[c] = l > r ? l : r; }
          let s = l3.b[0]; for (let c = 0; c < H; c++) s += pm[c] * l3.w[c];
          out[i * N + j] = s;
        }
        return raw ? out : softmaxRows(out, N, N);
      }
      const o = linRows(hT, N, dec[0]);
      if (raw) return o;
      if (f.type === 'categorical') return softmaxRows(o, N, dec[0].dout);
      if (f.type === 'mask') { for (let i = 0; i < o.length; i++) o[i] = sigmoid(o[i]); return o; }
      if (f.type === 'mask_one') return softmaxRows(o, 1, N);
      return o;                                                                 // scalar
    }
    if (f.loc === 'edge') {
      const a1 = linRows(hT, N, dec[0]), a2 = linRows(hT, N, dec[1]), a3 = linRows(edgeF, N * N, dec[2]);
      const D = dec[0].dout, pred = new Float64Array(N * N * D);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let c = 0; c < D; c++)
        pred[(i * N + j) * D + c] = a1[i * D + c] + a2[j * D + c] + a3[(i * N + j) * D + c];
      if (f.type === 'pointer') {
        const q = linRows(hT, N, dec[3]), l4 = dec[4], out = new Float64Array(N * N * N);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) {
          let s = l4.b[0];
          for (let c = 0; c < H; c++) { const a = pred[(i * N + j) * H + c], b2 = q[k * H + c]; s += (a > b2 ? a : b2) * l4.w[c]; }
          out[(i * N + j) * N + k] = s;
        }
        return softmaxRows(out, N * N, N);
      }
      if (f.type === 'mask') { for (let i = 0; i < pred.length; i++) pred[i] = sigmoid(pred[i]); return pred; }
      if (f.type === 'mask_one') return softmaxRows(pred, 1, N * N);
      return pred;
    }
    // graph
    const emb = new Float64Array(H3);
    for (let c = 0; c < H3; c++) { let m = -Infinity; for (let i = 0; i < N; i++) if (hT[i * H3 + c] > m) m = hT[i * H3 + c]; emb[c] = m; }
    const pn = linRows(emb, 1, dec[0]), pg = linRows(graphF, 1, dec[1]);
    const out = new Float64Array(pn.length);
    for (let i = 0; i < out.length; i++) out[i] = pn[i] + pg[i];
    if (f.type === 'categorical') return softmaxRows(out, 1, out.length);
    if (f.type === 'mask') { for (let i = 0; i < out.length; i++) out[i] = sigmoid(out[i]); return out; }
    return out;
  }

  // Full forward pass with optional recording for attribution.
  // opts.record: {from, to}   keep frozen nonlinearities (ReLU masks, max-routing), inputs of the transcoder site,
  //                           hint values read, and raw decoder logits for steps in [from, to]
  // opts.stopAt: t            stop after step t (inclusive)
  // opts.patch: {t, i, dy}    add dy (H) to node i's new hidden state at step t (feature intervention)
  // opts.hints: [..]          use these per-step hint inputs instead of the model's own decoded ones
  function runRecord(M, inst, opts) {
    opts = opts || {};
    const N = inst.n, H = M.H, P = M.proc;
    const steps = Math.min(inst.length - 1, opts.stopAt !== undefined ? opts.stopAt + 1 : Infinity);
    const inFields = M.fields.filter(f => f.stage === 'input'), hintFields = M.fields.filter(f => f.stage === 'hint');
    const inputs = {}; for (const f of inFields) inputs[f.name] = prep(f, inst.inputs[f.name], N);
    let hints = {}; for (const f of hintFields) hints[f.name] = prep(f, inst.hints0[f.name], N);
    let hidden = new Float64Array(N * H);
    const R = opts.record, rec = [], Z = [], MS = [], hintsRead = [];
    for (let t = 0; t < steps; t++) {
      if (opts.hints && opts.hints[t]) hints = opts.hints[t];
      hintsRead.push(hints);
      const nodeF = new Float64Array(N * H), edgeF = new Float64Array(N * N * H), graphF = new Float64Array(H);
      for (const f of inFields) encodeInto(f, inputs[f.name], N, H, nodeF, edgeF, graphF);
      for (const f of hintFields) encodeInto(f, hints[f.name], N, H, nodeF, edgeF, graphF);
      const z = new Float64Array(N * 2 * H);                                    // concat(node_fts, hidden)
      for (let i = 0; i < N; i++) { z.set(nodeF.subarray(i * H, (i + 1) * H), i * 2 * H); z.set(hidden.subarray(i * H, (i + 1) * H), i * 2 * H + H); }
      const keep = R && t >= R.from && t <= R.to;
      const m1 = linRows(z, N, P.m1), m2 = linRows(z, N, P.m2), me = linRows(edgeF, N * N, P.me), mg = linRows(graphF, 1, P.mg);
      const pre = new Float64Array(N * N * H), mask0 = keep ? new Uint8Array(N * N * H) : null;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {                // i = sender, j = receiver
        const o = (i * N + j) * H;
        for (let c = 0; c < H; c++) { const v = m2[i * H + c] + m1[j * H + c] + me[o + c] + mg[c]; if (v > 0) { pre[o + c] = v; if (mask0) mask0[o + c] = 1; } }
      }
      const a0 = linRows(pre, N * N, P.mlp0), mask1 = keep ? new Uint8Array(N * N * H) : null;
      for (let k = 0; k < a0.length; k++) { if (a0[k] < 0) a0[k] = 0; else if (mask1 && a0[k] > 0) mask1[k] = 1; }
      const mm = linRows(a0, N * N, P.mlp1);
      const agg = new Float64Array(N * H).fill(-Infinity), route = keep ? new Int32Array(N * H) : null;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const o = (i * N + j) * H; for (let c = 0; c < H; c++) if (mm[o + c] > agg[j * H + c]) { agg[j * H + c] = mm[o + c]; if (route) route[j * H + c] = i; } }
      const h1 = linRows(z, N, P.o1), h2 = linRows(agg, N, P.o2), hNew = new Float64Array(N * H);
      for (let i = 0; i < N; i++) {
        let mean = 0; const o = i * H;
        for (let c = 0; c < H; c++) { const v = h1[o + c] + h2[o + c]; hNew[o + c] = v > 0 ? v : 0; mean += hNew[o + c]; }
        mean /= H; let vr = 0;
        for (let c = 0; c < H; c++) vr += (hNew[o + c] - mean) ** 2;
        const inv = 1 / Math.sqrt(vr / H + 1e-5);
        for (let c = 0; c < H; c++) hNew[o + c] = (hNew[o + c] - mean) * inv * P.lnScale[c] + P.lnOffset[c];
      }
      if (opts.patch && opts.patch.t === t) { const o = opts.patch.i * H; for (let c = 0; c < H; c++) hNew[o + c] += opts.patch.dy[c]; }
      const zr = [], mr = [];
      for (let i = 0; i < N; i++) { zr.push(Float32Array.from(z.subarray(i * 2 * H, (i + 1) * 2 * H))); mr.push(Float32Array.from(agg.subarray(i * H, (i + 1) * H))); }
      Z.push(zr); MS.push(mr);
      const hT = new Float64Array(N * 3 * H);
      for (let i = 0; i < N; i++) { hT.set(nodeF.subarray(i * H, (i + 1) * H), i * 3 * H); hT.set(hidden.subarray(i * H, (i + 1) * H), i * 3 * H + H); hT.set(hNew.subarray(i * H, (i + 1) * H), i * 3 * H + 2 * H); }
      const r = { t };
      if (keep) {
        Object.assign(r, { z, agg, hidden, hNew, mask0, mask1, route, hT, edgeF, graphF, logits: {} });
        for (const f of hintFields) if (f.loc === 'node') r.logits[f.name] = decodeHint(f, hT, edgeF, graphF, N, H, true);
      }
      rec.push(r);
      if (t < inst.length - 2) {
        const next = {};
        for (const f of hintFields) next[f.name] = decodeHint(f, hT, edgeF, graphF, N, H);
        hints = next;
      }
      hidden = hNew;
    }
    return { z: Z, msgs: MS, rec, hintsRead, inputs };
  }

  function runModel(M, inst) { const r = runRecord(M, inst); return { z: r.z, msgs: r.msgs }; }

  // pointer-decoder internals for target (i -> j) at a recorded step: frozen max-selection per channel
  function pointerSel(f, r, i, j, N, H) {
    const dec = f.dec, hT = r.hT;
    const p1 = linRows(hT.subarray(i * 3 * H, (i + 1) * 3 * H), 1, dec[0]), p2 = linRows(hT.subarray(j * 3 * H, (j + 1) * 3 * H), 1, dec[1]);
    const p3 = linRows(r.edgeF.subarray((j * N + i) * H, (j * N + i + 1) * H), 1, dec[2]);
    const sel = new Uint8Array(H);
    for (let c = 0; c < H; c++) sel[c] = p1[c] > p2[c] + p3[c] ? 1 : 0;
    return sel;
  }

  return { loadModel, runModel, runRecord, pointerSel, linRows, prep };
});
