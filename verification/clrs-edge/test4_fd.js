// Tests 3-5 (handoff section 6) for the edge-site graph, on an independent replay of the replacement model.
//  E0  : replaying the replacement model (transcoder output + error node, winners / sigma / ReLU pattern frozen) with no change must reproduce the real
//        hidden states and target logits.
//  FD  : nudge one latent's activation in the replay. (a) later pre-activations (one step on) must change by exactly (edge weight / activation) x nudge;
//        (b) every target logit must change by exactly (total effect) x nudge. The replay is linear in the nudge, so a large nudge is used on purpose.
//  SUM : explain() already checks, for every latent and every target, that the sum of incoming edges plus the explicit bias equals its value (checks.maxAbs).
const fs = require('fs'), path = require('path'), ROOT = path.join(__dirname, '..', '..'), MODELS = process.env.MODELS, INST = process.env.INST;
const E = require(ROOT + '/clrs-edge-attrib.js'), MP = require(ROOT + '/mpnn.js'), AT = require(ROOT + '/attrib.js');
const algos = (process.env.ALGOS || 'bfs,bellman_ford,dijkstra,dfs,mst_kruskal,floyd_warshall').split(','); let bad = 0;
const rnd = (() => { let s = 12345; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
for (const a of algos) {
  const M = (() => { const man = JSON.parse(fs.readFileSync(path.join(MODELS, a + '.json'))), b = fs.readFileSync(path.join(MODELS, a + '.bin')); return MP.loadModel(man, new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))); })();
  const T = E.prepTC(JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.json'))), fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.bin')));
  const inst = JSON.parse(fs.readFileSync(path.join(INST, a + '_103.json'))), N = inst.n, H = M.H, P = M.proc, t = Math.min(3, inst.length - 2);
  const meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'algos', a, 'meta.json'))), ptr = M.fields.some(f => f.name === 'pi_h' && f.stage === 'hint' && f.loc === 'node' && f.type === 'pointer') ? 'pi_h' : null;
  const ex = E.explain(MP, AT, M, T, inst, { t, i: 3 % N, main: meta.target, pointerField: ptr, window: 3, full: true }), pr = E.prune(ex), { St, tLo } = ex;
  // ---- replay of the replacement model; `bump` = {ts, p, q, d} adds d to the activation of latent q at pair p of step ts
  function replay(bump) {
    let h = Float64Array.from(St[tLo].hidden); const pres = {}; let hT = null, edgeF = null;
    for (let ts = tLo; ts <= t; ts++) {
      const S = St[ts], r = S.r, z = new Float64Array(N * 2 * H);
      for (let k = 0; k < N; k++) { z.set(r.z.subarray(k * 2 * H, k * 2 * H + H), k * 2 * H); z.set(h.subarray(k * H, (k + 1) * H), k * 2 * H + H); }
      const m1 = MP.linRows(z, N, P.m1), m2 = MP.linRows(z, N, P.m2), me = MP.linRows(r.edgeF, N * N, P.me), mg = MP.linRows(r.graphF, 1, P.mg);
      const yUsed = new Float64Array(N * N * H); pres[ts] = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const p = i * N + j, L = S.lat[p], s = new Float64Array(H); for (let c = 0; c < H; c++) s[c] = m2[i * H + c] + m1[j * H + c] + me[p * H + c] + mg[c];
        const act = L.idx.map((f, q) => { let v = T.b_enc[f]; for (let c = 0; c < H; c++) v += ((s[c] - T.s_mu[c]) / T.s_scale - T.b_pre[c]) * T.W_enc[c * T.D + f]; return v; });
        pres[ts].push(act.slice()); if (bump && bump.ts === ts && bump.p === p) act[bump.q] += bump.d;
        for (let c = 0; c < H; c++) { let v = T.b_dec[c]; L.idx.forEach((f, q) => { v += act[q] * T.W_dec[f * H + c]; }); yUsed[p * H + c] = v * T.y_scale + T.y_mu[c] + S.eps[p * H + c]; }
      }
      const hn = new Float64Array(N * H), h1 = MP.linRows(z, N, P.o1);
      for (let k = 0; k < N; k++) {
        const x = new Float64Array(H); for (let c = 0; c < H; c++) { let m = 0; const w = r.route[k * H + c]; m = yUsed[(w * N + k) * H + c]; x[c] = h1[k * H + c] + m * 0; }
        const mk = new Float64Array(H); for (let c = 0; c < H; c++) mk[c] = yUsed[(r.route[k * H + c] * N + k) * H + c];
        const h2 = MP.linRows(mk, 1, P.o2); let mean = 0; const u = new Float64Array(H);
        for (let c = 0; c < H; c++) { u[c] = S.Du[k * H + c] ? h1[k * H + c] + h2[c] : 0; mean += u[c]; } mean /= H;
        for (let c = 0; c < H; c++) hn[k * H + c] = P.lnScale[c] * (u[c] - mean) / S.sig[k] + P.lnOffset[c];
      }
      if (ts === t) { hT = new Float64Array(N * 3 * H); for (let k = 0; k < N; k++) { hT.set(r.z.subarray(k * 2 * H, k * 2 * H + H), k * 3 * H); hT.set(h.subarray(k * H, (k + 1) * H), k * 3 * H + H); hT.set(hn.subarray(k * H, (k + 1) * H), k * 3 * H + 2 * H); } edgeF = r.edgeF; }
      h = hn;
    }
    const logit = tg => { const f = M.fields.find(x => x.name === tg.field);
      if (f.type === 'pointer') { const p1 = MP.linRows(hT.subarray(tg.i * 3 * H, (tg.i + 1) * 3 * H), 1, f.dec[0]), p2 = MP.linRows(hT.subarray(tg.j * 3 * H, (tg.j + 1) * 3 * H), 1, f.dec[1]), p3 = MP.linRows(edgeF.subarray((tg.j * N + tg.i) * H, (tg.j * N + tg.i + 1) * H), 1, f.dec[2]);
        let v = f.dec[3].b[0]; for (let c = 0; c < H; c++) v += f.dec[3].w[c] * (tg.sel[c] ? p1[c] : p2[c] + p3[c]); return v; }
      const d0 = f.dec[0], c = tg.c || 0; let v = d0.b[c]; for (let k = 0; k < 3 * H; k++) v += d0.w[k * d0.dout + c] * hT[tg.i * 3 * H + k]; return v; };
    return { h, pres, logits: ex.targets.map(logit) };
  }
  const base = replay(null); let e0 = 0;
  const rt = St[t].r; for (let k = 0; k < N * H; k++) e0 = Math.max(e0, Math.abs(base.h[k] - rt.hNew[k]));
  ex.targets.forEach((tg, q) => { const f = M.fields.find(x => x.name === tg.field), real = tg.kind === 'pointer' ? rt.logits[tg.field][tg.i * N + tg.j] : rt.logits[tg.field][tg.i * f.dec[0].dout + (tg.c || 0)]; e0 = Math.max(e0, Math.abs(base.logits[q] - real)); });
  // ---- finite differences
  const edgeW = new Map(); for (let e = 0; e < ex.E; e++) edgeW.set(ex.eS[e] * 1e7 + ex.eD[e], ex.eW[e]);
  const feats = ex.nodes.map((nd, k) => [nd, k]).filter(([nd]) => nd.type === 'feature' && nd.t < t); let nA = 0, errA = 0, nB = 0, errB = 0, scaleB = 0;
  for (let trial = 0; trial < 20; trial++) {
    const [nd, k] = feats[Math.floor(rnd() * feats.length)], q = St[nd.t].lat[nd.pair].idx.indexOf(nd.f), d = 0.37, alt = replay({ ts: nd.t, p: nd.pair, q, d });
    for (const [nd2, k2] of ex.nodes.map((n2, k3) => [n2, k3]).filter(([n2]) => n2.type === 'feature' && n2.t === nd.t + 1)) {
      const qi = St[nd2.t].lat[nd2.pair].idx.indexOf(nd2.f), fd = (alt.pres[nd2.t][nd2.pair][qi] - base.pres[nd2.t][nd2.pair][qi]) / d, w = (edgeW.get(k * 1e7 + k2) || 0) / nd.act;
      nA++; errA = Math.max(errA, Math.abs(fd - w));
    }
    ex.targets.forEach((tg, q2) => { const l = ex.nid.get(tg.id), pred = pr.effects.get(l)[k], fd = (alt.logits[q2] - base.logits[q2]) / d; nB++; errB = Math.max(errB, Math.abs(fd - pred)); scaleB = Math.max(scaleB, Math.abs(pred)); });
  }
  const ok = e0 < 1e-8 && errA < 1e-8 && errB < 1e-7 * Math.max(1, scaleB) && ex.checks.maxAbs < 1e-9; if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'BAD '} ${a.padEnd(22)} E0 replay vs real ${e0.toExponential(1)} | edge weights vs FD (${nA}) ${errA.toExponential(1)} | total effects vs FD (${nB}, |max| ${scaleB.toFixed(2)}) ${errB.toExponential(1)} | sum test (${ex.checks.n}) ${ex.checks.maxAbs.toExponential(1)}`);
}
console.log(bad ? 'TESTS 3-5 FAILED' : 'TESTS 3-5 OK'); process.exit(bad ? 1 : 0);
