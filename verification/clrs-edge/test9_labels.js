// Test 9: the measured meanings must be reproducible by a second, simple implementation.  For the named latents with the highest held-out F1 of an algorithm, recompute
// precision / recall / F1 of the stored best condition on the held-out (odd-seed) graphs by brute force, and the firing counts, and compare with the stored table.
const fs = require('fs'), path = require('path'), ROOT = path.join(__dirname, '..', '..'), MODELS = process.env.MODELS, INST = process.env.INST, TABLES = process.env.TABLES || path.join(ROOT, 'edge_tc');
const E = require(ROOT + '/clrs-edge-attrib.js'), MP = require(ROOT + '/mpnn.js'); let bad = 0;
for (const algo of (process.env.ALGOS || 'bfs,dijkstra').split(',')) {
  const J = JSON.parse(fs.readFileSync(path.join(TABLES, algo + '.latents.json'))), meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'algos', algo, 'meta.json')));
  const man = JSON.parse(fs.readFileSync(path.join(MODELS, algo + '.json'))), b = fs.readFileSync(path.join(MODELS, algo + '.bin')), M = MP.loadModel(man, new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  const T = E.prepTC(JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', algo + '.json'))), fs.readFileSync(path.join(ROOT, 'edge_tc', algo + '.bin'))), H = M.H, pre = new Float64Array(T.D), tmp = new Float64Array(T.D);
  const pick = Object.entries(J.latents).filter(([f, v]) => v.tier !== '-').sort((a, c) => c[1].best[0][3] - a[1].best[0][3]).slice(0, 4), want = new Map(pick.map(([f, v]) => [+f, v]));
  // brute-force conditions, written independently of the labeller: only the ones the picked latents use
  const cond = {
    edge: c => c.s !== c.r && c.A[c.s][c.r] > 0, nonedge: c => c.s !== c.r && !(c.A[c.s][c.r] > 0), self: c => c.s === c.r, src_sender: c => c.s === c.src, src_receiver: c => c.r === c.src,
    sender_wins: c => c.wins > 0, receiver_changes: c => Math.abs(c.a[c.r] - c.b[c.r]) > 1e-6,
    sender_on: c => c.a[c.s] > 0.5, sender_off: c => !(c.a[c.s] > 0.5), receiver_on: c => c.a[c.r] > 0.5, receiver_off: c => !(c.a[c.r] > 0.5),
    receiver_turns_on: c => !(c.a[c.r] > 0.5) && c.b[c.r] > 0.5, receiver_stays_on: c => c.a[c.r] > 0.5 && c.b[c.r] > 0.5, receiver_stays_off: c => !(c.a[c.r] > 0.5) && !(c.b[c.r] > 0.5),
    frontier: c => c.s !== c.r && c.A[c.s][c.r] > 0 && c.a[c.s] > 0.5 && !(c.a[c.r] > 0.5), discovers: c => c.s !== c.r && c.A[c.s][c.r] > 0 && c.a[c.s] > 0.5 && !(c.a[c.r] > 0.5) && c.b[c.r] > 0.5,
    sender_marked: c => c.a[c.s] > 0.5, receiver_marked_now: c => c.a[c.r] > 0.5, receiver_marked_next: c => c.b[c.r] > 0.5, receiver_newly_marked: c => !(c.a[c.r] > 0.5) && c.b[c.r] > 0.5,
    sender_has_value: c => c.a[c.s] > 1e-6 || c.s === c.src, sender_no_value: c => !(c.a[c.s] > 1e-6 || c.s === c.src), receiver_has_value: c => c.a[c.r] > 1e-6 || c.r === c.src, receiver_no_value: c => !(c.a[c.r] > 1e-6 || c.r === c.src),
    receiver_gets_value: c => !(c.a[c.r] > 1e-6 || c.r === c.src) && (c.b[c.r] > 1e-6 || c.r === c.src), receiver_value_down: c => (c.a[c.r] > 1e-6 || c.r === c.src) && c.b[c.r] < c.a[c.r] - 1e-6,
    receiver_value_up: c => (c.a[c.r] > 1e-6 || c.r === c.src) && c.b[c.r] > c.a[c.r] + 1e-6,
    sender_value_lower: c => (c.a[c.s] > 1e-6 || c.s === c.src) && (c.a[c.r] > 1e-6 || c.r === c.src) && c.a[c.s] < c.a[c.r],
    receiver_changes_class: c => Math.round(c.a[c.r]) !== Math.round(c.b[c.r]) };
  const condOf = id => { let m; if (cond[id]) return cond[id];
    if ((m = /^sender_class_(\d+)$/.exec(id))) return c => Math.round(c.a[c.s]) === +m[1]; if ((m = /^receiver_class_(\d+)$/.exec(id))) return c => Math.round(c.a[c.r]) === +m[1];
    if ((m = /^receiver_becomes_(\d+)$/.exec(id))) return c => Math.round(c.a[c.r]) !== Math.round(c.b[c.r]) && Math.round(c.b[c.r]) === +m[1]; throw new Error('no brute-force version of ' + id); };
  const cnt = new Map([...want.keys()].map(f => [f, { act: 0, tp: [0, 0, 0] }])), nTrue = new Array(want.size).fill(0); let held = 0;
  for (const f of fs.readdirSync(INST).filter(f => f.startsWith(algo + '_')).sort()) {
    const seed = +/_(\d+)\.json$/.exec(f)[1]; if (seed % 2 === 0) continue;
    const inst = JSON.parse(fs.readFileSync(path.join(INST, f))), N = inst.n, steps = inst.length - 1, tt = inst.target_true, stride = Math.max(1, Math.ceil(steps / +(process.env.MAXSTEPS || 20)));
    const run = MP.runRecord(M, inst, { record: { from: 0, to: steps - 1 } });
    for (let t = 0; t < steps; t += stride) {
      const rec = run.rec[t], wins = new Int32Array(N * N); for (let r = 0; r < N; r++) for (let c = 0; c < H; c++) wins[rec.route[r * H + c] * N + r]++;
      for (let s = 0; s < N; s++) for (let r = 0; r < N; r++) { held++;
        const idx = E.encodeRow(T, rec.s, (s * N + r) * H, pre, tmp), c = { s, r, A: inst.adj, src: inst.source, wins: wins[s * N + r], a: tt[t], b: tt[t + 1] };
        const q = [...want.entries()].map(([f, v]) => condOf(J.conditions[v.best[0][0]].id)(c));
        want.forEach((v, f) => { const k = [...want.keys()].indexOf(f); if (q[k]) nTrue[k]++; if (idx.includes(f)) { const x = cnt.get(f); x.act++; if (q[k]) x.tp[0]++; } }); } }
  }
  [...want.entries()].forEach(([f, v], k) => { const x = cnt.get(f), P = x.act ? x.tp[0] / x.act : 0, R = nTrue[k] ? x.tp[0] / nTrue[k] : 0, F = P + R ? 2 * P * R / (P + R) : 0, s = v.best[0];
    const ok = Math.abs(P - s[1]) < 2e-3 && Math.abs(R - s[2]) < 2e-3 && Math.abs(F - s[3]) < 2e-3; if (!ok) bad++;
    console.log(`${ok ? 'ok  ' : 'BAD '} ${algo} f${f} "${J.conditions[s[0]].id}" brute force P ${P.toFixed(3)} R ${R.toFixed(3)} F1 ${F.toFixed(3)} | stored ${s[1]} ${s[2]} ${s[3]}  (held-out rows ${held})`); });
  if (held !== J.rows_held) { bad++; console.log('BAD  held-out row count', held, 'vs stored', J.rows_held); }
}
console.log(bad ? 'TEST 9 FAILED' : 'TEST 9 OK'); process.exit(bad ? 1 : 0);
