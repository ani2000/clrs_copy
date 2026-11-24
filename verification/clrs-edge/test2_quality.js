// Test 2 (handoff section 6): the transcoder must reconstruct the REAL messages of the site's own model. FVU = sum||y - y_hat||^2 / sum||y - mean(y)||^2
// over all pair rows of all steps. A transcoder that does not belong to this model would score about 1 (no better than the mean). Gate: FVU < 0.05.
// The training-time FVU (report.json, CLRS validation graphs) is printed for comparison: the tracer samples its own random graphs, so the numbers differ (up to ~7x for bridges / articulation points).
const fs = require('fs'), path = require('path'), E = require('../../clrs-edge-attrib.js'), MP = require('../../mpnn.js');
const ROOT = path.join(__dirname, '..', '..'), MODELS = process.env.MODELS, INST = process.env.INST;
const summary = JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', 'clrs-edge-summary.json'))).summary;
const algos = (process.env.ALGOS || Object.keys(summary).join(',')).split(','); let bad = 0;
for (const a of algos) {
  const man = JSON.parse(fs.readFileSync(path.join(MODELS, a + '.json'))), buf = fs.readFileSync(path.join(MODELS, a + '.bin'));
  const M = MP.loadModel(man, new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
  const tm = JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.json'))), T = E.prepTC(tm, fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.bin')));
  const H = T.H, pre = new Float64Array(T.D), tmp = new Float64Array(T.D), yh = new Float64Array(H);
  let num = 0, sum = new Float64Array(H), sq = 0, rows = 0, l0 = 0;
  for (const f of fs.readdirSync(INST).filter(f => f.startsWith(a + '_'))) {
    const inst = JSON.parse(fs.readFileSync(path.join(INST, f))), N = inst.n, steps = inst.length - 1;
    const run = MP.runRecord(M, inst, { record: { from: 0, to: steps - 1 } });
    for (let t = 0; t < steps; t++) { const r = run.rec[t];
      for (let p = 0; p < N * N; p++) {
        const idx = E.encodeRow(T, r.s, p * H, pre, tmp); E.decodeRow(T, idx, pre, yh); l0 += idx.length; rows++;
        for (let c = 0; c < H; c++) { const y = r.y[p * H + c]; num += (y - yh[c]) ** 2; sum[c] += y; sq += y * y; }
      } }
  }
  let tot = sq; for (let c = 0; c < H; c++) tot -= sum[c] * sum[c] / rows;
  const fvu = num / tot, ref = summary[a].nmse_val, ratio = fvu / ref;
  const ok = fvu < 0.05;
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'BAD '} ${a.padEnd(30)} FVU ${fvu.toFixed(4)}  (report val ${ref.toFixed(4)}, x${ratio.toFixed(2)})  L0 ${(l0 / rows).toFixed(2)}  rows ${rows}`);
}
console.log(bad ? 'TEST 2 FAILED: the transcoder does not match this model on some algorithms' : 'TEST 2 OK'); process.exit(bad ? 1 : 0);
