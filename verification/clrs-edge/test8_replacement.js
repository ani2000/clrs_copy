// Does the transcoder work as planned INSIDE the network?  The message MLP is replaced by the transcoder's reconstruction at every step (winners and hints are
// recomputed, i.e. the "full" mode of the plan), on the tracer's own graphs, and the predictions of the main target are compared with the unmodified network.
// Not a pass/fail gate on one number: the training report's retention (F1 kept) is printed next to it. Gate: agreement >= 0.80 and accuracy kept >= 0.80 for the tier-A algorithms.
const fs = require('fs'), path = require('path'), ROOT = path.join(__dirname, '..', '..'), MODELS = process.env.MODELS, INST = process.env.INST, PER = +(process.env.PER || 6);
const E = require(ROOT + '/clrs-edge-attrib.js'), MP = require(ROOT + '/mpnn.js');
const summary = JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', 'clrs-edge-summary.json'))).summary;
const algos = (process.env.ALGOS || Object.keys(summary).join(',')).split(','); let bad = 0;
for (const a of algos) {
  const man = JSON.parse(fs.readFileSync(path.join(MODELS, a + '.json'))), b = fs.readFileSync(path.join(MODELS, a + '.bin')), M = MP.loadModel(man, new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  const T = E.prepTC(JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.json'))), fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.bin')));
  const meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'algos', a, 'meta.json'))), TYPE = meta.target_type, H = M.H, pre = new Float64Array(T.D), tmp = new Float64Array(T.D), yh = new Float64Array(H);
  const site = (t, s, y) => { const N2 = y.length / H; for (let p = 0; p < N2; p++) { const idx = E.encodeRow(T, s, p * H, pre, tmp); E.decodeRow(T, idx, pre, yh); y.set(yh, p * H); } };
  const files = fs.readdirSync(INST).filter(f => f.startsWith(a + '_')).sort().slice(0, PER);
  let agree = 0, tot = 0, accO = 0, accR = 0, accN = 0;
  for (const f of files) {
    const inst = JSON.parse(fs.readFileSync(path.join(INST, f))), N = inst.n, steps = inst.length - 1, tt = inst.target_true, rec = { record: { from: 0, to: steps - 1 } };
    const o = MP.runRecord(M, inst, rec), r = MP.runRecord(M, inst, Object.assign({ site }, rec));
    const dec = (L, node) => TYPE === 'mask' ? (L[node] > 0 ? 1 : 0) : TYPE === 'categorical' ? (() => { const C = L.length / N; let k = 0; for (let c = 1; c < C; c++) if (L[node * C + c] > L[node * C + k]) k = c; return k; })() : TYPE === 'mask_one' ? (() => { let k = 0; for (let i = 1; i < N; i++) if (L[i] > L[k]) k = i; return k === node ? 1 : 0; })() : L[node];
    for (let t = 0; t < steps; t++) for (let node = 0; node < N; node++) {
      const x = dec(o.rec[t].logits[meta.target], node), y = dec(r.rec[t].logits[meta.target], node), truth = tt[t + 1][node];
      tot++; if (TYPE === 'scalar') { if (Math.abs(x - y) <= 0.05 * Math.max(1, Math.abs(x))) agree++; if (Math.abs(x - truth) <= 0.05 * Math.max(1, Math.abs(truth))) accO++; if (Math.abs(y - truth) <= 0.05 * Math.max(1, Math.abs(truth))) accR++; }
      else { if (x === y) agree++; if (x === Math.round(truth)) accO++; if (y === Math.round(truth)) accR++; } accN++;
    }
  }
  const s = summary[a], keep = accO ? accR / accO : NaN, gate = s.tier === 'A' ? (agree / tot >= 0.8 && keep >= 0.8) : true; if (!gate) bad++;
  console.log(`${gate ? 'ok  ' : 'BAD '} ${a.padEnd(30)} tier ${s.tier}  prediction agreement ${(100 * agree / tot).toFixed(1)}%  accuracy vs truth: network ${(100 * accO / accN).toFixed(1)}% -> replaced ${(100 * accR / accN).toFixed(1)}% (kept ${(100 * keep).toFixed(0)}%)  | training report: F1 kept ${(100 * s.retention_val).toFixed(0)}% at 16 nodes   [${files.length} graphs, ${tot} node-steps]`);
}
console.log(bad ? 'TEST 8 FAILED' : 'TEST 8 OK'); process.exit(bad ? 1 : 0);
