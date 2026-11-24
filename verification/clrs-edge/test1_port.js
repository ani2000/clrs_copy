// Test 1 (handoff section 6): the port must reproduce the reference outputs of clrs-transcoder-reference.json for all 12 algorithms.
const fs = require('fs'), path = require('path'), E = require('../../clrs-edge-attrib.js');
const REF = process.env.REF || '/Users/anindya/CLRS/new run/clrs-transcoder-reference.json', DIR = path.join(__dirname, '..', '..', 'edge_tc');
const ref = JSON.parse(fs.readFileSync(REF)).golden; let bad = 0, rows = 0, worst = 0, worstRef = 0;
// (a) vs an independent numpy float64 implementation on the same stored inputs: must agree to rounding (1e-9)
// (b) vs the reference's own stored outputs: those were computed from unrounded inputs, the stored inputs have 6 significant digits, so 2e-3 is the honest tolerance
const cross = process.env.CROSS ? JSON.parse(fs.readFileSync(process.env.CROSS)) : null;
for (const a of Object.keys(ref)) {
  const man = JSON.parse(fs.readFileSync(path.join(DIR, a + '.json'))), T = E.prepTC(man, fs.readFileSync(path.join(DIR, a + '.bin')));
  const pre = new Float64Array(T.D), tmp = new Float64Array(T.D), y = new Float64Array(T.H);
  for (const g of ref[a].edge.rows) {
    rows++; const idx = E.encodeRow(T, Float64Array.from(g.s), 0, pre, tmp); E.decodeRow(T, idx, pre, y);
    const same = idx.length === g.active_idx.length && idx.every((v, q) => v === g.active_idx[q]);
    let e = 0; for (let q = 0; q < idx.length; q++) e = Math.max(e, Math.abs(pre[idx[q]] - g.active_val[q]) / Math.max(1e-9, Math.abs(g.active_val[q])));
    for (let c = 0; c < T.H; c++) e = Math.max(e, Math.abs(y[c] - g.y_hat[c]) / Math.max(1e-3, Math.abs(g.y_hat[c])));
    for (let q = 0; q < 8; q++) e = Math.max(e, Math.abs(pre[q] - g.pre_first8[q]) / Math.max(1e-3, Math.abs(g.pre_first8[q])));
    worstRef = Math.max(worstRef, e);
    if (cross) { const c = cross[a][ref[a].edge.rows.indexOf(g)]; let e2 = idx.length === c.active_idx.length && idx.every((v, q) => v === c.active_idx[q]) ? 0 : Infinity;
      for (let q = 0; q < idx.length; q++) e2 = Math.max(e2, Math.abs(pre[idx[q]] - c.active_val[q]) / Math.abs(c.active_val[q])); for (let cc = 0; cc < T.H; cc++) e2 = Math.max(e2, Math.abs(y[cc] - c.y_hat[cc]) / Math.max(1e-3, Math.abs(c.y_hat[cc]))); worst = Math.max(worst, e2); if (e2 > 1e-9) { bad++; console.log('FAIL (numpy cross-check)', a, e2); } }
    if (!same || e > 2e-3) { bad++; console.log('FAIL', a, 'indices equal:', same, 'max rel err', e); }
  }
}
console.log(bad ? `TEST 1 FAILED (${bad} problems in ${rows} rows)` : `TEST 1 OK: ${rows} rows over ${Object.keys(ref).length} algorithms; active indices identical; vs independent numpy float64 ${cross ? 'worst relative error ' + worst.toExponential(2) : '(not run)'}; vs the stored reference outputs worst ${worstRef.toExponential(2)} (their inputs are rounded to 6 digits)`);
process.exit(bad ? 1 : 0);
