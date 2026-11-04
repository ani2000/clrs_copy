const fs = require('fs');
const APP = require('path').resolve(__dirname, '..');
const MPNN = require(APP + '/mpnn.js');
const files = fs.readdirSync('/tmp/verif').filter(f => f.endsWith('.json')).sort();
const cache = {}; let allOk = true;
for (const f of files) {
  const algo = f.replace(/_\d+_\d+\.json$/, '');
  if (!cache[algo]) cache[algo] = MPNN.loadModel(JSON.parse(fs.readFileSync(`${APP}/models/${algo}.json`)), fs.readFileSync(`${APP}/models/${algo}.bin`));
  const ref = JSON.parse(fs.readFileSync('/tmp/verif/' + f));
  const out = MPNN.runModel(cache[algo], ref.inst);
  let md = 0;
  for (let t = 0; t < ref.z.length; t++) for (let i = 0; i < ref.z[t].length; i++) {
    for (let c = 0; c < 256; c++) md = Math.max(md, Math.abs(out.z[t][i][c] - ref.z[t][i][c]));
    for (let c = 0; c < 128; c++) md = Math.max(md, Math.abs(out.msgs[t][i][c] - ref.msgs[t][i][c]));
  }
  const ok = out.z.length === ref.z.length && md < 2e-3; allOk &&= ok;
  console.log(f.padEnd(40), 'steps', String(ref.z.length).padStart(3), 'max|diff|', md.toExponential(2), ok ? 'OK' : 'MISMATCH');
}
console.log(allOk ? 'ALL MATCH' : 'MISMATCH');
