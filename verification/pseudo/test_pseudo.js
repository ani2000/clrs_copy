const P = require('./pseudo_block.js'); const fs = require('fs');
const dir = __dirname + '/inst/'; let bad = 0;
const files = fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort();
const verbose = process.argv[2];
for (const f of files) {
  const algo = f.replace(/(_\d+)?\.json$/, ''); const inst = JSON.parse(fs.readFileSync(dir + f));
  const S = P.steps(algo, inst);
  if (!S || S.length !== inst.target_true.length) { console.log('FAIL', f, 'steps', S && S.length, inst.target_true.length); bad++; continue; }
  const nl = P.CODE[algo].length; let err = 0;
  S.forEach((s, t) => { if (!s.lines.length || s.lines.some(i => i < 0 || i >= nl) || s.ctx.some(i => i < 0 || i >= nl || s.cur.includes(i)) || Object.keys(s.bind).some(i => i < 0 || i >= nl)) { err++; } if (!s.note) err++; });
  let extra = '';
  if (algo === 'dijkstra' || algo === 'mst_prim') { const ch = S.filter(s => /\((d = |via)|smallest key, \d/.test(s.note)).length; extra = ` inferredU=${ch}/${S.filter(s=>s.head.startsWith('Extract &')).length}`; }
  if (algo === 'mst_kruskal') extra = ` edge-notes=${S.filter(s => /Next lightest edge: \d|Different|Same set/.test(s.note)).length}/${S.length - 1}`;
  if (algo === 'bridges') extra = ' ' + S[S.length - 1].state.slice(-1)[0].join(': ');
  if (algo === 'strongly_connected_components') extra = ' ' + S[S.length - 1].state.slice(-1)[0].join(': ');
  if (algo === 'topological_sort') extra = ' ' + S[S.length - 1].state[0].join(': ');
  console.log((err ? 'ERR ' : 'ok  ') + f.padEnd(40), 'steps', S.length, extra); if (err) bad++;
  if (verbose && f === verbose + '.json') S.forEach((s, t) => console.log(String(t).padStart(2), JSON.stringify(s.lines), s.head, '|', s.note, '|', s.state.map(x => x.join('=')).join('; ')));
}
console.log(bad ? 'PROBLEMS: ' + bad : 'ALL OK');
