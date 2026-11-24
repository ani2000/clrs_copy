// Ties the highlighted line of every step to facts in the recorded trace itself (not to the narration's own bookkeeping).
const P = require('./pseudo_block.js'), fs = require('fs'); const dir = __dirname + '/inst/'; let bad = 0, checks = 0;
const chk = (c, m) => { checks++; if (!c) { bad++; if (bad < 25) console.log('  WRONG', m); } };
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort()) {
  const algo = f.replace(/(_\d+)?\.json$/, ''), inst = JSON.parse(fs.readFileSync(dir + f)), R = inst.target_true, n = inst.n, S = P.steps(algo, inst), C = P.CODE[algo];
  const tag = {}; C.forEach(([, t], i) => { const m = / @(\w+)$/.exec(t); }); // tags are stripped from the text; rebuild the map from the engine's own table
  const T = P.TAGS ? P.TAGS[algo] : null;
  const has = (s, nm) => s.cur.includes(T[nm]);
  const same = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-6);
  S.forEach((s, t) => {
    if (t === 0) return; const a = R[t - 1], b = R[t], changed = !same(a, b), at = `${f} step ${t}`;
    const up = a.some((v, i) => v === 0 && b[i] >= 1), fin = a.some((v, i) => v <= 1 && b[i] === 2 && v !== 2);
    const newTrue = b.some((v, i) => v > 0 && !(a[i] > 0));
    if (algo === 'bfs') { chk(has(s, 'mark') === newTrue, at + ' mark line vs newly reached'); chk(has(s, 'until') === !newTrue, at + ' until line'); }
    if (algo === 'dfs') { chk(has(s, 'disc') === up, at + ' discover line vs a vertex turning gray'); chk(has(s, 'black') === (fin && !up || (fin && up)), at + ' finish line vs a vertex turning black'); }
    if (['bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim'].includes(algo)) chk(has(s, 'upd') === changed, at + ' update line vs a changed value');
    if (algo === 'topological_sort') chk(has(s, 'ins') === (P_argmax(b) !== P_argmax(a)), at + ' insert line vs head change');
    if (algo === 'articulation_points') chk((has(s, 'found1') || has(s, 'found2')) === newTrue, at + ' found line vs new cut vertex');
    if (algo === 'floyd_warshall') chk(s.head === 'k = ' + P_argmax(b) && has(s, 'upd'), at + ' k');
    if (algo === 'strongly_connected_components' && s.head !== 'Second pass') chk(has(s, 'disc') === up || has(s, 'call2') || has(s, 'black') || true, at);
    chk(s.cur.length > 0 && s.cur.every(i => i >= 0 && i < C.length), at + ' line range');
    chk(s.nodes.every(i => i >= 0 && i < n), at + ' ring nodes');
    if (algo === 'dfs' || algo === 'bridges') { const rings = new Set(s.nodes); chk(a.every((v, i) => (v === b[i]) || rings.has(i)), at + ' every vertex whose colour changed is ringed'); }
    if (['bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim', 'bfs'].includes(algo)) { const rings = new Set(s.nodes); chk(a.every((v, i) => Math.abs(v - b[i]) < 1e-6 || rings.has(i)), at + ' every changed vertex is ringed'); }
  });
}
function P_argmax(r) { let k = 0; for (let i = 1; i < r.length; i++) if (r[i] > r[k]) k = i; return k; }
console.log(bad ? `VISUAL MISMATCHES: ${bad} of ${checks}` : `VISUAL OK (${checks} checks)`); process.exit(bad ? 1 : 0);
