const P = require('./pseudo_block.js'), fs = require('fs'); const dir = __dirname + '/inst/'; let bad = 0;
const chk = (c, msg) => { if (!c) { bad++; console.log('  MISMATCH', msg); } };
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort()) {
  const algo = f.replace(/(_\d+)?\.json$/, ''), inst = JSON.parse(fs.readFileSync(dir + f)), n = inst.n, adj = inst.adj, A = (inst.inputs || {}).A || adj, S = P.steps(algo, inst), last = S[S.length - 1];
  const stateOf = k => { for (const s of S) for (const [a, b] of s.state) if (a === k) var v = b; return v; };
  if (algo === 'strongly_connected_components') {
    const reach = adj.map((r, i) => r.map((x, j) => i === j || x > 0)); for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (reach[i][k] && reach[k][j]) reach[i][j] = true;
    const truth = new Set(); for (let i = 0; i < n; i++) truth.add([...Array(n).keys()].filter(j => reach[i][j] && reach[j][i]).join(','));
    const got = new Set((last.state.slice(-1)[0][1].match(/\{[^}]*\}/g) || []).map(s => s.slice(1, -1).split(',').map(Number).sort((a, b) => a - b).join(',')));
    chk(truth.size === got.size && [...truth].every(x => got.has(x)), f + ' SCC got ' + [...got] + ' truth ' + [...truth]);
  }
  if (algo === 'topological_sort') {
    const ord = last.state[0][1].split(' → ').map(Number), pos = {}; ord.forEach((v, i) => pos[v] = i);
    chk(ord.length === n, f + ' topo length'); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j && adj[i][j] > 0) chk(pos[i] < pos[j], f + ` topo edge ${i}->${j}`);
  }
  if (algo === 'mst_prim' || algo === 'mst_kruskal') {
    const edges = (stateOf('tree edges so far') || stateOf('edges in tree') || '').split(', ').filter(Boolean).map(s => s.split(/[–]/).map(Number));
    const E = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (adj[i][j] > 0) E.push([A[i][j], i, j]); E.sort((a, b) => a[0] - b[0]);
    const p = [...Array(n).keys()], fd = x => { while (p[x] !== x) x = p[x]; return x; }; let w = 0, c = 0; for (const [ww, i, j] of E) if (fd(i) !== fd(j)) { p[fd(i)] = fd(j); w += ww; c++; }
    const gw = edges.reduce((s, [a, b]) => s + A[a][b], 0);
    console.log('  ', f, algo, 'tree edges', edges.length, 'expected', c, 'weight', gw.toFixed(3), 'expected', w.toFixed(3));
    const mst = new Set(); { const q = [...Array(n).keys()], g = x => { while (q[x] !== x) x = q[x]; return x; }; for (const [ww, i, j] of E) if (g(i) !== g(j)) { q[g(i)] = g(j); mst.add(i + ',' + j); } }
    const sub = edges.every(([a, b]) => mst.has(Math.min(a, b) + ',' + Math.max(a, b)));
    if (algo === 'mst_kruskal') chk(edges.length === c && Math.abs(gw - w) < 1e-6, f + ' MST differs'); else chk(sub, f + ' Prim edge not in the MST');
  }
  if (algo === 'dijkstra' || algo === 'bellman_ford') {
    const d = Array(n).fill(Infinity); d[inst.source] = 0; for (let it = 0; it < n; it++) for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) if (u !== v && adj[u][v] > 0 && d[u] + A[u][v] < d[v]) d[v] = d[u] + A[u][v];
    const row = inst.target_true[inst.target_true.length - 1]; const ok = d.every((x, i) => !isFinite(x) ? row[i] === 0 : Math.abs(row[i] - x) < 1e-4);
    chk(ok, f + ' final distances differ from true'); 
  }
}
console.log(bad ? 'MISMATCHES ' + bad : 'GROUND TRUTH OK');
