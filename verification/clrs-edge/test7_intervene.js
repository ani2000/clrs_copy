// Interventions on the real model must run for every algorithm and report a finite change for every target; with factor 1 nothing may change.
const fs = require('fs'), path = require('path'), ROOT = path.join(__dirname, '..', '..'), MODELS = process.env.MODELS, INST = process.env.INST;
const E = require(ROOT + '/clrs-edge-attrib.js'), MP = require(ROOT + '/mpnn.js'), AT = require(ROOT + '/attrib.js');
const algos = (process.env.ALGOS || 'bfs,dfs,bellman_ford,dijkstra,dag_shortest_paths,mst_prim,mst_kruskal,topological_sort,articulation_points,bridges,strongly_connected_components,floyd_warshall').split(','); let bad = 0;
for (const a of algos) {
  const man = JSON.parse(fs.readFileSync(path.join(MODELS, a + '.json'))), b = fs.readFileSync(path.join(MODELS, a + '.bin')), M = MP.loadModel(man, new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  const T = E.prepTC(JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.json'))), fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.bin')));
  const inst = JSON.parse(fs.readFileSync(path.join(INST, a + '_104.json'))), meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'algos', a, 'meta.json')));
  const ptr = M.fields.some(f => f.name === 'pi_h' && f.stage === 'hint' && f.loc === 'node' && f.type === 'pointer') ? 'pi_h' : null;
  const ex = E.explain(MP, AT, M, T, inst, { t: Math.min(2, inst.length - 2), i: 2, main: meta.target, pointerField: ptr, window: 3 }), pr = E.prune(ex), pk = E.pack(ex, pr, inst);
  const feats = pk.pruned.nodes.filter(id => ex.nodes[ex.nid.get(id)].type === 'feature').slice(0, 3); let ok = feats.length > 0, worst = 0;
  for (const id of feats) for (const factor of [1, 0, 2, -1]) {
    const iv = E.intervene(MP, M, T, inst, ex, pr, id, factor);
    for (const r of iv.rows) { if (![r.base, r.actual, r.predicted].every(Number.isFinite)) ok = false; if (factor === 1) worst = Math.max(worst, Math.abs(r.actual - r.base), Math.abs(r.predicted - r.base)); }
    if (!(iv.decisions > 0)) ok = false;
  }
  if (worst > 1e-9) ok = false; if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'BAD '} ${a.padEnd(30)} ${feats.length} latents x 4 factors; factor 1 changes nothing (max ${worst.toExponential(1)})`);
}
console.log(bad ? 'TEST 7 FAILED' : 'TEST 7 OK'); process.exit(bad ? 1 : 0);
