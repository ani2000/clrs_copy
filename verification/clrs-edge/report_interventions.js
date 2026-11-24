// Not a gate: how often does the graph's linear prediction of a clamp agree in SIGN with the real model, and how large is the typical miss? (printed per algorithm)
const fs = require('fs'), path = require('path'), ROOT = path.join(__dirname, '..', '..'), MODELS = process.env.MODELS, INST = process.env.INST;
const E = require(ROOT + '/clrs-edge-attrib.js'), MP = require(ROOT + '/mpnn.js'), AT = require(ROOT + '/attrib.js');
for (const a of (process.env.ALGOS || 'bfs,bellman_ford,dijkstra,dfs,mst_kruskal,floyd_warshall').split(',')) {
  const man = JSON.parse(fs.readFileSync(path.join(MODELS, a + '.json'))), b = fs.readFileSync(path.join(MODELS, a + '.bin')), M = MP.loadModel(man, new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  const T = E.prepTC(JSON.parse(fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.json'))), fs.readFileSync(path.join(ROOT, 'edge_tc', a + '.bin')));
  let agree = 0, tot = 0, ratios = [], agreeF = 0, totF = 0, ratiosF = [];
  for (const f of fs.readdirSync(INST).filter(f => f.startsWith(a + '_1'))) {
    const inst = JSON.parse(fs.readFileSync(path.join(INST, f))), meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'algos', a, 'meta.json')));
    const ptr = M.fields.some(x => x.name === 'pi_h' && x.stage === 'hint' && x.loc === 'node' && x.type === 'pointer') ? 'pi_h' : null;
    const ex = E.explain(MP, AT, M, T, inst, { t: Math.min(2, inst.length - 2), i: 2, main: meta.target, pointerField: ptr, window: 3 }), pr = E.prune(ex), pk = E.pack(ex, pr, inst);
    const feats = pk.nodes.filter(n => n.type === 'feature' && pk.pruned.nodes.includes(n.id)).sort((x, y) => Math.abs(pk.influence[y.id]) - Math.abs(pk.influence[x.id])).slice(0, 4);
    for (const n of feats) { const iv = E.intervene(MP, M, T, inst, ex, pr, n.id, 0);
      for (const r of iv.rows) { const da = r.actual - r.base, dp = r.predicted - r.base; if (Math.abs(da) > 1e-3 && Math.abs(dp) > 1e-3) { tot++; if (Math.sign(da) === Math.sign(dp)) agree++; ratios.push(dp / da); }
        const dfx = r.actualFixed - r.base; if (Math.abs(dfx) > 1e-3 && Math.abs(dp) > 1e-3) { totF++; if (Math.sign(dfx) === Math.sign(dp)) agreeF++; ratiosF.push(dp / dfx); } } }
  }
  ratios.sort((x, y) => x - y); ratiosF.sort((x, y) => x - y); const med = ratios.length ? ratios[Math.floor(ratios.length / 2)] : NaN, medF = ratiosF.length ? ratiosF[Math.floor(ratiosF.length / 2)] : NaN;
  console.log(`${a.padEnd(22)} sign agreement ${agree}/${tot} (${tot ? (100 * agree / tot).toFixed(0) : '-'}%), median predicted/actual ${med.toFixed(2)}  |  with the hints held fixed: ${agreeF}/${totF} (${totF ? (100 * agreeF / totF).toFixed(0) : '-'}%), median ${medF.toFixed(2)}`);
}
