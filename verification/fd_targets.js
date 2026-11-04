const fs = require('fs');
const APP = require('path').resolve(__dirname, '..'), BASE = 'http://127.0.0.1:8774';
const MP = require(APP + '/mpnn.js'), AT = require(APP + '/attrib.js');
(async () => {
  let checked = 0, bad = 0;
  for (const [algo, main, ptr] of [['bfs','reach_h','pi_h'], ['dfs','color','pi_h'], ['bellman_ford','d','pi_h'], ['dijkstra','d','pi_h'], ['dag_shortest_paths','d','pi_h'], ['mst_prim','key','pi_h'], ['mst_kruskal','u',null], ['topological_sort','topo_head_h',null], ['articulation_points','is_cut_h','pi_h'], ['bridges','color','pi_h'], ['strongly_connected_components','color',null], ['floyd_warshall','k',null]]) {
    const M = MP.loadModel(JSON.parse(fs.readFileSync(`${APP}/models/${algo}.json`)), fs.readFileSync(`${APP}/models/${algo}.bin`));
    const T = AT.prepTranscoder(await (await fetch(`${BASE}/api/weights?algo=${algo}`)).json());
    const inst = await (await fetch(`${BASE}/api/instance?algo=${algo}&n=8&p=0.5&seed=5`)).json();
    for (const i of [2, 5]) {
      const ex = AT.explain(MP, M, T, inst, { t: 0, i, main, pointerField: ptr, window: 1 });
      const H = M.H, N = inst.n, G = ex.G, eps = 1e-5, frozen = ex.run.hintsRead, base = ex.run.rec[0];
      // target logit with the new hidden state h' held at its original value (graph nodes = features + errors carry that path)
      const logitWith = (r, tg) => {
        const f = M.fields.find(x => x.name === tg.field);
        const hT = k => { const v = new Float64Array(3 * H); v.set(r.hT.subarray(k * 3 * H, k * 3 * H + 2 * H)); v.set(base.hNew.subarray(k * H, (k + 1) * H), 2 * H); return v; };
        if (tg.kind === 'pointer') { const d = f.dec, p1 = MP.linRows(hT(tg.i), 1, d[0]), p2 = MP.linRows(hT(tg.j), 1, d[1]), p3 = MP.linRows(r.edgeF.subarray((tg.j * N + tg.i) * H, (tg.j * N + tg.i + 1) * H), 1, d[2]);
          let s = d[3].b[0]; for (let c = 0; c < H; c++) s += (tg.sel[c] ? p1[c] : p2[c] + p3[c]) * d[3].w[c]; return s; }
        return MP.linRows(hT(tg.i), 1, f.dec[0])[tg.c || 0];
      };
      for (const tg of ex.targets) {
        const chk = logitWith(base, tg); if (Math.abs(chk - G.nodes[tg.id].logit) > 1e-6) { console.log('  base logit mismatch', tg.id); bad++; }
        for (let e = 0; e < G.eW.length; e++) {
          if (G.eD[e] !== tg.id) continue; const sn = G.nodes[G.eS[e]]; if (!sn || sn.type !== 'leaf' || sn.ftype === 'categorical') continue;
          let pert;
          if (sn.stage === 'input') { const inst2 = JSON.parse(JSON.stringify(inst)); const arr = inst2.inputs[sn.probe]; if (sn.j === null) arr[sn.i] += eps; else arr[sn.i][sn.j] += eps;
            pert = MP.runRecord(M, inst2, { record: { from: 0, to: 0 }, stopAt: 0, hints: frozen }); }
          else { const f = M.fields.find(x => x.name === sn.probe); if (f.loc === 'edge' && f.type === 'pointer') continue;
            const h2 = [Object.assign({}, frozen[0])]; const arr = Float64Array.from(h2[0][sn.probe]); const k = sn.loc === 'graph' ? 0 : sn.j === null ? sn.i : sn.i * N + sn.j; arr[k] += eps; h2[0][sn.probe] = arr;
            pert = MP.runRecord(M, inst, { record: { from: 0, to: 0 }, stopAt: 0, hints: h2 }); }
          const fd = (logitWith(pert.rec[0], tg) - chk) / eps * (sn.value || 1);   // edge = value x derivative
          const w = G.eW[e], rel = Math.abs(fd - w) / Math.max(1e-3, Math.abs(w)); checked++;
          if (rel > 1e-3) { bad++; if (bad < 6) console.log('  MISMATCH', algo, G.eS[e], '->', tg.id, 'w', w.toFixed(6), 'fd', fd.toFixed(6)); }
        }
      }
    }
  }
  console.log(`${checked} leaf->target edges checked (h' held fixed), ${bad} mismatches`);
})();
