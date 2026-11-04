const fs = require('fs');
const APP = require('path').resolve(__dirname, '..'), BASE = 'http://127.0.0.1:8774';
const MP = require(APP + '/mpnn.js'), AT = require(APP + '/attrib.js');
const META = { bfs: ['reach_h','pi_h'], dfs: ['color','pi_h'], bellman_ford: ['d','pi_h'], dijkstra: ['d','pi_h'], dag_shortest_paths: ['d','pi_h'], mst_prim: ['key','pi_h'], mst_kruskal: ['u',null], topological_sort: ['topo_head_h',null], articulation_points: ['is_cut_h','pi_h'], bridges: ['color','pi_h'], strongly_connected_components: ['color',null], floyd_warshall: ['k',null] };
const clone = o => JSON.parse(JSON.stringify(o));
(async () => {
  let worst = {}, total = 0, bad = 0;
  for (const [algo, [main, ptr]] of Object.entries(META)) {
    const M = MP.loadModel(JSON.parse(fs.readFileSync(`${APP}/models/${algo}.json`)), fs.readFileSync(`${APP}/models/${algo}.bin`));
    const T = AT.prepTranscoder(await (await fetch(`${BASE}/api/weights?algo=${algo}`)).json());
    const inst = await (await fetch(`${BASE}/api/instance?algo=${algo}&n=8&p=0.4&seed=11`)).json();
    const t = Math.min(2, inst.length - 2), i = 3;
    const t0 = Date.now();
    const ex = AT.explain(MP, M, T, inst, { t, i, main, pointerField: ptr, window: 3 });
    const ms = Date.now() - t0;
    const H = M.H, N = inst.n, G = ex.G, eps = 1e-3;
    const frozen = ex.run.hintsRead;
    // value of a destination node in a (possibly perturbed) run
    function valueOf(run, dst) {
      const nd = G.nodes[dst];
      if (nd.type === 'feature') { const r = run.rec[nd.t]; const x = new Float64Array(3 * H); x.set(r.z.subarray(nd.i * 2 * H, (nd.i + 1) * 2 * H)); x.set(r.agg.subarray(nd.i * H, (nd.i + 1) * H), 2 * H); return AT.transcode(T, x, r.hNew.subarray(nd.i * H, (nd.i + 1) * H)).pre[nd.f]; }
      const r = run.rec[t], L = r.logits[nd.field]; if (nd.kind === 'pointer') return L[nd.i * N + nd.j]; const f = M.fields.find(x => x.name === nd.field); return L[nd.i * f.dec[0].dout + (nd.c || 0)];
    }
    const rerun = (opts, instX) => MP.runRecord(M, instX || inst, Object.assign({ record: { from: 0, to: t }, stopAt: t, hints: frozen }, opts));
    const base = rerun({});
    const sameMasks = (a, ts) => { for (let s = ts; s <= t; s++) { const A = a.rec[s], B = base.rec[s]; for (let k = 0; k < A.mask0.length; k++) if (A.mask0[k] !== B.mask0[k] || A.mask1[k] !== B.mask1[k]) return false; for (let k = 0; k < A.route.length; k++) if (A.route[k] !== B.route[k]) return false; } return true; };
    const E = G.eW.length, idx = [...Array(E).keys()].filter(e => !G.eS[e].startsWith('B'));
    const kinds = {}; let tested = 0;
    for (let tries = 0; tries < 4000 && tested < 60; tries++) {
      const e = idx[Math.floor(Math.random() * idx.length)], s = G.eS[e], d = G.eD[e], w = G.eW[e], sn = G.nodes[s];
      let pert = null, scale = 1, kind;
      if (sn.type === 'feature') { kind = 'feature'; const dy = new Float64Array(H); for (let c = 0; c < H; c++) dy[c] = eps * T.dec[sn.f * H + c]; pert = rerun({ patch: { t: sn.t, i: sn.i, dy } }); scale = 1 / sn.act; }
      else if (sn.type === 'error') { kind = 'error'; const tc = G.TC[sn.t][sn.i]; const dy = new Float64Array(H); for (let c = 0; c < H; c++) dy[c] = eps * tc.err[c]; pert = rerun({ patch: { t: sn.t, i: sn.i, dy } }); }
      else if (sn.type === 'leaf' && sn.stage === 'hint') {
        kind = 'leaf:hint:' + sn.loc + ':' + sn.ftype; const f = M.fields.find(x => x.name === sn.probe); if (sn.ftype === 'categorical' || (f.loc === 'edge' && f.type === 'pointer')) continue;
        const h2 = frozen.map(o => o); h2[sn.t] = Object.assign({}, frozen[sn.t]); const arr = Float64Array.from(h2[sn.t][sn.probe]);
        let k; if (sn.loc === 'graph') k = 0; else if (sn.j === null) k = sn.i; else k = sn.i * N + sn.j;
        const v0 = arr[k]; if (!v0) continue; arr[k] = v0 * (1 + eps); h2[sn.t][sn.probe] = arr; pert = rerun({ hints: h2 }); scale = 1 / v0;
      } else continue;
      if (!sameMasks(pert, 0)) continue;
      const fd = (valueOf(pert, d) - valueOf(base, d)) / eps;          // derivative w.r.t. the perturbation direction
      const expect = w * scale * (sn.type === 'leaf' ? (sn.value !== undefined ? 1 : 1) : 1);
      const lin = sn.type === 'leaf' ? fd * scale * 0 + fd : fd;       // leaf perturbation is v0*eps, so fd == w
      const exp2 = sn.type === 'leaf' ? w : expect;
      const err = Math.abs(lin - exp2) / Math.max(1e-3, Math.abs(exp2));
      (kinds[kind] = kinds[kind] || []).push(err); tested++; total++; if (err > 2e-2) { bad++; if (bad < 6) console.log('  MISMATCH', algo, s, '->', d, 'w', exp2.toFixed(5), 'fd', lin.toFixed(5)); }
    }
    const summary = Object.entries(kinds).map(([k, v]) => `${k}: n=${v.length} worst rel err ${Math.max(...v).toExponential(1)}`).join(' | ');
    console.log(`${algo.padEnd(22)} t=${t} nodes ${Object.keys(G.nodes).length} edges ${E} pruned ${ex.Pr.nodes.length}/${ex.Pr.edges.length} (${(ex.Pr.keptInfluence*100).toFixed(0)}%) build ${ms}ms\n    ${summary}`);
  }
  console.log(`\n${total} edges finite-difference checked, ${bad} mismatches (>2% rel)`);
})();
