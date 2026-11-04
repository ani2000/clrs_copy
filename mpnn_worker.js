/* Web Worker: runs the in-browser MPNN forward pass and attribution graphs off the UI thread. */
importScripts('mpnn.js', 'attrib.js');
const MODELS = {}, TCS = {}, EXPL = new Map();   // EXPL: recent explanations, kept for interventions
function slimTarget(tg) { const o = Object.assign({}, tg); delete o.sel; return o; }
function pack(id, ex, ms, main, ptr) {
  const G = ex.G, Pr = ex.Pr, E = G.eW.length;
  const nodes = Object.values(G.nodes).map(nd => nd.type === 'logit' ? slimTarget(nd) : nd);
  // for the unpruned view send the strongest edges (by |w|); the pruned view is complete
  const order = Array.from({ length: E }, (_, k) => k).sort((a, b) => Math.abs(G.eW[b]) - Math.abs(G.eW[a])).slice(0, 6000);
  const top = order.map(e => ({ s: G.eS[e], d: G.eD[e], w: G.eW[e] }));
  return { type: 'attrib', id, nodes, pruned: { nodes: Pr.nodes, edges: Pr.edges }, top, influence: Pr.influence, totalEffect: Pr.totalEffect,
           keptInfluence: Pr.keptInfluence, nAll: Pr.nAll, eAll: Pr.eAll, targets: ex.targets.map(slimTarget), tLo: ex.tLo, ms, main, pointerField: ptr,
           rowLogits: Array.from(ex.run.rec[ex.run.rec.length - 1].logits[main]), rowPtr: ptr ? Array.from(ex.run.rec[ex.run.rec.length - 1].logits[ptr]) : null };
}
self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      MODELS[m.algo] = self.MPNN.loadModel(m.manifest, new Float32Array(m.buffer));
      self.postMessage({ type: 'loaded', algo: m.algo });
    } else if (m.type === 'tc') {
      TCS[m.algo] = self.Attrib.prepTranscoder(m.W);
      self.postMessage({ type: 'tcloaded', algo: m.algo });
    } else if (m.type === 'run') {
      const t0 = Date.now();
      const out = self.MPNN.runModel(MODELS[m.algo], m.inst);
      const T = out.z.length, N = m.inst.n, Z = new Float32Array(T * N * 256), S = new Float32Array(T * N * 128);
      for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) { Z.set(out.z[t][i], (t * N + i) * 256); S.set(out.msgs[t][i], (t * N + i) * 128); }
      self.postMessage({ type: 'result', id: m.id, T, N, z: Z, msgs: S, ms: Date.now() - t0 }, [Z.buffer, S.buffer]);
    } else if (m.type === 'attrib') {
      const t0 = Date.now(), M = MODELS[m.algo], T = TCS[m.algo];
      const ptr = M.fields.some(f => f.name === 'pi_h' && f.stage === 'hint' && f.loc === 'node' && f.type === 'pointer') ? 'pi_h' : null;
      const ex = self.Attrib.explain(self.MPNN, M, T, m.inst, { t: m.t, i: m.i, main: m.main, pointerField: ptr, window: m.window || 3 });
      EXPL.set(m.id, { ex, inst: m.inst, algo: m.algo }); while (EXPL.size > 4) EXPL.delete(EXPL.keys().next().value);
      self.postMessage(pack(m.id, ex, Date.now() - t0, m.main, ptr));
    } else if (m.type === 'intervene') {
      const rec = EXPL.get(m.ref); if (!rec) throw new Error('explanation expired; click the feature again');
      const iv = self.Attrib.intervene(self.MPNN, MODELS[rec.algo], TCS[rec.algo], rec.inst, rec.ex, m.t0, m.i0, m.q, m.factor);
      self.postMessage({ type: 'intervened', id: m.id, iv });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, algo: m.algo, error: String(err && err.stack || err) });
  }
};
