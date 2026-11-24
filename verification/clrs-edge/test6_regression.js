// Test 6 (handoff section 6): the previous (node-site) graph must be byte-identical before and after the edge-site change.
// OLD = a folder with mpnn.js and attrib.js from before (git show <old commit>:mpnn.js ...). MODELS = model bundles, REG = folder with w_<algo>.json (/api/weights) and i_<algo>.json (n=8, p=0.4, seed=42).
const fs = require('fs'), crypto = require('crypto'), path = require('path'); const R = path.join(__dirname, '..', '..') + '/', SC = '';
const code = { old: [require(process.env.OLD + '/mpnn.js'), require(process.env.OLD + '/attrib.js')], neu: [require(R + 'mpnn.js'), require(R + 'attrib.js')] };
let bad = 0;
for (const a of ['bfs', 'bellman_ford', 'dijkstra']) {
  const man = JSON.parse(fs.readFileSync(process.env.MODELS + '/' + a + '.json')), buf = fs.readFileSync(process.env.MODELS + '/' + a + '.bin'), f32 = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const Wt = JSON.parse(fs.readFileSync(process.env.REG + '/w_' + a + '.json')), inst = JSON.parse(fs.readFileSync(process.env.REG + '/i_' + a + '.json')), meta = JSON.parse(fs.readFileSync(R + 'algos/' + a + '/meta.json'));
  const hashes = {};
  for (const k of ['old', 'neu']) {
    const [MP, AT] = code[k], M = MP.loadModel(man, f32), T = AT.prepTranscoder(Wt);
    const ptr = M.fields.some(f => f.name === 'pi_h' && f.stage === 'hint' && f.loc === 'node' && f.type === 'pointer') ? 'pi_h' : null;
    const ex = AT.explain(MP, M, T, inst, { t: Math.min(2, inst.length - 2), i: 3, main: meta.target, pointerField: ptr, window: 3 });
    const h = crypto.createHash('sha256'); h.update(JSON.stringify(ex.G.nodes)); h.update(Buffer.from(Float64Array.from(ex.G.eW).buffer)); h.update(JSON.stringify([ex.G.eS, ex.G.eD])); h.update(JSON.stringify(ex.Pr.nodes)); h.update(JSON.stringify(ex.Pr.edges)); h.update(JSON.stringify(ex.Pr.totalEffect));
    hashes[k] = h.digest('hex').slice(0, 16) + ` (${ex.G.eW.length} edges, ${Object.keys(ex.G.nodes).length} nodes, pruned ${ex.Pr.nodes.length}/${ex.Pr.edges.length})`;
  }
  const same = hashes.old === hashes.neu; if (!same) bad++; console.log(same ? 'IDENTICAL' : 'DIFFERENT', a, hashes.old, same ? '' : '| ' + hashes.neu);
}
console.log(bad ? 'REGRESSION: the previous graph changed' : 'REGRESSION OK: previous graph identical for bfs, bellman_ford, dijkstra (n=8, p=0.4, seed=42)'); process.exit(bad ? 1 : 0);
