// End-to-end integrity: for each of the 12 algorithms, everything the page shows must come from THAT algorithm.
const { chromium } = require('playwright');
const BASE = process.env.BASE || 'http://127.0.0.1:8774';
const ALGOS = ['bfs', 'dfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'mst_prim', 'mst_kruskal', 'topological_sort',
               'articulation_points', 'bridges', 'strongly_connected_components', 'floyd_warshall'];
let pass = 0, fail = 0; const check = (n, ok, x = '') => { ok ? pass++ : fail++; console.log((ok ? 'PASS ' : 'FAIL ') + n + (x ? '  ' + x : '')); };
(async () => {
  const b = await chromium.launch({ channel: 'chrome' }); const page = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  const errs = []; page.on('pageerror', e => errs.push(e.message)); page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.goto(BASE + '/viewer3.html');
  await page.waitForFunction(() => typeof AV !== 'undefined' && AV.data, null, { timeout: 180000 });
  const meta = await page.evaluate(() => fetch('/api/algos').then(r => r.json()));
  for (const a of ALGOS) {
    await page.selectOption('#inAlgo', a);
    // wait for: model bundle ready, a graph computed IN THE BROWSER, and its trace
    await page.waitForFunction(x => curAlgo === x && LOCAL.models[x] && LOCAL.models[x].state === 'ready', a, { timeout: 180000 });
    await page.evaluate(() => { const s = +document.getElementById('inSeed').value; document.getElementById('inSeed').value = String(s + 1); document.getElementById('inSeed').dispatchEvent(new Event('change')); });
    await page.waitForFunction(x => curAlgo === x && G && G._local && AV.data && AV.data.key.startsWith([x, G.seed, G.n, G.length].join('|') + '|'), a, { timeout: 180000 });
    const r = await page.evaluate(async x => {
      const n = G.n, p = +document.getElementById('inP').value, seed = G.seed;
      const [srv, inst, fl, man] = await Promise.all([fetch(`/api/graph?algo=${x}&n=${n}&p=${p}&seed=${seed}`).then(r => r.json()), fetch(`/api/instance?algo=${x}&n=${n}&p=${p}&seed=${seed}`).then(r => r.json()),
                                                    fetch(`/api/feature_labels?algo=${x}`).then(r => r.json()), fetch(`/models/${x}.json`).then(r => r.json())]);
      let md = 0; for (let t = 0; t < srv.z.length; t++) for (let i = 0; i < n; i++) { for (let c = 0; c < 256; c++) md = Math.max(md, Math.abs(srv.z[t][i][c] - G.z[t][i][c])); for (let c = 0; c < 128; c++) md = Math.max(md, Math.abs(srv.msgs[t][i][c] - G.msgs[t][i][c])); }
      const specIn = man.fields.filter(f => f.stage === 'input').map(f => f.name).sort(), specHint = man.fields.filter(f => f.stage === 'hint').map(f => f.name).sort();
      const step = AV.data.t, tt = G.target_true;
      const truthCells = [...document.querySelectorAll('#nodeTableBody tr')].map(tr => tr.children[3].textContent);
      const truthOk = truthCells.length === n && truthCells.every((c, i) => Math.abs(parseFloat(c) - tt[step + 1][i]) < 0.006);
      const fields = new Set(man.fields.map(f => f.name));
      return { W_target: W.target, W_type: W.target_type, manAlgo: man.algo, stepsLocal: G.length, stepsSrv: srv.length, maxDiff: md,
               instIn: Object.keys(G._inst.inputs).sort().join(','), specIn: specIn.join(','), instHint: Object.keys(G._inst.hints0).sort().join(','), specHint: specHint.join(','),
               trajOk: JSON.stringify(G.target_true) === JSON.stringify(inst.target_true), flOk: JSON.stringify(FL) === JSON.stringify(fl), flSites: FL.total_sites,
               attrMain: AV.data.main, attrFields: [...new Set(AV.data.targets.map(t => t.field))], attrFieldsOk: AV.data.targets.every(t => fields.has(t.field)),
               leafProbesOk: AV.data.nodes.filter(nd => nd.type === 'leaf' && nd.probe !== 'state').every(nd => fields.has(nd.probe)),
               truthOk, question: document.getElementById('question').innerText, dropdown: document.getElementById('inAlgo').value, status: document.getElementById('status').innerText };
    }, a);
    const exp = meta[a];
    check(`${a}: model manifest, dropdown, weights all say "${a}"`, r.manAlgo === a && r.dropdown === a && r.W_target === exp.target && r.W_type === exp.target_type, `target ${r.W_target} (${r.W_type})`);
    check(`${a}: in-browser activations == server JAX run of ${a}'s checkpoint`, r.stepsLocal === r.stepsSrv && r.maxDiff < 2e-3, `${r.stepsLocal} steps, max|diff| ${r.maxDiff.toExponential(1)}`);
    check(`${a}: instance fields == ${a}'s CLRS spec`, r.instIn === r.specIn && r.instHint === r.specHint, `hints ${r.instHint}`);
    check(`${a}: trajectory shown == CLRS reference (server instance)`, r.trajOk);
    check(`${a}: feature labels are ${a}'s own`, r.flOk, `${r.flSites} labelled sites`);
    check(`${a}: attribution traced ${a}'s target with ${a}'s fields`, r.attrMain === exp.target && r.attrFieldsOk && r.leafProbesOk, `targets ${r.attrFields.join('+')}`);
    check(`${a}: node table truth column == ${a}'s true hint`, r.truthOk);
    // play: steps advance through this algorithm's own trajectory, re-traced at each step
    const before = await page.evaluate(() => curStep);
    if (await page.evaluate(() => G.length > 3)) {
      await page.evaluate(() => { const r = document.getElementById('stepRange'); r.value = 0; r.dispatchEvent(new Event('input')); });
      await page.click('#btnPlay');
      await page.waitForFunction(() => curStep >= 2, null, { timeout: 30000 });
      await page.click('#btnPlay');
      const pl = await page.evaluate(x => {
        const circles = [...document.querySelectorAll('#graphSvg circle[data-tip^="node:"]')];
        const fills = circles.map((c, i) => c.getAttribute('fill') === (i === G.source ? '#f59e0b' : colorForNode(i, curStep)));
        return { step: curStep, colorsOk: fills.every(Boolean), algo: curAlgo };
      }, a);
      await page.waitForFunction(x => AV.data && AV.data.key.startsWith(x + '|') && AV.data.t === Math.min(curStep, G.length - 2), a, { timeout: 60000 });
      check(`${a}: Play advances ${a}'s own trajectory and re-traces`, pl.algo === a && pl.colorsOk && pl.step >= 2, `stopped at step ${pl.step}`);
    }
  }
  // rapid switching: no leakage between algorithms
  for (const x of ['dfs', 'bfs', 'dfs', 'floyd_warshall', 'dfs']) { await page.selectOption('#inAlgo', x); await page.waitForTimeout(120); }
  await page.waitForFunction(() => curAlgo === 'dfs' && G && AV.data && AV.data.key.startsWith('dfs|'), null, { timeout: 180000 });
  await page.waitForTimeout(2500);
  const leak = await page.evaluate(() => ({ W: W.target, keys: AV.data.key, main: AV.data.main, table: document.querySelector('#nodeTableHead').innerText, q: document.getElementById('question').innerText,
    leafOk: AV.data.nodes.filter(n => n.type === 'leaf' && n.probe !== 'state').every(n => ['pos','A','adj','pi_h','color','d','f','s_prev','s','u','v','s_last','time'].includes(n.probe)) }));
  check('rapid dfs→bfs→dfs→floyd→dfs switching ends fully on dfs (no leakage)', leak.W === 'color' && leak.main === 'color' && leak.table.includes('color') && leak.q.includes('color') && leak.leafOk, leak.keys);
  check('no page errors', errs.length === 0, errs.slice(0, 3).join(' | '));
  console.log(`\n${pass} passed, ${fail} failed`); await b.close(); process.exit(fail ? 1 : 0);
})().catch(e => { console.log('HARNESS ERROR', e.message); process.exit(2); });
