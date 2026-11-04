const { chromium } = require('playwright');
const BASE = process.env.BASE || 'http://127.0.0.1:8774', OUT = __dirname;
let pass = 0, fail = 0; const check = (n, ok, x = '') => { ok ? pass++ : fail++; console.log((ok ? 'PASS ' : 'FAIL ') + n + (x ? '  ' + x : '')); };
(async () => {
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errs = []; page.on('pageerror', e => errs.push(e.message)); page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.goto(BASE + '/viewer3.html');
  const t0 = Date.now();
  await page.waitForFunction(() => typeof AV !== 'undefined' && AV.data && document.querySelectorAll('#attribSvg .nd').length > 3, null, { timeout: 180000 });
  check('attribution graph rendered from the real model', true, `${Date.now() - t0} ms after load`);
  const info = await page.evaluate(() => ({ q: document.getElementById('question').innerText, chips: document.getElementById('outprobs').innerText, stat: document.getElementById('gstat').innerText,
    rows: [...document.querySelectorAll('#attribSvg .rowlab')].map(t => t.textContent), cols: [...document.querySelectorAll('#attribSvg .collab')].map(t => t.textContent).slice(0, 5),
    feats: document.querySelectorAll('#attribSvg .nd circle').length, leaves: AV.gr.nodes.filter(n => n.type === 'leaf').length, edges: AV.edgeEls.length, pinned: AV.pinned, insp: document.getElementById('featDetail').innerText.slice(0, 80) }));
  console.log(JSON.stringify(info, null, 1));
  check('rows: predictions, then features/leaves per step', info.rows[0].startsWith('PREDICTIONS OF STEP') && info.rows.some(r => r.includes('FEATURES')) && info.rows.some(r => r.includes('LEAVES')));
  check('columns: bias + one per graph node', info.cols[0] === 'bias' && info.cols[1].startsWith('node 0'));
  check('question + output chips present', info.q.startsWith('Step') && info.chips.length > 0);
  check('most influential feature inspected on load', info.insp.length > 10);
  await page.screenshot({ path: OUT + '/grid_pruned.png' });
  const nd = page.locator('#attribSvg .nd').nth(2); const bb = await nd.boundingBox();
  await page.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2); await page.waitForTimeout(300);
  const hl = await page.evaluate(() => ({ hi: document.querySelectorAll('#attribSvg .edge.hi').length, dim: document.querySelectorAll('#attribSvg .edge.dim').length, card: !document.querySelector('.hc').hidden && document.querySelector('.hc').innerText.split('\n')[0] }));
  check('hover highlights connected edges + shows card', hl.hi > 0 && !!hl.card, JSON.stringify(hl));
  await page.mouse.move(5, 995);
  const btn = page.locator('#ivNeg');
  if (await btn.count()) { await btn.click(); await page.waitForFunction(() => document.getElementById('ivResult').innerText.includes('downstream') || document.getElementById('ivResult').innerText.includes('expired'), null, { timeout: 60000 });
    const ivt = await page.locator('#ivResult').innerText(); check('intervention re-runs the real model', ivt.includes('downstream'), ivt.replace(/\n/g, ' | ').slice(0, 220)); }
  else check('intervention button present', false);
  await page.locator('button[data-v="all"]').click(); await page.waitForTimeout(800);
  const allN = await page.evaluate(() => AV.edgeEls.length); check('All mode shows more edges', allN > info.edges, `${info.edges} -> ${allN}`);
  await page.screenshot({ path: OUT + '/grid_all.png' });
  await page.locator('button[data-v="supernodes"]').click(); await page.waitForTimeout(800);
  const sup = await page.evaluate(() => ({ super: AV.gr.super, cols: [...document.querySelectorAll('#attribSvg .collab')].map(t => t.textContent) }));
  check('Supernodes groups by label / probe', sup.super && sup.cols.includes('error'), sup.cols.slice(0, 8).join(', '));
  await page.screenshot({ path: OUT + '/grid_super.png' });
  await page.locator('button[data-v="bynode"]').click(); await page.locator('button[data-v="pruned"]').click(); await page.waitForTimeout(300);
  const beforeKey = await page.evaluate(() => AV.data.key);
  await page.locator('#graphSvg circle[data-tip^="node:"]').nth(3).click();
  await page.waitForFunction(k => AV.data && AV.data.key !== k, beforeKey, { timeout: 60000 });
  check('clicking a simulation node traces that node', true, await page.evaluate(() => AV.data.key));
  await page.evaluate(() => { const r = document.getElementById('stepRange'); r.value = Math.max(0, +r.max - 1); r.dispatchEvent(new Event('input')); });
  await page.waitForFunction(() => AV.data && AV.data.t === Math.min(curStep, G.length - 2), null, { timeout: 60000 });
  check('changing step re-traces', true, 'step ' + await page.evaluate(() => AV.data.t));
  for (const a of ['dfs', 'floyd_warshall', 'dijkstra', 'mst_kruskal']) {
    await page.selectOption('#inAlgo', a);
    await page.waitForFunction(x => AV.data && AV.data.key.startsWith(x + '|') && document.querySelectorAll('#attribSvg .nd').length > 2, a, { timeout: 180000 });
    const q = await page.evaluate(() => document.getElementById('question').innerText);
    check(`${a}: graph traced`, true, q.slice(0, 110));
    await page.screenshot({ path: `${OUT}/grid_${a}.png` });
  }
  check('no page errors', errs.length === 0, errs.slice(0, 3).join(' | '));
  console.log(`\n${pass} passed, ${fail} failed`); await browser.close(); process.exit(fail ? 1 : 0);
})().catch(e => { console.log('HARNESS ERROR', e.message); process.exit(2); });
