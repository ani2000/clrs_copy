// Cross-browser check (Chromium, Firefox, WebKit = Safari engine) with Playwright, run on a GitHub runner.
// usage: node tools/browsers_check.mjs https://address
import { chromium, firefox, webkit } from 'playwright';

const base = (process.argv[2] || '').replace(/\/+$/, '').replace(/\/viewer3\.html$/, '');
if (!/^https?:\/\//.test(base)) { console.error('usage: browsers_check.mjs <https://address>'); process.exit(2); }
const ALGOS = ['bfs', 'dfs', 'bridges', 'mst_kruskal', 'floyd_warshall'];
let ok = true;
const configs = [
  ['Chromium desktop', chromium, { viewport: { width: 1280, height: 800 } }],
  ['Firefox desktop', firefox, { viewport: { width: 1280, height: 800 } }],
  ['WebKit (Safari) desktop', webkit, { viewport: { width: 1280, height: 800 } }],
  ['WebKit (iPhone size)', webkit, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 3 }],
];
for (const [name, type, opts] of configs) {
  const browser = await type.launch();
  const ctx = await browser.newContext(opts);
  const page = await ctx.newPage();
  const problems = [];
  page.on('pageerror', e => problems.push('page error: ' + String(e.message).split('\n')[0].slice(0, 140)));
  page.on('console', m => { if (m.type() === 'error') problems.push('console: ' + m.text().slice(0, 140)); });
  page.on('requestfailed', r => { if (!/ERR_ABORTED|NS_BINDING_ABORTED|cancelled|aborted/i.test(String(r.failure() && r.failure().errorText))) problems.push('request failed: ' + r.url().slice(-60)); });
  const rows = []; let good = true;
  const t0 = Date.now();
  await page.goto(base + '/viewer3.html');
  for (const a of ALGOS) {
    const t = Date.now();
    try {
      if (a !== 'bfs') await page.selectOption('#inAlgo', a);
      await page.waitForFunction(a => curAlgo === a && G && !G._preview && document.querySelectorAll('#graphSvg circle').length > 0, a, { timeout: 90000, polling: 25 });
      const g = Date.now() - t;
      await page.waitForFunction(a => !!(AV.data && AV.data.key.startsWith(a + '|')), a, { timeout: 90000, polling: 25 });
      const at = Date.now() - t;
      const k = await page.evaluate(() => (G.n > 1 ? ((selected ? selected.node : 0) + 1) % G.n : 0));
      const tc = Date.now();
      await page.evaluate(k => selectTarget(k), k);
      await page.waitForFunction(([a, k]) => !!(AV.data && AV.data.i === k && AV.data.key.startsWith(a + '|')), [a, k], { timeout: 60000, polling: 25 });
      rows.push(`${a.padEnd(14)} graph ${String(g).padStart(6)} ms  attribution ${String(at).padStart(6)} ms  click ${String(Date.now() - tc).padStart(5)} ms  PASS`);
    } catch (e) { good = false; rows.push(`${a.padEnd(14)} FAILED (${String(e.message).split('\n')[0].slice(0, 80)})`); }
  }
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
  console.log(`\n== ${name}  (${(Date.now() - t0) / 1000} s)`); rows.forEach(r => console.log('  ' + r));
  if (opts.isMobile && overflow) { problems.push('page is wider than the phone screen'); }
  if (problems.length) { console.log('  problems:'); [...new Set(problems)].slice(0, 8).forEach(p => console.log('    ' + p)); }
  if (!good || problems.length) ok = false;
  console.log(`  ${good && !problems.length ? 'PASS' : 'FAIL'}`);
  await browser.close();
}
console.log(ok ? '\nRESULT: PASS in every browser' : '\nRESULT: FAIL');
process.exit(ok ? 0 : 1);
