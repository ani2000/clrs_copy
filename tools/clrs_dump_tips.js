// Prints the hover-text tables of explain.js (GL and GL_EDGE) as JSON.   node tools/clrs_dump_tips.js <absolute path of explain.js>
// explain.js is a browser script; this loads it with a minimal fake browser so the tables can be read without a browser.
const noop = () => {}; const el = () => ({ addEventListener: noop, style: {}, classList: { add: noop, remove: noop, toggle: noop }, setAttribute: noop, appendChild: noop, querySelectorAll: () => [], closest: () => null });
global.window = { addEventListener: noop, AV_EDGE_ON: () => false }; global.document = { addEventListener: noop, createElement: el, body: el(), getElementById: () => null, querySelectorAll: () => [], documentElement: el() };
global.requestAnimationFrame = noop; global.getComputedStyle = () => ({}); global.innerWidth = 1440; global.innerHeight = 900; global.scrollX = 0; global.scrollY = 0;
require(process.argv[2]); const H = window.HoverCard; console.log(JSON.stringify({ GL: H.GL, GL_EDGE: H.GL_EDGE }));
