// Fails if the pseudo-code engine inside viewer3.html is not identical to pseudo_block.js (the file the tests run on).
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', '..', 'viewer3.html'), 'utf8');
const core = fs.readFileSync(path.join(__dirname, 'pseudo_block.js'), 'utf8').replace("if (typeof module !== 'undefined') module.exports = PSEUDO;\n", '');
if (!html.includes('<script>\n' + core + '</script>')) { console.log('MISMATCH: the engine in viewer3.html differs from verification/pseudo/pseudo_block.js'); process.exit(1); }
console.log('inline engine == pseudo_block.js');
