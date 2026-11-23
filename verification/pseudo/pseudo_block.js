/* ---------- pseudo-code pane: which line of the textbook algorithm each recorded step corresponds to ----------
   The model's trace gives, for every step, one "hint" row (reached set, DFS colours, distances, keys, current edge or
   intermediate vertex ...). Comparing a row with the previous one tells what the algorithm just did, so each step can
   be mapped to a pseudo-code line plus a one-sentence description and the main variables. Where the row alone is not
   enough (Dijkstra, Prim, Kruskal, bridges, articulation points) the missing detail is rebuilt from the graph and used
   only if it agrees with the recorded trace; otherwise the plain version is shown. Nothing here touches the model.
   Every step reports three things:
     cur  - the innermost line(s) being executed (shown in full colour; the colour is the nesting depth of that line)
     ctx  - the enclosing loops / tests and the call chain (shown only as a thin bar in their own depth colour)
     bind - the value each loop currently has (u = 3, v = 5, k = 2 ...), shown as a small tag on that line */
const PSEUDO = (function () {
  const EPS = 1e-6;
  const TAGS = {};
  // a line ends with "@tag" to give it a name; the narrators below refer to lines by name, never by number
  const L = (algo, ...a) => {
    const tags = {};
    const rows = a.map((s, i) => { const m = /^( *)(.*?)(?:\s+@(\w+))?$/.exec(s); if (m[3]) tags[m[3]] = i; return [m[1].length >> 1, m[2]]; });
    TAGS[algo] = tags; return rows;
  };
  const CODE = {
    bfs: L('bfs',
      'BFS(G, s) @h',
      '  for each vertex v:  reach[v] ← 0,  π[v] ← NIL @init',
      '  reach[s] ← 1;  π[s] ← s @src',
      '  repeat @rep',
      '    prev ← reach @prev',
      '    for each edge (u, v) in E @edge',
      '      if prev[u] = 1 @cond',
      '        if π[v] is unset:  π[v] ← u @par',
      '        reach[v] ← 1 @mark',
      '  until reach = prev @until'),
    dfs: L('dfs',
      'DFS(G) @h',
      '  for each vertex u:  color[u] ← WHITE,  π[u] ← NIL @init',
      '  time ← 0 @time0',
      '  for each vertex u @outer',
      '    if color[u] = WHITE @outerif',
      '      DFS-VISIT(G, u) @callroot',
      'DFS-VISIT(G, u) @vh',
      '  time ← time + 1;  d[u] ← time;  color[u] ← GRAY @disc',
      '  for each v in Adj[u] @adj',
      '    if color[v] = WHITE @white',
      '      π[v] ← u @par',
      '      DFS-VISIT(G, v) @callv',
      '  color[u] ← BLACK @black',
      '  time ← time + 1;  f[u] ← time @fin'),
    bellman_ford: L('bellman_ford',
      'BELLMAN-FORD(G, w, s) @h',
      '  for each vertex v:  d[v] ← ∞,  π[v] ← NIL @init',
      '  d[s] ← 0 @src',
      '  for round ← 1 to |V| − 1 @round',
      '    for each edge (u, v) in E @edge',
      '      if d[v] > d[u] + w(u, v) @cond',
      '        d[v] ← d[u] + w(u, v) @upd',
      '        π[v] ← u @par'),
    dijkstra: L('dijkstra',
      'DIJKSTRA(G, w, s) @h',
      '  for each vertex v:  d[v] ← ∞,  π[v] ← NIL @init',
      '  d[s] ← 0 @src',
      '  Q ← all vertices @q',
      '  while Q is not empty @while',
      '    u ← EXTRACT-MIN(Q) @ext',
      '    for each v in Adj[u] @adj',
      '      if d[v] > d[u] + w(u, v) @cond',
      '        d[v] ← d[u] + w(u, v) @upd',
      '        π[v] ← u @par'),
    dag_shortest_paths: L('dag_shortest_paths',
      'DAG-SHORTEST-PATHS(G, w, s) @h',
      '  topologically sort the vertices of G @topo',
      '  for each vertex v:  d[v] ← ∞;   d[s] ← 0 @init',
      '  for each vertex u, in topological order @outer',
      '    for each v in Adj[u] @adj',
      '      if d[v] > d[u] + w(u, v) @cond',
      '        d[v] ← d[u] + w(u, v);  π[v] ← u @upd'),
    mst_prim: L('mst_prim',
      'MST-PRIM(G, w, r) @h',
      '  for each vertex v:  key[v] ← ∞,  π[v] ← NIL @init',
      '  key[r] ← 0 @src',
      '  Q ← all vertices @q',
      '  while Q is not empty @while',
      '    u ← EXTRACT-MIN(Q)          // u joins the tree @ext',
      '    for each v in Adj[u] @adj',
      '      if v ∈ Q and w(u, v) < key[v] @cond',
      '        π[v] ← u @par',
      '        key[v] ← w(u, v) @upd'),
    mst_kruskal: L('mst_kruskal',
      'MST-KRUSKAL(G, w) @h',
      '  A ← ∅ @a',
      '  for each vertex v:  MAKE-SET(v) @make',
      '  sort the edges of E by weight @sort',
      '  for each edge (u, v) in E, lightest first @edge',
      '    if FIND-SET(u) ≠ FIND-SET(v) @cond',
      '      A ← A ∪ {(u, v)} @add',
      '      UNION(u, v) @union',
      '  return A @ret'),
    topological_sort: L('topological_sort',
      'TOPOLOGICAL-SORT(G) @h',
      '  list ← empty @l',
      '  for each vertex u @outer',
      '    if u is not visited yet @outerif',
      '      VISIT(u) @call',
      '  return list @ret',
      'VISIT(u) @vh',
      '  mark u as visited @vis',
      '  for each v in Adj[u] @adj',
      '    if v is not visited yet @white',
      '      VISIT(v) @callv',
      '  insert u at the front of list      // u has finished @ins'),
    articulation_points: L('articulation_points',
      'ARTICULATION-POINTS(G) @h',
      '  time ← 0;   visited[u] ← false for all u @init',
      '  for each vertex u @outer',
      '    if not visited[u] @outerif',
      '      DFS-AP(u, NIL) @call',
      'DFS-AP(u, p) @vh',
      '  visited[u] ← true;  d[u] ← low[u] ← ++time;  children ← 0 @disc',
      '  for each v in Adj[u] @adj',
      '    if v is not visited @tree',
      '      children ← children + 1 @kids',
      '      DFS-AP(v, u) @rec',
      '      low[u] ← min(low[u], low[v]) @lowupd',
      '      if p ≠ NIL and low[v] ≥ d[u] @cond1',
      '        u is an articulation point @found1',
      '    else if v ≠ p @back',
      '      low[u] ← min(low[u], d[v]) @backupd',
      '  if p = NIL and children ≥ 2 @cond2',
      '    u is an articulation point @found2'),
    bridges: L('bridges',
      'BRIDGES(G) @h',
      '  time ← 0;   visited[u] ← false for all u @init',
      '  for each vertex u @outer',
      '    if not visited[u] @outerif',
      '      DFS-BR(u, NIL) @callroot',
      'DFS-BR(u, p) @vh',
      '  visited[u] ← true;  d[u] ← low[u] ← ++time @disc',
      '  for each v in Adj[u] @adj',
      '    if v is not visited @tree',
      '      DFS-BR(v, u) @callv',
      '      low[u] ← min(low[u], low[v]) @lowupd',
      '      if low[v] > d[u] @cond',
      '        (u, v) is a bridge @found',
      '    else if v ≠ p @back',
      '      low[u] ← min(low[u], d[v]) @backupd'),
    strongly_connected_components: L('strongly_connected_components',
      'SCC(G) @h',
      '  for each vertex u:  color[u] ← WHITE @init',
      '  for each vertex u                       // pass 1 on G @outer',
      '    if color[u] = WHITE @outerif',
      '      DFS-VISIT(G, u) @callroot',
      '  Gᵀ ← G with every edge reversed @rev',
      '  for each vertex u:  color[u] ← WHITE @reset',
      '  for each u, in decreasing order of f[u]  // pass 2 on Gᵀ @outer2',
      '    if color[u] = WHITE @if2',
      '      comp ← new empty component @newc',
      '      DFS-VISIT(Gᵀ, u), adding every vertex it reaches to comp @call2',
      '  return the components @ret',
      'DFS-VISIT(H, u) @vh',
      '  color[u] ← GRAY @disc',
      '  for each v in Adj_H[u] @adj',
      '    if color[v] = WHITE @white',
      '      DFS-VISIT(H, v) @callv',
      '  color[u] ← BLACK @black',
      '  f[u] ← ++time                           // pass 1 only @fin'),
    floyd_warshall: L('floyd_warshall',
      'FLOYD-WARSHALL(W) @h',
      '  n ← rows(W);   D ← W @init',
      '  for k ← 1 to n @k',
      '    for i ← 1 to n @i',
      '      for j ← 1 to n @j',
      '        D[i][j] ← min(D[i][j],  D[i][k] + D[k][j]) @upd'),
  };

  const f2 = x => (Math.round(x * 100) / 100).toFixed(2);
  const list = a => a.length ? a.join(', ') : 'none';
  const argmax = r => { let b = 0; for (let i = 1; i < r.length; i++) if (r[i] > r[b]) b = i; return b; };

  function steps(algo, inst) {
    const R = inst.target_true, n = inst.n, adj = inst.adj, src = inst.source;
    const A = (inst.inputs && inst.inputs.A) || adj;
    const T = R ? R.length : 0, out = [];
    const code = CODE[algo], tg = TAGS[algo];
    const ix = nm => { if (!(nm in tg)) throw new Error('unknown line ' + nm); return tg[nm]; };
    const diff = t => { const o = []; if (t < 1) return o; for (let i = 0; i < n; i++) if (Math.abs(R[t][i] - R[t - 1][i]) > EPS) o.push({ i, a: R[t - 1][i], b: R[t][i] }); return o; };
    // cur: names of the innermost executing lines; x.ctx: extra enclosing lines; x.bind: {line name: 'u = 3'}
    const S = (cur, head, note, state, nodes, x) => {
      x = x || {};
      const c = [...new Set(cur.map(ix))].sort((a, b) => a - b), ctx = new Set((x.ctx || []).map(ix));
      for (const i of c) {                                  // every enclosing loop / test of a current line
        let d = code[i][0];
        for (let j = i - 1; j >= 0 && d > 0; j--) { const dj = code[j][0]; if (dj < d) { if (dj > 0) ctx.add(j); d = dj; } }
      }
      for (const i of c) ctx.delete(i);
      const bind = {}; for (const [k, v] of Object.entries(x.bind || {})) bind[ix(k)] = v;
      return { lines: c, cur: c, ctx: [...ctx].sort((a, b) => a - b), bind, head, note, state: state || [], nodes: nodes || [] };
    };
    const edge = (u, v) => adj[u][v] > 0;

    // ---- DFS-style walk shared by dfs, bridges, scc (replays the colour changes: gives the stack, times, parents)
    function colourWalk(mode) {
      const st = { stack: [], time: 0, d: {}, par: {}, low: {}, fin: [], bridges: [], comps: [], comp: null, phase: 1 };
      const rootNames = () => mode === 'scc' && st.phase === 2 ? ['outer2', 'if2', 'newc', 'call2'] : mode === 'dfs' || mode === 'scc' ? ['outer', 'outerif', 'callroot'] : ['outer', 'outerif', 'callroot'];
      for (let t = 0; t < T; t++) {
        if (t === 0) { out.push(S(mode === 'dfs' ? ['init', 'time0'] : ['init'], 'Start', mode === 'scc' ? 'Start the first depth-first search on the original graph (pass 1).' : 'Every vertex starts WHITE (not seen yet).', [['time', 0]], [])); continue; }
        const prev = R[t - 1], cur = R[t];
        if (mode === 'scc' && cur.every(v => v === 0) && prev.some(v => v !== 0)) {
          st.phase = 2; st.stack = []; st.comp = null;
          out.push(S(['rev', 'reset'], 'Second pass', 'First pass done. Reverse every edge and reset the colours; the second search takes vertices in decreasing finishing time f[u]. Each new tree will be one component.',
            [['finishing order (pass 1)', list(st.fin)], ['components', 0]], [])); continue;
        }
        const disc = [], fin = [];
        for (let i = 0; i < n; i++) { if (prev[i] === 0 && cur[i] >= 1) disc.push(i); if (prev[i] <= 1 && cur[i] === 2 && (prev[i] === 1 || cur[i] === 2)) fin.push(i); }
        const notes = [], curN = new Set(), ctxN = new Set(), bind = {}, nodes = [];
        const depth0 = st.stack.length;
        for (const v of fin) {
          const depth = st.stack.length;
          const k = st.stack.lastIndexOf(v); if (k >= 0) st.stack.splice(k, 1);
          st.time++; nodes.push(v);
          if (depth >= 2) ctxN.add('callv');
          if (mode === 'scc' && st.phase === 1) st.fin.push(v);
          if (mode === 'bridges') {
            let low = st.d[v]; const p = st.par[v];
            for (let w = 0; w < n; w++) if (edge(v, w) && w !== p && st.d[w] !== undefined) low = Math.min(low, st.d[w]);
            for (const c of Object.keys(st.par)) if (st.par[c] === v && st.low[c] !== undefined) low = Math.min(low, st.low[c]);
            st.low[v] = low;
            if (p !== undefined && p !== null) {
              const isB = low > st.d[p];
              if (isB) { curN.add('found'); st.bridges.push(`${p}–${v}`); bind.found = `(u, v) = (${p}, ${v})`; } else { curN.add('lowupd'); curN.add('cond'); bind.cond = `low[${v}] = ${low}, d[${p}] = ${st.d[p]}`; }
              bind.adj = `u = ${p}, v = ${v}`;
              notes.push(`Vertex ${v} is finished and control returns to its parent ${p}. low[${v}] = ${low}, d[${p}] = ${st.d[p]}, so the edge ${p}–${v} ${isB ? 'is a bridge (nothing below it reaches back above it)' : 'is not a bridge (there is another way around)'}.`);
            } else { curN.add('outer'); bind.outer = `u = ${v}`; notes.push(`Root ${v} is finished; it has no parent edge to test, so the outer loop moves on.`); }
          } else if (mode === 'dfs') { curN.add('black'); curN.add('fin'); bind.black = `u = ${v}`; notes.push(`Vertex ${v} has no unvisited neighbours left: it turns BLACK (finished), f[${v}] = ${st.time}.`); }
          else if (st.phase === 1) { curN.add('black'); curN.add('fin'); bind.black = `u = ${v}`; notes.push(`Vertex ${v} is finished (pass 1), f[${v}] = ${st.time}.`); }
          else {
            if (!st.stack.length && st.comp) { st.comps.push(st.comp.slice()); curN.add('call2'); ctxN.add('newc'); bind.call2 = `component {${st.comp.join(', ')}}`; notes.push(`The search tree is complete: vertices {${st.comp.join(', ')}} form one strongly connected component.`); st.comp = null; }
            else { curN.add('black'); bind.black = `u = ${v}`; notes.push(`Vertex ${v} is finished (pass 2).`); }
          }
          if (mode !== 'bridges' && depth >= 1) for (const r of rootNames()) ctxN.add(r);
          if (mode === 'bridges' && depth >= 2) ctxN.add('callv');
        }
        for (const v of disc) {
          const parent = st.stack.length ? st.stack[st.stack.length - 1] : null;
          st.time++; st.d[v] = mode === 'bridges' ? Object.keys(st.d).length + 1 : st.time; st.par[v] = parent; st.stack.push(v); nodes.push(v);
          curN.add('disc'); bind.disc = `u = ${v}`;
          for (const r of rootNames()) ctxN.add(r);
          if (parent === null) bind[mode === 'scc' && st.phase === 2 ? 'outer2' : 'outer'] = `u = ${v}`;
          else { for (const nm of [mode === 'bridges' ? 'tree' : 'white', 'adj', 'callv']) ctxN.add(nm); if (mode === 'dfs') ctxN.add('par'); bind.adj = `u = ${parent}, v = ${v}`; }
          if (mode === 'scc' && st.phase === 2) { if (parent === null) st.comp = [v]; else if (st.comp) st.comp.push(v); }
          notes.push(parent === null ? `Start a new search tree at vertex ${v}: it turns GRAY (discovered), d[${v}] = ${st.d[v]}.` :
            `Follow the edge ${parent}→${v}: vertex ${v} was WHITE${mode === 'dfs' ? `, so π[${v}] ← ${parent}` : ''}; the call DFS-VISIT(${v}) starts and ${v} turns GRAY, d[${v}] = ${st.d[v]}.`);
        }
        if (!disc.length && !fin.length) {
          const top = st.stack.length ? st.stack[st.stack.length - 1] : null;
          if (top === null) { curN.add(mode === 'scc' && st.phase === 2 ? 'if2' : 'outerif'); notes.push('Looking for the next WHITE vertex to start a new search from.'); }
          else {
            curN.add(mode === 'bridges' ? 'back' : 'white'); bind.adj = `u = ${top}`;
            if (st.stack.length >= 2) ctxN.add('callv');
            for (const r of rootNames()) ctxN.add(r);
            notes.push(`Checking the next neighbour of vertex ${top}: nothing changes (it was already seen, or the scan moves on).`);
          }
        }
        const state = [['search path (gray, top last)', st.stack.length ? st.stack.join(' → ') : 'empty'], ['time', st.time]];
        if (mode === 'bridges') state.push(['bridges found', list(st.bridges)]);
        if (mode === 'scc') state.push(st.phase === 1 ? ['finished so far (pass 1)', list(st.fin)] : ['components found', st.comps.length ? st.comps.map(c => '{' + c.join(',') + '}').join(' ') : 'none']);
        // bridges has no separate call-site lines for the root test; keep only names that exist in this algorithm
        const okN = nm => nm in tg;
        out.push(S([...curN].filter(okN), disc.length || fin.length ? (disc.length ? 'Discover' : 'Finish') : 'Scan', notes.join(' '), state, nodes, { ctx: [...ctxN].filter(okN), bind: Object.fromEntries(Object.entries(bind).filter(([k]) => okN(k))) }));
      }
    }

    // ---- relaxation algorithms
    const reached = (row, v) => v === src || row[v] > EPS;
    function explainFrom(u, ch, prev, weightOnly) {
      return ch.every(c => edge(u, c.i) && Math.abs((weightOnly ? 0 : prev[u]) + A[u][c.i] - c.b) < 1e-4);
    }
    function relaxText(ch, u) { return ch.map(c => `d[${c.i}]: ${reached(R[0], c.i) && c.a <= EPS && c.i !== src ? '∞' : f2(c.a)} → ${f2(c.b)}`).join(';  ') + (u !== null ? `  (via vertex ${u})` : ''); }

    function relax(kind) {
      const marked = new Set(), par = {}, treeEdges = [];
      for (let t = 0; t < T; t++) {
        if (t === 0) { out.push(S(kind === 'dag' ? ['topo', 'init'] : ['init', 'src'], 'Start',
          kind === 'bf' || kind === 'dag' || kind === 'dij' ? `Every distance is ∞ except d[${src}] = 0 (the source).` : `Every key is ∞ except key[${src}] = 0 (the start vertex).`, [], src !== null ? [src] : [], { bind: { [kind === 'dag' ? 'init' : 'src']: `s = ${src}` } })); continue; }
        const ch = diff(t), prev = R[t - 1];
        let u = null, silently = [];
        if (kind === 'dij' || kind === 'prim') {
          if (ch.length) {
            const cand = []; for (let v = 0; v < n; v++) if (!marked.has(v) && reached(prev, v)) cand.push(v);
            cand.sort((a, b) => (prev[a] - prev[b]) || (a - b));
            for (const c of cand) { if (explainFrom(c, ch, prev, kind === 'prim')) { u = c; break; } silently.push(c); }
            if (u === null) silently = [];
            else { for (const c of silently) marked.add(c); marked.add(u); }
          }
        } else if (kind === 'dag' || kind === 'bf') {
          if (ch.length) for (let c = 0; c < n && u === null; c++) if (reached(prev, c) && explainFrom(c, ch, prev, false)) u = c;
        }
        if (kind === 'prim' && ch.length) for (const c of ch) par[c.i] = u;
        let state = [], cur, head, note, ctx = [], bind = {};
        const ids = ch.map(c => c.i); const nodes = ids.slice(); if (u !== null) nodes.unshift(u);
        const idList = ids.join(', ');
        if (kind === 'bf') {
          bind.round = `round = ${t}`;
          if (ch.length) { cur = ['upd', 'par']; head = `Round ${t}`; note = `Round ${t}: every edge is checked. Improved: ${relaxText(ch, u === null || ch.length > 1 ? null : u)}.`; bind.edge = u !== null && ch.length === 1 ? `(u, v) = (${u}, ${ids[0]})` : `improving v = ${idList}`; }
          else { cur = ['cond']; head = `Round ${t}`; note = `Round ${t}: every edge is checked and the test d[v] > d[u] + w(u, v) never succeeds, so the distances are final.`; }
          state = [['distances found', R[t].filter((x, i) => reached(R[t], i)).length + ' of ' + n], ['improved this round', list(ids)]];
        } else if (kind === 'dag') {
          if (u !== null) bind.outer = `u = ${u}`;
          if (ch.length) { cur = ['upd']; head = 'Relax'; note = `Next vertex in topological order${u !== null ? ' is ' + u : ''}: its outgoing edges improve ${relaxText(ch, null)}.`; bind.adj = `v = ${idList}`; }
          else { cur = ['cond']; head = 'Next vertex'; note = 'Take the next vertex in topological order; none of its outgoing edges improves a distance.'; }
          state = [['distances found', R[t].filter((x, i) => reached(R[t], i)).length + ' of ' + n]];
        } else if (kind === 'dij') {
          if (u !== null) bind.while = `u = ${u}`;
          if (ch.length) { cur = ['upd', 'par']; ctx = ['ext']; head = 'Extract & relax'; note = u !== null ? `Take the closest unfinished vertex, ${u} (d = ${f2(prev[u])}), and relax its edges: ${relaxText(ch, null)}.` : `Take the closest unfinished vertex and relax its edges: ${relaxText(ch, null)}.`; bind.adj = `v = ${idList}`; }
          else { cur = ['cond']; ctx = ['ext']; head = 'Extract'; note = 'Take the closest unfinished vertex; the test d[v] > d[u] + w(u, v) fails for every neighbour, so nothing improves.'; }
          state = [['finished (extracted)', list([...marked])], ['distances found', R[t].filter((x, i) => reached(R[t], i)).length + ' of ' + n]];
        } else {
          for (const c of [...silently, u].filter(x => x !== null && x !== undefined)) if (par[c] !== undefined && par[c] !== null) treeEdges.push(`${par[c]}–${c}`);
          if (u !== null) bind.while = `u = ${u}`;
          if (ch.length) { cur = ['par', 'upd']; ctx = ['ext']; head = 'Extract & update'; note = u !== null ? `Take the vertex with the smallest key, ${u}, into the tree and update its neighbours: ${ch.map(c => `key[${c.i}] ← ${f2(c.b)}`).join(', ')}.` : `Update neighbours' keys: ${ch.map(c => `key[${c.i}] ← ${f2(c.b)}`).join(', ')}.`; bind.adj = `v = ${idList}`; }
          else { cur = ['cond']; ctx = ['ext']; head = 'Extract'; note = 'Take the vertex with the smallest key into the tree; no neighbour gets a smaller key, so the test w(u, v) < key[v] fails everywhere.'; }
          state = [['tree edges so far', list(treeEdges)], ['in tree', list([...marked])]];
        }
        out.push(S(cur, head, note, state, nodes, { ctx, bind }));
      }
    }

    function kruskal() {
      // Rebuild the edge order from the weights. Rows are grouped into blocks that share the same start vertex u; the
      // rebuilt order is used only for blocks where it agrees with the recorded trace (otherwise the plain text is shown).
      const E = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (adj[i][j] > 0) E.push([A[i][j], i, j]);
      E.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
      const p = [...Array(n).keys()]; const find = x => { while (p[x] !== x) x = p[x]; return x; };
      const sim = E.map(([w, i, j]) => { const a = find(i), b = find(j), acc = a !== b; if (acc) p[a] = b; return { w, u: i, v: j, acc }; });
      const sb = []; for (const e of sim) { if (sb.length && sb[sb.length - 1].u === e.u) sb[sb.length - 1].edges.push(e); else sb.push({ u: e.u, edges: [e] }); }
      const rb = []; for (let t = 1; t < T; t++) { const u = argmax(R[t]); if (rb.length && rb[rb.length - 1].u === u) rb[rb.length - 1].rows.push(t); else rb.push({ u, rows: [t] }); }
      let ok = 0; while (ok < rb.length && ok < sb.length && rb[ok].u === sb[ok].u) ok++;
      const name = e => `${e.u}–${e.v}`; const added = []; const rowBlock = {};
      rb.forEach((b, k) => b.rows.forEach((t, q) => { rowBlock[t] = { k, q, len: b.rows.length }; }));
      for (let t = 0; t < T; t++) {
        if (t === 0) { out.push(S(['a', 'make', 'sort'], 'Start', 'The tree A is empty; every vertex is in its own set. The edges are sorted by weight.', [['edges in tree', 'none']], [])); continue; }
        const u = argmax(R[t]), { k, q, len } = rowBlock[t];
        if (k >= ok) { out.push(S(q === 0 ? ['edge'] : ['cond'], 'Edge', `Look at the next lightest edge: it starts at vertex ${u}.`, [['edges in tree', list(added)]], [u], { bind: { edge: `(u, v) starts at ${u}` } })); continue; }
        const es = sb[k].edges;
        if (es.length === 1) {
          const e = es[0];
          if (q === 0) out.push(S(['edge'], 'Next edge', `Next lightest edge: ${name(e)} (weight ${f2(e.w)}). Are its two ends already in the same set?`, [['edges in tree', list(added)]], [e.u, e.v], { bind: { edge: `(u, v) = (${e.u}, ${e.v}),  w = ${f2(e.w)}` } }));
          else { if (e.acc && q === len - 1 && !added.includes(name(e))) added.push(name(e));
            out.push(S(e.acc ? ['add', 'union'] : ['cond'], e.acc ? 'Add edge' : 'Skip edge', e.acc ? `Different sets, so edge ${name(e)} joins the tree and the two sets are merged.` : `Same set already, so edge ${name(e)} would close a cycle and is skipped.`, [['edges in tree', list(added)]], [e.u, e.v], { ctx: ['edge'], bind: e.acc ? { add: `(${e.u}, ${e.v})`, edge: `(u, v) = (${e.u}, ${e.v})` } : { cond: `FIND-SET(${e.u}) = FIND-SET(${e.v})`, edge: `(u, v) = (${e.u}, ${e.v})` } })); }
        } else {
          if (q === len - 1) for (const e of es) if (e.acc && !added.includes(name(e))) added.push(name(e));
          out.push(S(['cond'], 'Edges', `The next lightest edges all start at vertex ${u}: ${es.map(e => `${name(e)} (${e.acc ? 'added' : 'skipped'})`).join(', ')}.`, [['edges in tree', list(added)]], [...new Set(es.flatMap(e => [e.u, e.v]))], { ctx: ['edge'], bind: { edge: `${es.length} edges from ${u}` } }));
        }
      }
    }

    function floyd() {
      // the trace has one row per stop-over vertex k = 0, 1, 2, ... (row t belongs to k = t)
      for (let t = 0; t < T; t++) {
        const k = argmax(R[t]);
        out.push(S(t === 0 ? ['init', 'upd'] : ['upd'], `k = ${k}`,
          (t === 0 ? 'D starts as the table of direct edge weights. ' : '') + `Stop-over vertex ${k}: for every pair (i, j) keep the shorter of "direct" and "through ${k}".`,
          [['stop-overs allowed so far', k === 0 ? 'vertex 0' : 'vertices 0 … ' + k]], [k],
          { bind: { k: `k = ${k}`, i: 'every i', j: 'every j', upd: `via vertex ${k}` } }));
      }
    }

    function topo() {
      const order = [];
      for (let t = 0; t < T; t++) {
        if (t === 0) { out.push(S(['l'], 'Start', 'The list is empty. Depth-first search begins; vertices are added to the front of the list as they finish.', [['list (front → back)', 'empty']], [])); continue; }
        const h = argmax(R[t]);
        if (h !== argmax(R[t - 1])) {
          order.unshift(h);
          out.push(S(['ins'], 'Finish', `Vertex ${h} just finished, so it goes to the front of the list.`, [['list (front → back)', order.join(' → ')]], [h], { bind: { ins: `u = ${h}` } }));
        } else out.push(S(t === T - 1 ? ['ret'] : ['white'], t === T - 1 ? 'Done' : 'Searching', t === T - 1 ? 'Every vertex has finished: the list is the topological order.' : 'The depth-first search is exploring (checking neighbours, going deeper); no vertex finished at this step.', [['list (front → back)', order.join(' → ')]], []));
      }
    }

    function artic() {
      // replay a DFS from vertex 0 (neighbours in index order) to learn why each vertex is a cut vertex; used only if the
      // replay finds exactly the same vertices as the recorded trace
      const kind = {}; { const d = Array(n).fill(-1), low = Array(n).fill(0); let time = 0;
        const go = (u, p) => { d[u] = low[u] = ++time; let kids = 0, cut1 = false;
          for (let v = 0; v < n; v++) if (edge(u, v) || edge(v, u)) { if (d[v] < 0) { kids++; go(v, u); low[u] = Math.min(low[u], low[v]); if (p >= 0 && low[v] >= d[u]) cut1 = true; } else if (v !== p) low[u] = Math.min(low[u], d[v]); }
          if (cut1) kind[u] = 'found1'; if (p < 0 && kids >= 2) kind[u] = 'found2'; };
        for (let u = 0; u < n; u++) if (d[u] < 0) go(u, -1); }
      const last = R[T - 1], truth = []; for (let i = 0; i < n; i++) if (last[i] > 0) truth.push(i);
      const known = truth.length === Object.keys(kind).length && truth.every(i => kind[i]);
      const found = [];
      for (let t = 0; t < T; t++) {
        if (t === 0) { out.push(S(['init'], 'Start', 'No vertex is marked yet.', [['articulation points', 'none']], [])); continue; }
        const nw = diff(t).filter(c => c.b > c.a).map(c => c.i);
        if (nw.length) {
          found.push(...nw);
          const cur = known ? [...new Set(nw.map(v => kind[v]))] : ['found1', 'found2'];
          const bind = {}; for (const v of nw) if (known) bind[kind[v]] = `u = ${v}`;
          const why = known ? nw.map(v => kind[v] === 'found2' ? `${v} is the root of the search and has two or more DFS children` : `a child of ${v} cannot reach above ${v} (low[child] ≥ d[${v}])`).join('; ') : '';
          out.push(S(cur, 'Found', `Vertex ${list(nw)} is an articulation point: removing it would split the graph into separate pieces${why ? ` (${why})` : ''}.`, [['articulation points', list(found)]], nw, { bind }));
        } else out.push(S(['adj'], 'Searching', 'The depth-first search keeps computing discovery times d[u] and low values; no new articulation point yet.', [['articulation points', list(found)]], []));
      }
    }

    function bfs() {
      for (let t = 0; t < T; t++) {
        const cur = R[t]; const cnt = cur.filter(v => v > 0).length;
        if (t === 0) { out.push(S(['init', 'src'], 'Start', `Only the source ${src} is reached.`, [['reached', cnt + ' of ' + n]], src !== null ? [src] : [], { bind: { src: `s = ${src}` } })); continue; }
        const nw = diff(t).filter(c => c.b > c.a).map(c => c.i);
        if (nw.length) {
          const pairs = nw.map(v => { for (let u = 0; u < n; u++) if (R[t - 1][u] > 0 && adj[u][v] > 0) return [u, v]; return [null, v]; });
          const via = pairs.map(([u, v]) => u === null ? String(v) : `${v} (from ${u})`);
          out.push(S(['par', 'mark'], 'Expand', `Every already-reached vertex passes on to its neighbours. Newly reached: ${via.join(', ')}.`, [['reached', cnt + ' of ' + n], ['new this step', list(nw)]], nw,
            { ctx: ['prev'], bind: { edge: pairs.map(([u, v]) => `(${u === null ? '?' : u}, ${v})`).join(' '), mark: `v = ${nw.join(', ')}` } }));
        } else out.push(S(['until'], 'Stop', 'Every edge was scanned and nothing new was reached, so reach equals prev and the loop stops.', [['reached', cnt + ' of ' + n]], [], { ctx: ['edge'] }));
      }
    }

    if (!R || !T) return null;
    switch (algo) {
      case 'bfs': bfs(); break;
      case 'dfs': colourWalk('dfs'); break;
      case 'bridges': colourWalk('bridges'); break;
      case 'strongly_connected_components': colourWalk('scc'); break;
      case 'bellman_ford': relax('bf'); break;
      case 'dijkstra': relax('dij'); break;
      case 'dag_shortest_paths': relax('dag'); break;
      case 'mst_prim': relax('prim'); break;
      case 'mst_kruskal': kruskal(); break;
      case 'topological_sort': topo(); break;
      case 'articulation_points': artic(); break;
      case 'floyd_warshall': floyd(); break;
      default: return null;
    }
    return out;
  }

  const cache = new WeakMap();
  function forStep(algo, inst, step) {
    if (!inst || !CODE[algo]) return null;
    let s = cache.get(inst);
    if (!s || s.algo !== algo) { let arr = null; try { arr = steps(algo, inst); } catch (e) { arr = null; } s = { algo, arr }; cache.set(inst, s); }
    return s.arr ? s.arr[Math.max(0, Math.min(step, s.arr.length - 1))] : null;
  }
  return { CODE, forStep, steps };
})();
if (typeof module !== 'undefined') module.exports = PSEUDO;
