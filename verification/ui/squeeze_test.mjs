// Code lines of the pseudo-code pane must stay readable even with absurdly long tags, at five widths.   env: URL
// Forces absurdly long tags on every line and checks that the code text is still readable at the narrowest layouts.
import { spawn } from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
const sleep=ms=>new Promise(r=>setTimeout(r,ms)); const port=9100+Math.floor(Math.random()*300);
const p=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu',`--remote-debugging-port=${port}`,`--user-data-dir=${fs.mkdtempSync(path.join(os.tmpdir(),'is-'))}`,'about:blank'],{stdio:'ignore'});
let ws; for(let i=0;i<100&&!ws;i++){try{ws=(await(await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl}catch{await sleep(150)}}
const sock=new WebSocket(ws); await new Promise(r=>sock.addEventListener('open',r)); let id=0; const cb=new Map();
sock.addEventListener('message',m=>{const d=JSON.parse(m.data); if(d.id&&cb.has(d.id)){cb.get(d.id)(d.result);cb.delete(d.id)}});
const send=(method,params={},sessionId)=>new Promise(r=>{const i=++id;cb.set(i,r);sock.send(JSON.stringify({id:i,method,params,sessionId}))});
const {targetId}=await send('Target.createTarget',{url:'about:blank'}); const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true}); const S=(m,p)=>send(m,p,sessionId);
await S('Runtime.enable'); const ev=async e=>(await S('Runtime.evaluate',{expression:e,returnByValue:true,awaitPromise:true})).result?.value;
let bad=0;
for (const [w,h,m] of [[1440,900,false],[1181,800,false],[1024,800,false],[390,844,true],[320,700,true]]) {
  await S('Emulation.setDeviceMetricsOverride',{width:w,height:h,deviceScaleFactor:1,mobile:m});
  await S('Page.navigate',{url:''+(process.env.URL||'http://127.0.0.1:8792/viewer3.html')+''});
  for(let i=0;i<300;i++){ if(await ev(`typeof G!=='undefined'&&G&&!G._preview&&document.querySelectorAll('#pcCode .pl').length>0`)) break; await sleep(100); }
  const r=await ev(`(() => { const out=[]; document.querySelectorAll('#pcCode .pl').forEach(l => { const b=l.querySelector('.bd'); b.hidden=false; b.textContent='(12, 1) (3, 2) (3, 4) (6, 8) (15, 9) (0, 11) (7, 7) (1, 1)'; l.classList.add('on'); });
    document.querySelectorAll('#pcCode .pl .tx').forEach(x => { const r=x.getBoundingClientRect(); const lh=parseFloat(getComputedStyle(x).lineHeight)||18; if (r.width<70 || r.height>lh*5) out.push([x.textContent.slice(0,20), Math.round(r.width), Math.round(r.height)]); });
    return {bad: out, panel: document.getElementById('pnlCode').getBoundingClientRect().width|0, over: document.documentElement.scrollWidth - innerWidth}; })()`);
  console.log(w+'px', JSON.stringify(r)); if (r.bad.length||r.over>0) bad++;
}
console.log(bad?'SQUEEZE PROBLEMS':'SQUEEZE OK'); p.kill(); process.exit(bad?1:0);
