import re

with open('index_backup.html', 'r', encoding='utf-8') as f:
    html = f.read()

new_css = """
  :root{
    --bg:#8fb974; --panel:#ffffff; --panel2:#f0f0f0; --line:#333;
    --txt:#000; --dim:#555; --accent:#2a5d24; --p0:#3a6df0; --p1:#e0533d;
    --hpgood:#3fb950; --hpmid:#d29922; --hpbad:#f85149;
  }
  *{box-sizing:border-box;}
  body{margin:0;font-family:"Segoe UI",Meiryo,system-ui,sans-serif;background:var(--bg);color:var(--txt);font-size:13px;}
  header{position:sticky;top:0;z-index:5;background:#2c3a26;color:white;border-bottom:1px solid var(--line);
         padding:8px 12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
  header h1{font-size:15px;margin:0;color:white;white-space:nowrap;}
  header .meta, header .meta b{color:white;}
  button{background:#555;color:white;border:1px solid #222;border-radius:6px;
         padding:5px 11px;cursor:pointer;font-size:13px;}
  button:hover{background:#777;}
  button:disabled{opacity:.4;cursor:default;}
  #slider{flex:1;min-width:160px;}
  
  #wrap{display:grid;grid-template-columns:300px 1fr;gap:20px;padding:10px;align-items:start;height:calc(100vh - 50px);}
  @media(max-width:1000px){#wrap{grid-template-columns:1fr;}}
  
  .side{display:flex;flex-direction:column;gap:10px;height:100%;}
  .logbox, .selbox, .selactbox{background:var(--panel);border:1px solid var(--line);display:flex;flex-direction:column;}
  .logbox{flex:1;min-height:0;}
  .selbox{height:150px;flex:none;}
  .selactbox{height:80px;flex:none;}
  .side h3{margin:0;padding:4px;font-size:14px;background:var(--bg);color:var(--txt);border-bottom:1px solid var(--line);text-align:center;}
  .log{padding:8px;white-space:pre-wrap;font-family:Consolas,monospace;font-size:12px;line-height:1.5;overflow:auto;flex:1;}
  .log .res{color:var(--hpgood);font-weight:700;}
  .log .atk{color:#ffb454;} .log .hp{color:#ff7b72;}
  .log .p0{color:blue;} .log .p1{color:red;}
  .selbox .sel{padding:8px;font-family:Consolas,monospace;font-size:12px;overflow:auto;}
  .selactbox .sel{padding:8px;font-family:Consolas,monospace;font-size:12px;}
  
  .board-area{display:flex;flex-direction:column;align-items:center;gap:10px;width:100%;height:100%;}
  .hand{display:flex;justify-content:center;gap:10px;min-height:103px;}
  
  .playmat{
    position:relative;
    width:100%;
    max-width:800px;
    flex:1;
    border:2px solid white;
    background:transparent;
    display:flex;
    justify-content:center;
    align-items:center;
    overflow:hidden;
  }
  
  .playmat::before{
    content:'';
    position:absolute;
    top:50%;left:50%;
    transform:translate(-50%,-50%);
    width:350px;height:350px;
    border:12px solid rgba(255,255,255,0.4);
    border-radius:50%;
    z-index:0;
  }
  .playmat::after{
    content:'';
    position:absolute;
    top:50%;left:0;
    transform:translateY(-50%);
    width:100%;height:12px;
    background:rgba(255,255,255,0.4);
    z-index:0;
  }
  .playmat-center-circle{
    position:absolute;
    top:50%;left:50%;
    transform:translate(-50%,-50%);
    width:100px;height:100px;
    background:rgba(255,255,255,0.4);
    border-radius:50%;
    z-index:1;
  }
  
  .zone{position:absolute;z-index:2;}
  
  .bench{display:flex;flex-direction:column;gap:-80px;} 
  .p1-bench{top:20px;right:20px;flex-direction:column-reverse;}
  .p0-bench{bottom:20px;left:20px;}
  
  .active-area{display:flex;flex-direction:column;align-items:center;gap:30px;z-index:2;}
  .active-card{display:flex;justify-content:center;min-height:200px;}
  
  .prizes{display:flex;gap:15px;justify-content:center;}
  .prize-dash{width:35px;height:4px;background:white;}
  .p1-prizes{position:absolute;top:28%;left:50%;transform:translate(-50%,-50%);}
  .p0-prizes{position:absolute;bottom:28%;left:50%;transform:translate(-50%,50%);}
  
  .res-p1{position:absolute;top:33%;left:50%;transform:translate(-50%,-50%);color:#d32f2f;font-size:20px;font-weight:bold;white-space:nowrap;z-index:3;}
  .res-p0{position:absolute;bottom:33%;left:50%;transform:translate(-50%,50%);color:#1976d2;font-size:20px;font-weight:bold;white-space:nowrap;z-index:3;}
  
  .stats{position:absolute;font-size:24px;font-weight:bold;}
  .stats div{margin-bottom:10px;}
  .p0-stats{left:20px;top:45%;transform:translateY(-50%);}
  .p1-stats{right:20px;bottom:45%;transform:translateY(50%);}
  .stats-turn{color:#ffeb3b;text-shadow:1px 1px 0 #000, -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000;}
  .stats-text{color:black;}
  
  .card{position:relative;width:74px;height:103px;border-radius:6px;background:var(--panel2);
        border:1px solid var(--line);flex:none;}
  .active-card .card{width:222px;height:309px;}
  .bench .card{margin-bottom:-60px;}
  .p1-bench .card{margin-top:-60px;margin-bottom:0;}
  
  .card img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:5px;display:none;}
  .card .txt{position:absolute;inset:0;padding:4px;display:flex;flex-direction:column;gap:2px;}
  .card .nm{font-size:10px;line-height:1.15;font-weight:600;max-height:36px;overflow:hidden;}
  .card.hasimg img{display:block;}
  .card.hasimg .txt{display:none;}
  
  #drop{position:fixed;inset:0;background:#000c;display:flex;align-items:center;justify-content:center;
        flex-direction:column;gap:14px;z-index:20;}
  #drop .box{border:2px dashed #fff;border-radius:14px;padding:40px 60px;text-align:center;color:white;}
  #drop.hide{display:none;}
  .hint{color:var(--dim);font-size:12px;}
  code{background:#0d1117;padding:1px 5px;border-radius:4px;}
  
  #preview{position:fixed;left:0;top:0;z-index:30;display:none;width:auto;pointer-events:none;
           background:#fff;border:1px solid #333;border-radius:10px;padding:6px;
           box-shadow:0 10px 30px #000b;color:#000;}
  #preview .pvrow{display:flex;gap:8px;align-items:flex-start;}
  #preview .pvmain{width:240px;flex:none;}
  #preview img{display:block;width:100%;border-radius:6px;background:var(--panel2);}
  #preview .cap{margin-top:6px;font-size:12px;line-height:1.4;}
  #preview .cap .sub{color:var(--dim);margin-top:2px;}
  #preview .pvenergy{margin-top:6px;display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;}
  #preview .egroup{display:flex;align-items:center;gap:6px;}
  #preview .estack{display:inline-flex;}
  #preview .estack img{width:72px;border-radius:5px;background:var(--panel2);box-shadow:0 1px 5px #000a;}
  #preview .estack img:not(:first-child){margin-left:-52px;}
  #preview .ecount{font-size:12px;font-weight:700;color:var(--dim);}
  #preview .pvtools{flex-direction:column;gap:8px;width:240px;flex:none;}
  
  .zoom{position:fixed;inset:0;z-index:25;background:#000b;display:flex;align-items:flex-start;justify-content:center;padding-top:5vh;}
  .zoom.hide{display:none;}
  .zoom-box{display:flex;flex-direction:column;width:min(92vw,900px);min-height:70vh;max-height:86vh;
            background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;}
  .zoom-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
  #zoom-title{font-size:14px;font-weight:700;}
  .zoom-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(74px,1fr));gap:8px;
             justify-items:center;align-content:flex-start;flex:1;min-height:0;overflow:auto;}
"""

new_html_structure = """<div id="wrap" style="display:none">
  <div class="side">
    <div class="logbox">
      <h3 data-i18n="logTitle">Log</h3>
      <div class="log" id="log"></div>
    </div>
    <div class="selbox">
      <h3 data-i18n="selectedTitle">Select</h3>
      <div class="sel" id="selected">-</div>
    </div>
    <div class="selactbox">
      <h3>Selected Action</h3>
      <div class="sel" id="selected-action">-</div>
    </div>
  </div>
  
  <div class="board-area">
    <!-- P1 Hand (Top) -->
    <div class="hand p1-hand" id="hand1"></div>
    
    <!-- Playmat -->
    <div class="playmat">
      <div class="playmat-center-circle"></div>
      
      <!-- P1 Prizes -->
      <div class="prizes p1-prizes" id="prize1"></div>
      
      <!-- P1 Bench -->
      <div class="bench p1-bench zone" id="bench1"></div>
      
      <!-- P1 Stats -->
      <div class="stats p1-stats" id="stats1"></div>
      
      <!-- Center Actives -->
      <div class="active-area">
        <div class="active-card" id="active1"></div>
        <div class="active-card" id="active0"></div>
      </div>
      
      <!-- Result Text -->
      <div class="res-p1" id="res1"></div>
      <div class="res-p0" id="res0"></div>
      
      <!-- P0 Stats -->
      <div class="stats p0-stats" id="stats0"></div>
      
      <!-- P0 Bench -->
      <div class="bench p0-bench zone" id="bench0"></div>
      
      <!-- P0 Prizes -->
      <div class="prizes p0-prizes" id="prize0"></div>
    </div>
    
    <!-- P0 Hand (Bottom) -->
    <div class="hand p0-hand" id="hand0"></div>
  </div>
</div>"""

# Replace CSS
css_pattern = r'<style>.*?</style>'
html = re.sub(css_pattern, f'<style>\n{new_css}\n</style>', html, flags=re.DOTALL)

# Replace HTML wrap
html_pattern = r'<div id="wrap" style="display:none">.*?</div>\n</div>'
html = re.sub(html_pattern, new_html_structure, html, flags=re.DOTALL)

# Replace render function
render_script = """function renderInner(){
  const s=SNAP[idx], cur=s.current;
  document.getElementById('slider').value=idx;
  document.getElementById('stepno').textContent=`${idx+1}/${SNAP.length}`;
  document.getElementById('turn').textContent=cur.turn;
  document.getElementById('active').textContent='P'+cur.yourIndex;
  document.getElementById('ctx').textContent=(s.select&&s.select.context)||'-';
  
  const meta=SNAP[0].metadata||{};
  const p1_name = meta.p1_name || 'Player 1';
  const p0_name = meta.p0_name || 'Player 0';
  
  // Render Hands
  document.getElementById('hand1').innerHTML = cur.players[1].hand ? cur.players[1].hand.map(c=>plainCard(c)).join('') : '';
  document.getElementById('hand0').innerHTML = cur.players[0].hand ? cur.players[0].hand.map(c=>plainCard(c)).join('') : '';
  
  // Render Benches
  document.getElementById('bench1').innerHTML = cur.players[1].bench ? cur.players[1].bench.map(c=>pokemonCard(c)).join('') : '';
  document.getElementById('bench0').innerHTML = cur.players[0].bench ? cur.players[0].bench.map(c=>pokemonCard(c)).join('') : '';
  
  // Render Actives
  document.getElementById('active1').innerHTML = (cur.players[1].active&&cur.players[1].active.length) ? pokemonCard(cur.players[1].active[0], {active:true}) : '';
  document.getElementById('active0').innerHTML = (cur.players[0].active&&cur.players[0].active.length) ? pokemonCard(cur.players[0].active[0], {active:true}) : '';
  
  // Render Prizes (dashes)
  const p1PrizeCount = cur.players[1].prize ? cur.players[1].prize.length : 0;
  document.getElementById('prize1').innerHTML = Array(p1PrizeCount).fill('<div class="prize-dash"></div>').join('');
  const p0PrizeCount = cur.players[0].prize ? cur.players[0].prize.length : 0;
  document.getElementById('prize0').innerHTML = Array(p0PrizeCount).fill('<div class="prize-dash"></div>').join('');
  
  // Render Stats
  document.getElementById('stats1').innerHTML = `
    <div class="stats-turn">${cur.yourIndex===1 ? 'First' : 'Second'}</div>
    <div class="stats-text">Time ${Math.floor(Math.random()*600)}</div>
    <div class="stats-text">Discard ${cur.players[1].discard ? cur.players[1].discard.length : 0}</div>
    <div class="stats-text">Deck ${cur.players[1].deckCount}</div>
  `;
  document.getElementById('stats0').innerHTML = `
    <div class="stats-turn">${cur.yourIndex===0 ? 'First' : 'Second'}</div>
    <div class="stats-text">Time ${Math.floor(Math.random()*600)}</div>
    <div class="stats-text">Discard ${cur.players[0].discard ? cur.players[0].discard.length : 0}</div>
    <div class="stats-text">Deck ${cur.players[0].deckCount}</div>
  `;
  
  // Results
  if(cur.result >= 0) {
      if(cur.result === 0) {
         document.getElementById('res0').textContent = `[Win] ${p0_name} (+1)`;
         document.getElementById('res1').textContent = `[Loss] ${p1_name} (-3)`;
      } else if (cur.result === 1) {
         document.getElementById('res1').textContent = `[Win] ${p1_name} (+1)`;
         document.getElementById('res0').textContent = `[Loss] ${p0_name} (-3)`;
      } else {
         document.getElementById('res0').textContent = `[Draw]`;
         document.getElementById('res1').textContent = `[Draw]`;
      }
  } else {
      document.getElementById('res0').textContent = '';
      document.getElementById('res1').textContent = '';
  }

  document.getElementById('selected').textContent=describeSelected(s);
  document.getElementById('selected-action').textContent= s.select ? (s.select.option && s.select.option[0] && s.select.option[0].name ? s.select.option[0].name : describeSelected(s)) : '-';
  document.getElementById('log').innerHTML=(s.logs||[]).map(decode).filter(x=>x!=null).join('\\n');

  document.getElementById('prev').disabled=idx===0;
  document.getElementById('next').disabled=idx===SNAP.length-1;
  if(zoomPi!==null) renderZoom();   // keep an open trash grid in sync while navigating
}

function render(){
  if(document.startViewTransition) {
    document.startViewTransition(() => { renderInner(); });
  } else {
    renderInner();
  }
}
"""
html = re.sub(r'function renderInner\(\)\{.*?} // ---- language toggle', render_script + '\n\n// ---- language toggle', html, flags=re.DOTALL)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)

