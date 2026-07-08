import re

with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

render_script = """// ---- main render ----
function renderInner(){
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

pattern = r'// ---- main render ----.*?// ---- language toggle \+ initial apply ----'
new_html = re.sub(pattern, render_script + '\n\n// ---- language toggle + initial apply ----', html, flags=re.DOTALL)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(new_html)

print("Done")
