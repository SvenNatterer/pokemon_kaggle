import re

with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. CSS Updates
html = html.replace('.active-card .card{width:222px;height:309px;}', '.active-card .card{width:133px;height:185px;}')

# Pokeball Playmat Center
html = html.replace('border:12px solid rgba(255,255,255,0.4);', 'border:4px solid white;')
html = html.replace('height:12px;\n    background:rgba(255,255,255,0.4);', 'height:4px;\n    background:white;')
html = html.replace('background:rgba(255,255,255,0.4);', 'border:4px solid white; background:transparent;')

new_css = """
  .card{position:relative;width:74px;height:103px;border-radius:6px;background:var(--panel2);
        border:1px solid var(--line);flex:none;}
  .active-card .card{width:133px;height:185px;}
  
  .dmg-text {
    position:absolute; top:-8px; left:-5px;
    color:red; font-size:20px; font-weight:900;
    text-shadow: 2px 2px 0 #000, -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 0 3px 2px rgba(0,0,0,0.5);
    z-index:10; font-family: "Impact", sans-serif;
  }
  .active-card .dmg-text { font-size:36px; top:-15px; left:-10px; }
  
  .energy-icons {
    position:absolute; top:-10px; right:-15px;
    display:flex; flex-wrap:wrap; gap:2px; max-width:40px;
    z-index:10;
  }
  .energy-icon {
    width:22px; height:22px; border-radius:50%; border:2px solid white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    background-size: cover;
    background-color: #eee;
  }
  .active-card .energy-icons { right: -25px; max-width: 60px; top:-15px; }
  .active-card .energy-icon { width: 32px; height: 32px; }
"""
html = re.sub(r'\.card\{position:relative;width:74px;.*?\.active-card \.card\{width:222px;height:309px;\}', new_css, html, flags=re.DOTALL)

# JS Updates
js_old = """function pokemonCard(p, opts={}){
  if(!p) return `<div class="card ${opts.active?'active':''} faceup"><div class="txt"><div class="nm">${T('none')}</div></div></div>`;
  const cls='card mon'+(opts.active?' active':'')+' faceup';   // 'mon' = battle/bench pokemon (gets the HP shade)
  const url=cardImgUrl(p.id);
  const badges=[];
  for(const s of (opts.status||[])) badges.push(`<span class="badge sc">${s}</span>`);
  for(const t of (p.tools||[])) badges.push(`<span class="badge tool" title="${nm(t.id)}">${T('tool')}</span>`);
  const img = url?`<img src="${url}" onload="this.parentElement.classList.add('hasimg')" onerror="this.remove()">`:'';
  const vt = p.serial ? ` style="view-transition-name: card_${p.serial};"` : '';
  return `<div class="${cls}"${vt} title="${nm(p.id)} HP${p.hp}/${p.maxHp}" data-id="${p.id}" data-hp="${p.hp}" data-mhp="${p.maxHp}" data-encards="${(p.energyCards||[]).map(e=>e.id).join(',')}" data-tools="${(p.tools||[]).map(t=>t.id).join(',')}">
    ${img}
    <div class="badges">${badges.join('')}</div>
    <div class="txt">
      <div class="nm">${p.name||nm(p.id)}</div>
      <div class="stat">
        ${energyDots(p.energies)}
        <div class="hp">HP ${p.hp}/${p.maxHp}</div>
        ${hpBar(p.hp,p.maxHp)}
      </div>
    </div>
  </div>`;
}"""

js_new = """function pokemonCard(p, opts={}){
  if(!p) return `<div class="card ${opts.active?'active':''} faceup"><div class="txt"><div class="nm">${T('none')}</div></div></div>`;
  const cls='card mon'+(opts.active?' active':'')+' faceup';
  const url=cardImgUrl(p.id);
  const badges=[];
  for(const s of (opts.status||[])) badges.push(`<span class="badge sc">${s}</span>`);
  for(const t of (p.tools||[])) badges.push(`<span class="badge tool" title="${nm(t.id)}">${T('tool')}</span>`);
  
  let dmgHtml = '';
  const damage = p.maxHp - p.hp;
  if (damage > 0) {
      dmgHtml = `<div class="dmg-text">${damage}</div>`;
  }
  
  let energyHtml = '';
  if (p.energies && p.energies.length > 0) {
      const eIcons = p.energies.map(t => {
          let color = '#ccc';
          if(t===1) color='#78C850'; // Grass
          if(t===2) color='#F08030'; // Fire
          if(t===3) color='#6890F0'; // Water
          if(t===4) color='#F8D030'; // Lightning
          if(t===5) color='#F85888'; // Psychic
          if(t===6) color='#C03028'; // Fighting
          if(t===7) color='#705848'; // Darkness
          if(t===8) color='#B8B8D0'; // Metal
          if(t===9) color='#7038F8'; // Dragon
          if(t===0) color='#A8A8A8'; // Colorless
          // We can use a solid color for the energy icon
          return `<div class="energy-icon" style="background:${color}" title="Energy ${t}"></div>`;
      }).join('');
      energyHtml = `<div class="energy-icons">${eIcons}</div>`;
  }

  const img = url?`<img src="${url}" onload="this.parentElement.classList.add('hasimg')" onerror="this.remove()">`:'';
  const vt = p.serial ? ` style="view-transition-name: card_${p.serial};"` : '';
  
  return `<div class="${cls}"${vt} title="${nm(p.id)} HP${p.hp}/${p.maxHp}" data-id="${p.id}" data-hp="${p.hp}" data-mhp="${p.maxHp}" data-encards="${(p.energyCards||[]).map(e=>e.id).join(',')}" data-tools="${(p.tools||[]).map(t=>t.id).join(',')}">
    ${img}
    ${dmgHtml}
    ${energyHtml}
    <div class="badges">${badges.join('')}</div>
    <div class="txt">
      <div class="nm">${p.name||nm(p.id)}</div>
      <div class="stat">
        ${energyDots(p.energies)}
        <div class="hp">HP ${p.hp}/${p.maxHp}</div>
        ${hpBar(p.hp,p.maxHp)}
      </div>
    </div>
  </div>`;
}"""

html = html.replace(js_old, js_new)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Update completed.")
