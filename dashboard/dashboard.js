let deckNames = {};
let activeDecks = [];
let watchedModels = new Set();

async function loadWatched() {
    try {
        const res = await fetch('/api/watched');
        if (res.ok) {
            const data = await res.json();
            watchedModels = new Set(data.watched || []);
        }
    } catch(e) { console.warn('Could not load watched models'); }
}

async function toggleWatch(modelName) {
    if (watchedModels.has(modelName)) {
        watchedModels.delete(modelName);
    } else {
        watchedModels.add(modelName);
    }
    try {
        await fetch('/api/watched', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ watched: Array.from(watchedModels) })
        });
    } catch(e) { console.warn('Could not save watched models'); }
    // Re-render only the eye buttons without full refresh
    document.querySelectorAll('.watch-btn').forEach(btn => {
        const m = btn.dataset.model;
        const active = watchedModels.has(m);
        btn.textContent = active ? '\ud83d\udc41' : '\ud83d\udc41\ufe0e';
        btn.title = active ? 'Beobachtung aktiv – Replays werden generiert' : 'Klicken zum Beobachten';
        btn.style.opacity = active ? '1' : '0.25';
        btn.style.filter = active ? 'none' : 'grayscale(1)';
        btn.style.transform = active ? 'scale(1.2)' : 'scale(1)';
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadWatched();
        try {
            const nameRes = await fetch('/decks/deck_names.json');
            if (nameRes.ok) {
                deckNames = await nameRes.json();
            }
        } catch(e) { console.warn('Could not load deck names'); }
        
        try {
            const activeRes = await fetch('/decks/active_decks.json');
            if (activeRes.ok) {
                activeDecks = await activeRes.json();
            }
        } catch(e) { console.warn('Could not load active decks'); }
        
        // Render Roster immediately
        renderRoster(deckNames);

        loadReplays();
        const replayRefresh = document.getElementById('replay-refresh');
        if (replayRefresh) {
            replayRefresh.addEventListener('click', loadReplays);
        }

        const response = await fetch('/decks/pairwise_winrates.json');
        if (!response.ok) throw new Error('Data not found');
        const data = await response.json();
        
        let currentData = {};
        try {
            const curRes = await fetch('/decks/current_generation_winrates.json');
            if (curRes.ok) {
                currentData = await curRes.json();
            }
        } catch(e) {}
        
        renderDashboard(data, currentData);
    } catch (e) {
        console.error(e);
        document.getElementById('leaderboard-body').innerHTML = `
            <tr><td colspan="4" style="text-align:center; padding: 2rem; color:#f43f5e;">No tournament data found yet.<br>Please run <code>python src/auto_arena.py</code> first!</td></tr>
        `;
    }

    // Custom Tooltip Logic
    const tooltip = document.getElementById('custom-tooltip');
    const heatmapContainer = document.getElementById('heatmap-container');
    if (heatmapContainer && tooltip) {
        heatmapContainer.addEventListener('mouseover', (e) => {
            const cell = e.target.closest('.heatmap-cell');
            if (cell && cell.dataset.tooltip) {
                tooltip.textContent = cell.dataset.tooltip;
                tooltip.classList.add('visible');
            }
        });
        heatmapContainer.addEventListener('mousemove', (e) => {
            if (tooltip.classList.contains('visible')) {
                // Keep tooltip inside viewport if possible
                let left = e.pageX + 15;
                let top = e.pageY + 15;
                if (left + tooltip.offsetWidth > window.innerWidth) {
                    left = e.pageX - tooltip.offsetWidth - 15;
                }
                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            }
        });
        heatmapContainer.addEventListener('mouseout', (e) => {
            tooltip.classList.remove('visible');
        });
    }
});

function renderDashboard(data, currentData, eloRatings) {
    const decks = new Set();
    const stats = {};
    
    // Parse the pairwise data
    for (const deckA in data) {
        decks.add(deckA);
        if (!stats[deckA]) stats[deckA] = { wins: 0, total: 0, pw: 0, dw: 0, bw: 0, pl: 0, dl: 0, bl: 0 };
        
        for (const deckB in data[deckA]) {
            decks.add(deckB);
            const matchData = data[deckA][deckB];
            const wins = matchData.wins || 0;
            const matches = matchData.matches || 0;
            
            // Only add half the matches to total since it's mirrored data 
            // (we don't want to double count when iterating over both A->B and B->A)
            // Wait, auto_tourney.py already mirrors data, but if we iterate all keys A and B, we would double count.
            // Let's just track globally:
        }
    }
    
    // Recalculate accurately without double counting
    const processedPairs = new Set();
    for (const deckA in data) {
        for (const deckB in data[deckA]) {
            const pairId = [deckA, deckB].sort().join('-');
            if (!processedPairs.has(pairId)) {
                processedPairs.add(pairId);
                const winsA = data[deckA][deckB][0] || 0;
                const matches = data[deckA][deckB][1] || 0;
                const winsB = matches - winsA;
                
                if (!stats[deckA]) stats[deckA] = { wins: 0, total: 0, pw: 0, dw: 0, bw: 0, pl: 0, dl: 0, bl: 0 };
                if (!stats[deckB]) stats[deckB] = { wins: 0, total: 0, pw: 0, dw: 0, bw: 0, pl: 0, dl: 0, bl: 0 };
                
                stats[deckA].wins += winsA;
                stats[deckA].total += matches;
                stats[deckB].wins += winsB;
                stats[deckB].total += matches;
            }
        }
    }

    if (currentData) {
        const currProcessedPairs = new Set();
        for (const deckA in currentData) {
            for (const deckB in currentData[deckA]) {
                const pairId = [deckA, deckB].sort().join('-');
                if (!currProcessedPairs.has(pairId)) {
                    currProcessedPairs.add(pairId);
                    const cd = currentData[deckA][deckB];
                    if (!stats[deckA]) stats[deckA] = { wins: 0, total: 0, pw: 0, dw: 0, bw: 0, pl: 0, dl: 0, bl: 0 };
                    if (!stats[deckB]) stats[deckB] = { wins: 0, total: 0, pw: 0, dw: 0, bw: 0, pl: 0, dl: 0, bl: 0 };
                    
                    if (stats[deckA].pl === undefined) {
                        stats[deckA].pl = 0; stats[deckA].dl = 0; stats[deckA].bl = 0;
                        stats[deckB].pl = 0; stats[deckB].dl = 0; stats[deckB].bl = 0;
                    }
                    
                    stats[deckA].pw += cd[2] || 0;
                    stats[deckA].dw += cd[3] || 0;
                    stats[deckA].bw += cd[7] || 0;
                    stats[deckA].pl += cd[5] || 0;
                    stats[deckA].dl += cd[6] || 0;
                    stats[deckA].bl += cd[8] || 0;
                    
                    stats[deckB].pw += cd[5] || 0;
                    stats[deckB].dw += cd[6] || 0;
                    stats[deckB].bw += cd[8] || 0;
                    stats[deckB].pl += cd[2] || 0;
                    stats[deckB].dl += cd[3] || 0;
                    stats[deckB].bl += cd[7] || 0;
                }
            }
        }
    }

    const deckList = Array.from(decks).sort((a,b) => {
        // Natural sort for deck names
        const numA = parseInt(a.replace(/\\D/g, '')) || 0;
        const numB = parseInt(b.replace(/\\D/g, '')) || 0;
        if (numA !== numB) return numA - numB;
        return a.localeCompare(b);
    });
    
    // 1. Render Leaderboard
    const leaderboard = deckList.map(deck => {
        const deckStats = stats[deck];
        const winrate = deckStats.total > 0 ? (deckStats.wins / deckStats.total) * 100 : 0;
        const elo = eloRatings && eloRatings[deck] ? eloRatings[deck] : 1200.0;
        return { name: deck, ...deckStats, winrate, elo };
    }).filter(deck => {
        const isEliminated = activeDecks.length > 0 && !activeDecks.includes(deck.name);
        return !isEliminated; // Hide them completely
    }).sort((a, b) => b.elo - a.elo);

    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = leaderboard.map((deck, index) => {
        const alias = formatName(deck.name);
        const opacity = '1';
        const elimBadge = '';
        
        const totalWins = deck.pw + deck.bw + deck.dw;
        const pwPct = totalWins > 0 ? (deck.pw / totalWins) * 100 : 33.3;
        const bwPct = totalWins > 0 ? (deck.bw / totalWins) * 100 : 33.3;
        const dwPct = totalWins > 0 ? (deck.dw / totalWins) * 100 : 33.3;
        
        return `
        <tr style="opacity: ${opacity};">
            <td>#${index + 1}</td>
            <td contenteditable="true" class="editable-name" data-model="${deck.name}" style="border: 1px dashed rgba(255,255,255,0.2); border-radius: 4px; padding: 4px 8px; cursor: text; outline: none; transition: border-color 0.2s;" onblur="saveAlias(this)" onfocus="window.isEditing = true; this.style.borderColor='var(--accent-1)'"><strong>${alias}</strong>${elimBadge}</td>
            <td style="font-family: monospace; font-size: 0.85em; opacity: 0.7;">${deck.name}</td>
            <td style="font-weight: bold; color: #f59e0b;">${Math.round(deck.elo)}</td>
            <td>
                ${deck.winrate.toFixed(1)}%
                <div class="winrate-bar"><div class="winrate-fill" style="width: ${deck.winrate}%"></div></div>
            </td>
            <td style="color:var(--text-muted)">${deck.total} matches</td>
            <td style="min-width: 200px; vertical-align: middle;">
                <div style="display: flex; gap: 1rem; justify-content: space-between; font-size: 0.85em; font-weight: 600; padding: 0 4px;">
                    <span title="Prize Cards 🏆" style="color: #10b981;">${deck.pw} 🏆</span>
                    <span title="Bench-Out 🪑" style="color: #38bdf8;">${deck.bw} 🪑</span>
                    <span title="Deck-Out 🚫" style="color: #f43f5e;">${deck.dw} 🚫</span>
                </div>
                <div style="margin-top: 6px; height: 6px; width: 100%; background: rgba(255,255,255,0.05); border-radius: 3px; display: flex; overflow: hidden;">
                    <div style="width: ${pwPct}%; background: #10b981;"></div>
                    <div style="width: ${bwPct}%; background: #38bdf8;"></div>
                    <div style="width: ${dwPct}%; background: #f43f5e;"></div>
                </div>
            </td>
            <td style="text-align:center; vertical-align:middle;">
                <button class="watch-btn" data-model="${deck.name}"
                    onclick="toggleWatch('${deck.name}')"
                    title="${watchedModels.has(deck.name) ? 'Beobachtung aktiv – Replays werden generiert' : 'Klicken zum Beobachten'}"
                    style="background:none; border:none; cursor:pointer; font-size:1.3em; transition: all 0.2s;
                           opacity:${watchedModels.has(deck.name) ? '1' : '0.25'};
                           filter:${watchedModels.has(deck.name) ? 'none' : 'grayscale(1)'};
                           transform:${watchedModels.has(deck.name) ? 'scale(1.2)' : 'scale(1)'};"
                >👁</button>
            </td>
        </tr>
        `;
    }).join('');



    // 2. Render Heatmap
    const container = document.getElementById('heatmap-container');
    let heatmapDecks = [];
    if (activeDecks.length > 0) {
        heatmapDecks = [...activeDecks].sort((a,b)=>{
            const numA = parseInt(a.replace(/\D/g, '')) || 0;
            const numB = parseInt(b.replace(/\D/g, '')) || 0;
            if (numA !== numB) return numA - numB;
            return a.localeCompare(b);
        });
    } else if (currentData && Object.keys(currentData).length > 0) {
        heatmapDecks = Object.keys(currentData).sort((a,b)=>{
            const numA = parseInt(a.replace(/\D/g, '')) || 0;
            const numB = parseInt(b.replace(/\D/g, '')) || 0;
            if (numA !== numB) return numA - numB;
            return a.localeCompare(b);
        });
    } else {
        heatmapDecks = deckList;
    }
    const size = heatmapDecks.length + 1;
    container.style.gridTemplateColumns = `repeat(${size}, minmax(60px, 1fr))`;
    
    // Header row
    let html = `<div class="heatmap-cell heatmap-header"></div>`;
    heatmapDecks.forEach(d => {
        html += `<div class="heatmap-cell heatmap-header" data-tooltip="${d}">${formatName(d)}</div>`;
    });
        
    heatmapDecks.forEach(deckA => {
        // Row header
        html += `<div class="heatmap-cell heatmap-header" data-tooltip="${deckA}" style="justify-content:flex-start">${formatName(deckA)}</div>`;
        
        heatmapDecks.forEach(deckB => {
            const idxA = heatmapDecks.indexOf(deckA);
            const idxB = heatmapDecks.indexOf(deckB);
            
            if (deckA === deckB) {
                html += `<div class="heatmap-cell" style="background: rgba(255,255,255,0.02);">-</div>`;
            } else {
                let wins = 0, matches = 0;
                let pw_a = 0, dw_a = 0, bw_a = 0, pw_b = 0, dw_b = 0, bw_b = 0;
                if (currentData && currentData[deckA] && currentData[deckA][deckB]) {
                    const data = currentData[deckA][deckB];
                    wins = data[0] || 0;
                    matches = data[1] || 0;
                    pw_a = data[2] || 0;
                    dw_a = data[3] || 0;
                    // wins_b is data[4], pw_b is data[5], dw_b is data[6]
                    pw_b = data[5] || 0;
                    dw_b = data[6] || 0;
                    bw_a = data[7] || 0;
                    bw_b = data[8] || 0;
                }
                
                if (matches === 0) {
                    html += `<div class="heatmap-cell" style="background: rgba(255,255,255,0.05); color: #666;">N/A</div>`;
                } else {
                    const winrate = wins / matches;
                    const losses = matches - wins;
                    
                    // Color gradient from Red (loss) to Green (win)
                    const r = Math.floor(255 * (1 - winrate));
                    const g = Math.floor(255 * winrate);
                    const color = `rgba(${r}, ${g}, 0, 0.3)`;
                    
                    // Tooltip with detailed info
                    const title = `${formatName(deckA)} vs ${formatName(deckB)}\n\nSiege: ${wins} (Prize: ${pw_a}, Deckout: ${dw_a}, Bench: ${bw_a})\nNiederlagen: ${losses} (Prize: ${pw_b}, Deckout: ${dw_b}, Bench: ${bw_b})`;
                    
                    html += `<div class="heatmap-cell" style="background: ${color};" data-tooltip="${title}">
                        <div style="font-weight: bold; font-size: 1.1em;">${wins}:${losses}</div>
                        <div style="font-size: 0.7em; opacity: 0.8; margin-top: 2px;">(P:${pw_a} D:${dw_a} B:${bw_a})</div>
                    </div>`;
                }
            }
        });
    });
    
    container.innerHTML = html;
}

window.saveAlias = function(el) {
    const model = el.getAttribute('data-model');
    // Remove the elimination badge HTML if present before saving
    let clone = el.cloneNode(true);
    const badge = clone.querySelector('.elim-badge');
    if (badge) badge.remove();
    const text = clone.innerText.trim();
    
    const customAliases = JSON.parse(localStorage.getItem('modelAliases')) || {};
    customAliases[model] = text;
    localStorage.setItem('modelAliases', JSON.stringify(customAliases));
    el.style.borderColor = 'rgba(255,255,255,0.2)';
    
    // Slight visual feedback
    el.style.background = 'rgba(16, 185, 129, 0.1)';
    setTimeout(() => { el.style.background = 'transparent'; }, 500);
    window.isEditing = false;
}

function formatName(name) {
    const customAliases = JSON.parse(localStorage.getItem('modelAliases')) || {};
    if (customAliases[name]) return customAliases[name];

    const matchBank = name.match(/deck_(bank_\d+)/);
    const matchRegular = name.match(/deck_(\d+)/);
    
    let id = '';
    if (matchBank) {
        id = matchBank[1];
    } else if (matchRegular) {
        id = matchRegular[1];
    }
    
    if (id && deckNames[id]) {
        return `Deck ${id.replace('bank_', 'Bank ')} (${deckNames[id]})`;
    }
    return name;
}

function renderRoster(names) {
    const grid = document.getElementById('roster-grid');
    if (!grid) return;
    
    // Sort keys numerically, filtering by active status if available
    let sortedIds = Object.keys(names).sort((a, b) => parseInt(a) - parseInt(b));
    if (activeDecks.length > 0) {
        sortedIds = sortedIds.filter(id => activeDecks.includes(id));
    }
    
    grid.innerHTML = sortedIds.map(id => `
        <div class="roster-item">
            <span class="roster-id">D${id}</span>
            <span class="roster-name">${names[id]}</span>
        </div>
    `).join('');
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[ch]));
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return '';
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
}

function formatReplayDate(seconds) {
    if (!seconds) return '';
    return new Date(seconds * 1000).toLocaleString();
}

function replayTitle(replay) {
    const meta = replay.metadata || {};
    if (meta.p0_name || meta.p1_name) {
        return `${meta.p0_name || 'Player 0'} vs ${meta.p1_name || 'Player 1'}`;
    }
    return replay.path.split('/').pop();
}

function renderReplays(replays) {
    const list = document.getElementById('replay-list');
    const status = document.getElementById('replay-status');
    if (!list || !status) return;

    if (!replays.length) {
        status.textContent = 'No replay JSON files found yet.';
        list.innerHTML = '';
        return;
    }

    status.textContent = `${replays.length} replay${replays.length === 1 ? '' : 's'} available`;
    list.innerHTML = replays.map(replay => {
        const snapshots = replay.snapshots == null ? 'unknown steps' : `${replay.snapshots} steps`;
        const modified = formatReplayDate(replay.mtime);
        const size = formatBytes(replay.size);
        return `
            <article class="replay-item">
                <div class="replay-item-title">${escapeHtml(replayTitle(replay))}</div>
                <div class="replay-meta">
                    <span class="replay-pill">${escapeHtml(replay.group)}</span>
                    <span class="replay-pill">${escapeHtml(snapshots)}</span>
                    ${size ? `<span class="replay-pill">${escapeHtml(size)}</span>` : ''}
                </div>
                <div class="replay-path">${escapeHtml(replay.path)}${modified ? `<br>${escapeHtml(modified)}` : ''}</div>
                <div class="replay-links">
                    <button class="btn btn-primary" onclick="launchHeroz('${escapeHtml(replay.url)}')" type="button">HERoz Viz</button>
                    <a class="btn btn-secondary" href="${escapeHtml(replay.url)}" target="_blank" rel="noopener">JSON</a>
                </div>
            </article>
        `;
    }).join('');
}

async function loadReplays() {
    const status = document.getElementById('replay-status');
    if (status) status.textContent = 'Loading replays...';

    try {
        const res = await fetch('/api/replays', { cache: 'no-store' });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const data = await res.json();
        renderReplays(data.replays || []);
    } catch (err) {
        console.error('Could not load replays:', err);
        if (status) status.textContent = 'Replay list needs the Flask dashboard server on port 8050.';
    }
}
window.launchHeroz = async function(url) {
    try {
        const res = await fetch(url);
        const text = await res.text();
        const obj = JSON.parse(text);

        const input = document.createElement("input");
        input.type = "hidden";
        input.name = "json";
        
        if ("steps" in obj) {
            input.value = JSON.stringify(obj["steps"][0][0]["visualize"]);
        } else {
            input.value = text;
        }

        const form = document.createElement("form");
        form.method = "POST";
        form.action = "https://ptcgvis.heroz.jp/Visualizer/Replay/0";
        form.target = "_blank";
        form.appendChild(input);

        document.body.appendChild(form);
        form.submit();
        
        setTimeout(() => document.body.removeChild(form), 1000);
    } catch (err) {
        console.error("Failed to launch HERoz visualizer:", err);
        alert("Fehler beim Starten des Visualizers.");
    }
};

// Auto-refresh logic
window.isEditing = false;

async function pollData() {
    if (window.isEditing) return; // Prevent overwriting DOM while user is typing
    
    try {
        const nameRes = await fetch('/decks/deck_names.json');
        if (nameRes.ok) deckNames = await nameRes.json();
        
        const activeRes = await fetch('/decks/active_decks.json');
        if (activeRes.ok) activeDecks = await activeRes.json();
        
        const response = await fetch('/decks/pairwise_winrates.json');
        if (response.ok) {
            const data = await response.json();
            
            let curData = {};
            const curRes = await fetch('/decks/current_generation_winrates.json');
            if (curRes.ok) curData = await curRes.json();
            
            let eloRatings = {};
            const eloRes = await fetch('/decks/elo_ratings.json');
            if (eloRes.ok) eloRatings = await eloRes.json();
            
            renderDashboard(data, curData, eloRatings);
            renderRoster(deckNames);
        }
    } catch (err) {
        console.error('Error polling data:', err);
    }
}

// Control Panel Logic
const btnStart = document.getElementById('btn-start');
const btnPause = document.getElementById('btn-pause');
const btnReset = document.getElementById('btn-reset');

function checkServerStatus() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            const isRunning = data.running;
            btnStart.disabled = isRunning;
            btnPause.disabled = !isRunning;
            if (isRunning) {
                const ind = document.querySelector('.live-indicator');
                if (ind) {
                    ind.style.background = '#00ff88';
                    ind.style.boxShadow = '0 0 10px #00ff88';
                }
            } else {
                const ind = document.querySelector('.live-indicator');
                if (ind) {
                    ind.style.background = '#888';
                    ind.style.boxShadow = 'none';
                }
            }
        })
        .catch(err => {
            console.error('Server offline?', err);
            btnStart.disabled = true;
            btnPause.disabled = true;
            const ind = document.querySelector('.live-indicator');
            if (ind) ind.style.background = '#ff3c3c';
        });
}

if (btnStart) {
    btnStart.addEventListener('click', () => {
        btnStart.disabled = true;
        fetch('/api/start', { method: 'POST' })
            .then(() => checkServerStatus());
    });
}

if (btnPause) {
    btnPause.addEventListener('click', () => {
        btnPause.disabled = true;
        fetch('/api/pause', { method: 'POST' })
            .then(() => checkServerStatus());
    });
}

if (btnReset) {
    btnReset.addEventListener('click', () => {
        if (confirm("🚨 BIST DU SICHER?\n\nDas löscht ALLE Modelle, Win-Rates, Ghost-Pool-Decks und fängt das Turnier komplett von Generation 1 mit Deck 1 bis 5 von vorne an!")) {
            btnReset.disabled = true;
            fetch('/api/reset', { method: 'POST' })
                .then(() => {
                    btnReset.disabled = false;
                    checkServerStatus();
                });
        }
    });
}

// Update status more frequently to reflect button states
setInterval(checkServerStatus, 2000);
checkServerStatus();

// Initial fetch
pollData();
setInterval(pollData, 5000);
