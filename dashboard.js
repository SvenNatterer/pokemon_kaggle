let deckNames = {};
let activeDecks = [];

document.addEventListener('DOMContentLoaded', async () => {
    try {
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

        // Start live status polling immediately
        setInterval(fetchStatus, 2000);
        fetchStatus();

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
            <tr><td colspan="4" style="text-align:center; padding: 2rem; color:#f43f5e;">No tournament data found yet.<br>Please run <code>python src/auto_tourney.py</code> first!</td></tr>
        `;
    }
});

function renderDashboard(data, currentData) {
    const decks = new Set();
    const stats = {};
    
    // Parse the pairwise data
    for (const deckA in data) {
        decks.add(deckA);
        if (!stats[deckA]) stats[deckA] = { wins: 0, total: 0 };
        
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
                
                if (!stats[deckA]) stats[deckA] = { wins: 0, total: 0 };
                if (!stats[deckB]) stats[deckB] = { wins: 0, total: 0 };
                
                stats[deckA].wins += winsA;
                stats[deckA].total += matches;
                stats[deckB].wins += winsB;
                stats[deckB].total += matches;
            }
        }
    }

    const deckList = Array.from(decks).sort((a,b) => {
        // Natural sort for deck names
        const numA = parseInt(a.replace(/\D/g, '')) || 0;
        const numB = parseInt(b.replace(/\D/g, '')) || 0;
        return numA - numB;
    });
    
    // 1. Render Leaderboard
    const leaderboard = deckList.map(deck => {
        const deckStats = stats[deck];
        const winrate = deckStats.total > 0 ? (deckStats.wins / deckStats.total) * 100 : 0;
        return { name: deck, ...deckStats, winrate };
    }).sort((a, b) => b.winrate - a.winrate);

    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = leaderboard.map((deck, index) => {
        const id = deck.name.replace(/\D/g, '');
        const isEliminated = activeDecks.length > 0 && !activeDecks.includes(id);
        const elimBadge = isEliminated ? ` <span style="font-size: 0.75rem; color: #f43f5e; background: rgba(244,63,94,0.1); padding: 2px 6px; border-radius: 4px; margin-left: 8px; font-weight: normal;">Eliminiert 👻</span>` : '';
        const opacity = isEliminated ? '0.5' : '1';
        
        return `
        <tr style="opacity: ${opacity};">
            <td>#${index + 1}</td>
            <td><strong>${formatName(deck.name)}</strong>${elimBadge}</td>
            <td>
                ${deck.winrate.toFixed(1)}%
                <div class="winrate-bar"><div class="winrate-fill" style="width: ${deck.winrate}%"></div></div>
            </td>
            <td style="color:var(--text-muted)">${deck.total} matches</td>
        </tr>
        `;
    }).join('');

    // 1.5 Render Win Conditions
    let totalPw = 0;
    let totalDw = 0;
    if (currentData) {
        const currProcessedPairs = new Set();
        for (const deckA in currentData) {
            for (const deckB in currentData[deckA]) {
                const pairId = [deckA, deckB].sort().join('-');
                if (!currProcessedPairs.has(pairId)) {
                    currProcessedPairs.add(pairId);
                    const cd = currentData[deckA][deckB];
                    totalPw += (cd[2] || 0) + (cd[5] || 0);
                    totalDw += (cd[3] || 0) + (cd[6] || 0);
                }
            }
        }
    }
    
    const pwEl = document.getElementById('stat-prize-wins');
    const dwEl = document.getElementById('stat-deckout-wins');
    if (pwEl && dwEl) {
        pwEl.innerText = totalPw;
        dwEl.innerText = totalDw;
        const totalWins = totalPw + totalDw;
        if (totalWins > 0) {
            document.getElementById('stat-bar-prize').style.width = `${(totalPw / totalWins) * 100}%`;
            document.getElementById('stat-bar-deckout').style.width = `${(totalDw / totalWins) * 100}%`;
        } else {
            document.getElementById('stat-bar-prize').style.width = `50%`;
            document.getElementById('stat-bar-deckout').style.width = `50%`;
        }
    }

    // 2. Render Heatmap
    const container = document.getElementById('heatmap-container');
    let heatmapDecks = [];
    if (currentData && Object.keys(currentData).length > 0) {
        heatmapDecks = Object.keys(currentData).sort((a,b)=>parseInt(a)-parseInt(b));
    } else if (activeDecks.length > 0) {
        heatmapDecks = [...activeDecks].sort((a,b)=>parseInt(a)-parseInt(b));
    } else {
        heatmapDecks = deckList;
    }
    const size = heatmapDecks.length + 1;
    container.style.gridTemplateColumns = `repeat(${size}, minmax(60px, 1fr))`;
    
    // Header row
    let html = `<div class="heatmap-cell heatmap-header"></div>`;
    heatmapDecks.forEach(d => {
        html += `<div class="heatmap-cell heatmap-header" title="${d}">${formatName(d)}</div>`;
    });
        
    heatmapDecks.forEach(deckA => {
        // Row header
        html += `<div class="heatmap-cell heatmap-header" title="${deckA}" style="justify-content:flex-start">${formatName(deckA)}</div>`;
        
        heatmapDecks.forEach(deckB => {
            if (deckA === deckB) {
                html += `<div class="heatmap-cell" style="background: rgba(255,255,255,0.02);">-</div>`;
            } else {
                let wins = 0, matches = 0;
                let pw_a = 0, dw_a = 0, pw_b = 0, dw_b = 0;
                if (currentData && currentData[deckA] && currentData[deckA][deckB]) {
                    const data = currentData[deckA][deckB];
                    wins = data[0] || 0;
                    matches = data[1] || 0;
                    pw_a = data[2] || 0;
                    dw_a = data[3] || 0;
                    // wins_b is data[4], pw_b is data[5], dw_b is data[6]
                    pw_b = data[5] || 0;
                    dw_b = data[6] || 0;
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
                    const title = `${formatName(deckA)} vs ${formatName(deckB)}\n\nSiege: ${wins} (Prize: ${pw_a}, Deckout: ${dw_a})\nNiederlagen: ${losses} (Prize: ${pw_b}, Deckout: ${dw_b})`;
                    
                    html += `<div class="heatmap-cell" style="background: ${color};" title="${title}">
                        <div style="font-weight: bold; font-size: 1.1em;">${wins}:${losses}</div>
                        <div style="font-size: 0.7em; opacity: 0.8; margin-top: 2px;">(P:${pw_a} D:${dw_a})</div>
                    </div>`;
                }
            }
        });
    });
    
    container.innerHTML = html;
}

function formatName(name) {
    const id = name.replace('.csv', '').replace('decks/', '').replace('deck_', '');
    if (deckNames[id]) {
        return `D${id}: ${deckNames[id]}`;
    }
    return `Deck ${id}`;
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

async function fetchStatus() {
    try {
        const res = await fetch('/decks/status.json');
        if (res.ok) {
            const data = await res.json();
            renderStatus(data);
        }
    } catch(e) {}
}

function renderStatus(data) {
    const textEl = document.getElementById('live-status-text');
    const fillEl = document.getElementById('status-progress-fill');
    const containerEl = document.getElementById('status-progress-container');
    
    if (textEl) {
        let text = data.action || "Idle";
        textEl.innerText = text;
    }
    
    if (fillEl && containerEl && data.total > 0) {
        containerEl.style.display = 'block';
        const percent = (data.completed / data.total) * 100;
        fillEl.style.width = `${percent}%`;
    } else if (containerEl) {
        containerEl.style.display = 'none';
    }
}

async function pollData() {
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
            
            renderDashboard(data, curData);
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
                document.querySelector('.live-indicator').style.background = '#00ff88';
                document.querySelector('.live-indicator').style.boxShadow = '0 0 10px #00ff88';
            } else {
                document.querySelector('.live-indicator').style.background = '#888';
                document.querySelector('.live-indicator').style.boxShadow = 'none';
                document.getElementById('live-status-text').textContent = "Pausiert / Gestoppt";
                document.getElementById('status-progress-container').style.display = 'none';
            }
        })
        .catch(err => {
            console.error('Server offline?', err);
            btnStart.disabled = true;
            btnPause.disabled = true;
            document.querySelector('.live-indicator').style.background = '#ff3c3c';
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
