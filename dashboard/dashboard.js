const API_BASE = window.location.port === '8080' ? 'http://127.0.0.1:8050' : window.location.origin;
let busy = false;
const replayBotIds = new Set();
const unsavedDeckNames = new Map();
const replayGroupOpenState = new Map();
let allReplays = [];

const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const percent = value => value == null ? 'missing' : `${(Number(value) * 100).toFixed(1)}%`;

async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {cache: 'no-store', ...options});
    let payload = {};
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(payload.message || `${response.status} ${response.statusText}`);
    return payload;
}

function showMessage(message, error = false) {
    const node = $('action-message');
    node.textContent = message || '';
    node.style.color = error ? '#f43f5e' : '#10b981';
}

function setBusy(value) {
    busy = value;
    document.querySelectorAll('.control-panel button').forEach(button => button.disabled = value);
}

function renderLeaderboard(rows) {
    $('leaderboard-body').innerHTML = rows.map(row => `
        <tr>
            <td>#${row.rank}</td>
            <td class="bot-cell">${row.is_champion ? '<strong title="Validation champion">👑 Champion</strong><br>' : ''}${renderDeckNameField(row)}<small class="bot-id" title="${escapeHtml(row.bot_id)}">ID: ${escapeHtml(row.bot_id)}</small>${row.model_path ? `<small class="model-path" title="${escapeHtml(row.model_path)}">Modell: ${escapeHtml(row.model_path)}</small>` : ''}</td>
            <td>${escapeHtml(row.bot_type)}</td>
            <td title="Arena ranking: Wilson 50%, Elo strength 35%, win rate 15%"><strong>${(row.ranking_score * 100).toFixed(1)}</strong></td>
            <td>${Math.round(row.elo)} <small>(${row.normalized_elo.toFixed(2)})</small></td>
            <td>${percent(row.arena_winrate)}</td>
            <td>${percent(row.arena_wilson)}</td>
            <td style="white-space:nowrap">${row.wins} / ${row.losses} / ${row.draws}</td>
            <td>${row.matches}</td>
            <td><label class="replay-eye" title="Replay für ${escapeHtml(row.display_name)} erzeugen"><input class="replay-bot-toggle" type="checkbox" value="${escapeHtml(row.bot_id)}" ${replayBotIds.has(row.bot_id) ? 'checked' : ''}><span aria-hidden="true">👁️</span><span class="sr-only">Replay für ${escapeHtml(row.display_name)}</span></label></td>
        </tr>`).join('') || '<tr><td colspan="10">No participants found.</td></tr>';
    renderHoldoutResults(rows);
    updateReplayButton();
}

function renderHoldoutResults(rows) {
    const completed = rows.filter(row => !row.holdout_missing && Number(row.holdout_games) > 0);
    $('holdout-results').innerHTML = completed.length ? completed.map(row => `
        <article class="holdout-result">
            <span class="holdout-name" title="${escapeHtml(row.display_name)}">${escapeHtml(row.display_name)}</span>
            <strong>${percent(row.holdout_winrate)}</strong>
            <small>Wilson ${percent(row.holdout_wilson)} · ${row.holdout_games} Spiele</small>
        </article>`).join('') : '<span class="muted-line">Noch keine Holdout-Ergebnisse.</span>';
}

function renderMatchupMatrix(rows, pairwiseResults) {
    const container = $('matchup-matrix');
    if (!rows.length) {
        container.innerHTML = '<span class="muted-line">Keine Teilnehmer gefunden.</span>';
        return;
    }
    const results = new Map();
    for (const pair of pairwiseResults || []) {
        results.set(`${pair.bot_a}\u0000${pair.bot_b}`, pair);
        results.set(`${pair.bot_b}\u0000${pair.bot_a}`, {
            ...pair, wins_a: pair.wins_b, wins_b: pair.wins_a,
        });
    }
    const name = row => row.display_name || row.bot_id;
    const header = rows.map(row => `<th title="${escapeHtml(name(row))}">${escapeHtml(name(row))}</th>`).join('');
    const body = rows.map(row => `<tr><th title="${escapeHtml(name(row))}">${escapeHtml(name(row))}</th>${rows.map(opponent => {
        if (row.bot_id === opponent.bot_id) return '<td class="matchup-self" aria-label="gleicher Teilnehmer">—</td>';
        const result = results.get(`${row.bot_id}\u0000${opponent.bot_id}`);
        const games = result ? Number(result.wins_a) + Number(result.wins_b) + Number(result.draws) : 0;
        if (!games) return '<td class="matchup-empty">–</td>';
        const rate = (Number(result.wins_a) + 0.5 * Number(result.draws)) / games;
        const hue = Math.round(rate * 120);
        const title = `${result.wins_a} Siege / ${result.wins_b} Niederlagen / ${result.draws} Unentschieden (${games} Spiele)`;
        return `<td class="matchup-result" style="--matchup-hue:${hue}" title="${escapeHtml(title)}"><strong>${percent(rate)}</strong><small>${games} Sp.</small></td>`;
    }).join('')}</tr>`).join('');
    container.innerHTML = `<table class="matchup-matrix"><thead><tr><th>Zeile \\ Spalte</th>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderDeckNameField(row) {
    const name = unsavedDeckNames.get(row.bot_id) || String(row.display_name || '');
    return `<label class="deck-name-field"><input class="deck-name-input" type="text" value="${escapeHtml(name)}" data-bot-id="${escapeHtml(row.bot_id)}" aria-label="Name für Checkpoint ${escapeHtml(row.bot_id)}" maxlength="100" title="Diesen Checkpoint unabhängig umbenennen"></label>`;
}

function updateReplayButton() {
    const selected = replayBotIds.size;
    $('replay-status').dataset.watchStatus = selected
        ? `${selected} Bot${selected === 1 ? '' : 's'} für automatische Replays markiert.`
        : 'Keine Bots für automatische Replays markiert.';
}

function renderEvaluation(evaluation, champion, participants = []) {
    const state = evaluation.state || 'idle';
    $('evaluation-progress').value = Number(evaluation.progress || 0);
    $('evaluation-status').textContent = state === 'idle'
        ? 'No evaluation running.'
        : `${state}: ${evaluation.bot_id || ''} — ${evaluation.completed_games || 0}/${evaluation.planned_games || 0} games, ` +
          `${evaluation.wins || 0} wins, ${evaluation.losses || 0} losses, ${evaluation.draws || 0} draws` +
          (evaluation.error ? ` — ${evaluation.error}` : '');
    $('btn-evaluate').disabled = busy || state === 'running';
    const selection = evaluation.selection || {};
    const winner = selection.summary || null;
    const oldWilson = Number(champion?.summary?.wilson95_score_lb || 0);
    const perspectiveGap = Number(winner?.perspective_score_gap || 0);
    const clearsPerspectiveGate = perspectiveGap <= 0.10;
    const clearsWilsonGate = Number(winner?.wilson95_score_lb || 0) >= oldWilson + 0.01;
    const candidateConfigs = evaluation.configuration?.candidates || [];
    const winnerConfig = candidateConfigs.find(item => item.label === winner?.candidate || item.bot_id === winner?.candidate);
    const isPpoWinner = !winnerConfig || winnerConfig.bot_type === 'ppo';
    const promotable = Boolean(winner && isPpoWinner && clearsPerspectiveGate && clearsWilsonGate);
    $('btn-promote').disabled = busy || state !== 'completed' || !evaluation.selection_file || !promotable;
    $('btn-promote').textContent = winner ? `2. ${winner.candidate} promoten` : '2. Promote champion';
    $('btn-promote').title = !winner ? 'Erst eine Validation abschließen.'
        : !isPpoWinner ? 'Rule-Bots können evaluiert, aber nicht als PPO-Champion promotet werden.'
        : !clearsPerspectiveGate ? `Nicht promotierbar: Perspektiven-Differenz ${percent(perspectiveGap)} ist größer als 10,0%.`
        : !clearsWilsonGate ? `Nicht promotierbar: Wilson ${percent(winner.wilson95_score_lb)} muss mindestens ${percent(oldWilson + 0.01)} erreichen.`
        : `${winner.candidate} als Champion promoten.`;
    $('champion-status').textContent = champion && champion.candidate
        ? `Current champion: ${champion.candidate} (Wilson ${percent(champion.summary?.wilson95_score_lb)})`
        : 'No champion selected. Run validation, then promote its winner.';
    renderEvaluationResults(evaluation, winner, promotable, isPpoWinner, clearsPerspectiveGate, clearsWilsonGate, oldWilson, participants);
}

function renderEvaluationResults(evaluation, winner, promotable, isPpoWinner, clearsPerspectiveGate, clearsWilsonGate, oldWilson, participants) {
    const rows = Array.isArray(evaluation.results) ? evaluation.results : [];
    if (!rows.length) {
        $('evaluation-results').innerHTML = '<span class="muted-line">Noch keine Validation-Ergebnisse.</span>';
        return;
    }
    const winnerName = winner?.candidate || rows[0]?.candidate || '';
    const gate = !winner ? 'Kein Gewinner gespeichert.'
        : promotable ? '✅ Dieser Gewinner kann promotet werden.'
        : !isPpoWinner ? 'ℹ️ Rule-Bot-Auswertung: keine PPO-Champion-Promotion.'
        : !clearsPerspectiveGate ? `⛔ Nicht promotierbar: Perspektiven-Differenz ${percent(winner.perspective_score_gap)} > 10,0%.`
        : !clearsWilsonGate ? `⛔ Nicht promotierbar: Wilson muss mindestens ${percent(oldWilson + 0.01)} erreichen.`
        : '⛔ Nicht promotierbar.';
    $('evaluation-results').innerHTML = `
        <div class="evaluation-winner"><strong>Ausgewählter Gewinner: ${escapeHtml(winnerName)}</strong><span>${escapeHtml(gate)}</span></div>
        <div class="evaluation-result-grid">${rows.map((row, index) => `
            <article class="evaluation-result ${row.candidate === winnerName ? 'is-winner' : ''}">
                <div><strong>#${index + 1} ${escapeHtml(row.candidate)}</strong>${row.candidate === winnerName ? '<span class="winner-badge">Gewinner</span>' : ''}</div>
                <small>Modell: ${escapeHtml(modelPathForCandidate(row.candidate, evaluation, participants))}</small>
                <dl><div><dt>Score</dt><dd>${percent(row.score_rate)}</dd></div><div><dt>Wilson</dt><dd>${percent(row.wilson95_score_lb)}</dd></div><div><dt>Schlechtester Gegner</dt><dd>${percent(row.worst_score_rate)}</dd></div><div><dt>Perspektiven-Differenz</dt><dd>${percent(row.perspective_score_gap)}</dd></div><div><dt>W / L / D</dt><dd>${row.wins} / ${row.losses} / ${row.draws}</dd></div><div><dt>Spiele</dt><dd>${row.games}</dd></div></dl>
            </article>`).join('')}</div>`;
}

function modelPathForCandidate(candidate, evaluation, participants) {
    const configuredCandidates = evaluation.configuration?.candidates || [];
    const configuredCandidate = configuredCandidates.find(item => item.label === candidate || item.bot_id === candidate);
    if (configuredCandidate?.model_path) return configuredCandidate.model_path;
    const participant = participants.find(item => item.bot_id === candidate || String(item.model_path || '').split('/').pop().replace(/\.zip$/, '') === candidate);
    if (participant?.model_path) return participant.model_path;
    const configured = evaluation.configuration?.models || [];
    const paths = Array.isArray(configured) ? configured : [configured];
    return paths.find(path => String(path).split('/').pop().replace(/\.zip$/, '') === candidate) || candidate;
}

function renderStatus(data) {
    const arena = data.arena || {};
    const state = arena.state || 'stopped';
    let statusText = `Arena: ${state}${arena.worker_alive ? ` · worker ${arena.worker_pid}` : ''}`;
    if (arena.error) {
        statusText += ` · <span style="color: var(--danger, #ff4d4f);">${arena.error}</span>`;
    }
    
    let activeMatchesHtml = '';
    const active = arena.active_matches || [];
    if (active.length > 0) {
        activeMatchesHtml = '<div style="margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.9rem; text-align: center;">' +
            active.map((match, i) => `<div>⚔️ Match ${i+1}: <strong>${match.bot_a}</strong> vs <strong>${match.bot_b}</strong></div>`).join('') +
            '</div>';
    } else if (arena.current_match) {
        activeMatchesHtml = `<div style="margin-top: 0.5rem; font-size: 0.9rem; text-align: center;">⚔️ Match: <strong>${arena.current_match.bot_a}</strong> vs <strong>${arena.current_match.bot_b}</strong></div>`;
    }
    
    $('arena-state').innerHTML = `<div>${statusText}</div>${activeMatchesHtml}`;
    $('btn-start').disabled = busy || state === 'running';
    $('btn-pause').disabled = busy || state !== 'running';
    $('btn-stop').disabled = busy || state === 'stopped';
    // The periodic refresh must not replace a focused name input. Replacing the
    // leaderboard DOM here would interrupt typing and can trigger a premature save.
    const editingDeckName = document.activeElement?.classList.contains('deck-name-input');
    if (!editingDeckName) renderLeaderboard(data.leaderboard || []);
    renderMatchupMatrix(data.leaderboard || [], data.pairwise_results || []);
    renderEvaluation(data.evaluation || {}, data.champion || {}, data.participants || []);

    const evaluationBots = (data.participants || []).filter(p => p.enabled && ['ppo', 'rule_based'].includes(p.bot_type));
    const selected = [...$('evaluation-bot').selectedOptions].map(option => option.value);
    $('evaluation-bot').innerHTML = evaluationBots.map(p => {
        const filename = p.bot_type === 'rule_based'
            ? String(p.model_path || 'rule_based')
            : String(p.model_path || '').split('/').pop();
        const tags = (p.tags || []).length ? ` [${p.tags.join(', ')}]` : '';
        const unavailable = !['loadable', 'cooldown'].includes(p.load_status);
        const status = p.load_status === 'cooldown'
            ? ' — Arena-Cooldown (Validation möglich)'
            : unavailable ? ` — NICHT VERFÜGBAR: ${p.load_status}` : '';
        const title = [p.model_path, p.load_error].filter(Boolean).join(' — ');
        return `<option value="${escapeHtml(p.bot_id)}" title="${escapeHtml(title)}" ${unavailable ? 'disabled' : ''}>${escapeHtml(p.display_name)} — ${escapeHtml(filename)}${escapeHtml(tags)}${escapeHtml(status)}</option>`;
    }).join('');
    for (const id of selected) {
        const option = [...$('evaluation-bot').options].find(item => item.value === id);
        if (option) option.selected = true;
    }

    const failures = data.errors || [];
    const loadable = (data.participants || []).length - failures.length;
    $('bot-diagnostics').innerHTML = `<strong>${loadable} loadable / ${(data.participants || []).length} total</strong>` +
        (failures.length ? `<ul>${failures.map(p => `<li><code>${escapeHtml(p.bot_id)}</code>: ${escapeHtml(p.load_error)}</li>`).join('')}</ul>` : '<p>No load errors.</p>');

}

async function refreshAll() {
    try {
        renderStatus(await api('/api/refresh'));
        await loadReplays();
    } catch (error) {
        showMessage(error.message, true);
    }
}

async function action(path, body) {
    if (busy) return;
    setBusy(true);
    showMessage('Working…');
    try {
        const result = await api(path, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
        showMessage(result.message || 'Done.');
    } catch (error) {
        showMessage(error.message, true);
    } finally {
        setBusy(false);
        await refreshAll();
    }
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(1)} KB`;
}

function normalizeReplay(replay) {
    if (replay.source && replay.collection) return replay;
    const parts = String(replay.path || '').replaceAll('\\', '/').split('/');
    const folder = parts[0] === 'replays' ? parts[1] : '';
    if (folder === 'arena') return {...replay, source: 'arena', group: 'Arena', collection: 'Arena'};
    if (folder === 'kaggle') {
        const submission = parts.length > 3 ? parts[2] : '';
        return {...replay, source: 'kaggle', group: 'Kaggle', collection: submission ? `Submission ${submission}` : 'Kaggle'};
    }
    if (folder === 'test') {
        const collection = parts.length > 3 ? parts[2] : 'Tests';
        return {...replay, source: 'test', group: 'Tests', collection};
    }
    return {...replay, source: 'other', group: replay.group || 'Other', collection: replay.collection || 'Other'};
}

function replayTitle(replay) {
    const meta = replay.metadata || {};
    return meta.p0_name || meta.p1_name
        ? `${meta.p0_name || 'Player 0'} vs ${meta.p1_name || 'Player 1'}`
        : replay.name;
}

function replayGroupTitle(replay) {
    if (replay.collection && replay.collection !== replay.group) {
        return `${replay.group} · ${replay.collection}`;
    }
    return replay.group || 'Other';
}

function replayGroupDescription(replay) {
    return replay.source === 'kaggle' ? String(replay.submission_description || '').trim() : '';
}

function formatReplayDate(mtime) {
    const date = new Date(Number(mtime) * 1000);
    return Number.isNaN(date.getTime())
        ? ''
        : new Intl.DateTimeFormat('de-DE', {dateStyle: 'medium', timeStyle: 'short'}).format(date);
}

function renderReplayCard(replay) {
    const url = `${API_BASE}${replay.url}`;
    const meta = replay.metadata || {};
    const details = [
        formatReplayDate(replay.mtime),
        formatBytes(replay.size),
        replay.snapshots == null ? '' : `${replay.snapshots} Schritte`,
        meta.episode_id ? `Episode ${meta.episode_id}` : '',
        replay.result || '',
    ].filter(Boolean);
    return `<article class="replay-item"><div class="replay-item-title">${escapeHtml(replayTitle(replay))}</div>
        <div class="replay-meta">${details.map(value => `<span class="replay-pill">${escapeHtml(value)}</span>`).join('')}</div>
        ${replay.status ? `<div class="replay-status-detail">${escapeHtml(replay.status)}</div>` : ''}
        <div class="replay-path">${escapeHtml(replay.path)}</div><div class="replay-links">
        <button class="btn btn-primary" onclick="launchHeroz('${escapeHtml(url)}')">HERoz Viz</button>
        <a class="btn btn-secondary" href="${escapeHtml(url)}" target="_blank" rel="noopener">JSON</a></div></article>`;
}

function renderReplaySummary(replays) {
    const counts = {arena: 0, kaggle: 0, test: 0, other: 0};
    for (const replay of replays) counts[replay.source in counts ? replay.source : 'other'] += 1;
    $('replay-summary').innerHTML = [
        ['arena', 'Arena'],
        ['kaggle', 'Kaggle'],
        ['test', 'Tests'],
        ['other', 'Other'],
    ].filter(([source]) => counts[source]).map(([source, label]) =>
        `<button class="replay-summary-pill" type="button" data-replay-source="${source}"><strong>${counts[source]}</strong><span>${label}</span></button>`
    ).join('');
}

function renderReplays(replays = allReplays) {
    const query = $('replay-search').value.trim().toLocaleLowerCase('de');
    const source = $('replay-source').value;
    const sort = $('replay-sort').value;
    const visible = replays.filter(replay => {
        if (source !== 'all' && replay.source !== source) return false;
        if (!query) return true;
        const meta = replay.metadata || {};
        return [
            replay.name, replay.path, replay.group, replay.collection,
            replay.submission_description, meta.p0_name, meta.p1_name,
            meta.episode_id, replay.result, replay.status,
        ].some(value => String(value || '').toLocaleLowerCase('de').includes(query));
    }).sort((left, right) => {
        if (sort === 'oldest') return Number(left.mtime) - Number(right.mtime);
        if (sort === 'name') return replayTitle(left).localeCompare(replayTitle(right), 'de');
        return Number(right.mtime) - Number(left.mtime);
    });

    const watchStatus = $('replay-status').dataset.watchStatus || 'Keine Bots für automatische Replays markiert.';
    $('replay-status').textContent = `${watchStatus} ${visible.length} von ${replays.length} Replays angezeigt.`;
    renderReplaySummary(replays);

    const groups = new Map();
    for (const replay of visible) {
        const key = encodeURIComponent(`${replay.source || 'other'}\u0000${replay.collection || replay.group || 'Other'}`);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(replay);
    }
    const sourceOrder = {arena: 0, kaggle: 1, test: 2, other: 3};
    const groupedReplays = [...groups.entries()].sort(([, left], [, right]) =>
        (sourceOrder[left[0].source] ?? sourceOrder.other) - (sourceOrder[right[0].source] ?? sourceOrder.other)
    );
    $('replay-list').innerHTML = groupedReplays.map(([key, items], index) => {
        const title = replayGroupTitle(items[0]);
        const description = replayGroupDescription(items[0]);
        const isOpen = replayGroupOpenState.has(key) ? replayGroupOpenState.get(key) : Boolean(query || index === 0);
        return `<details class="replay-group" data-replay-group-key="${escapeHtml(key)}"${isOpen ? ' open' : ''}>
            <summary><span class="replay-group-heading"><span>${escapeHtml(title)}</span>${description ? `<span class="replay-group-description">${escapeHtml(description)}</span>` : ''}</span><span class="replay-group-count">${items.length} Replay${items.length === 1 ? '' : 's'}</span></summary>
            <div class="replay-grid">${items.map(renderReplayCard).join('')}</div>
        </details>`;
    }).join('') || '<div class="replay-empty">Keine Replays passen zu den aktuellen Filtern.</div>';
}

async function loadReplays() {
    try {
        allReplays = ((await api('/api/replays')).replays || []).map(normalizeReplay);
        renderReplays();
    }
    catch (error) { $('replay-status').textContent = error.message; }
}

window.launchHeroz = async function(url) {
    const visualizerUrl = new URL('/dashboard/heroz_visualizer.html', window.location.href);
    const popup = window.open(visualizerUrl.href, 'heroz-visualizer');
    if (!popup) return alert('Please allow pop-ups for this dashboard.');
    try {
        const text = await (await fetch(url)).text();
        const obj = JSON.parse(text);
        const replayJson = 'steps' in obj ? JSON.stringify(obj.steps[0][0].visualize) : text;
        const send = () => !popup.closed && popup.postMessage({type:'heroz-replay', replayJson}, visualizerUrl.origin);
        send(); setTimeout(send, 250); setTimeout(send, 750);
    } catch (error) { popup.close(); alert(`Replay error: ${error.message}`); }
};

$('btn-start').addEventListener('click', () => {
    const workersVal = parseInt($('num-workers').value, 10) || 4;
    action('/api/start', { workers: workersVal });
});
$('btn-pause').addEventListener('click', () => action('/api/pause'));
$('btn-stop').addEventListener('click', () => action('/api/stop'));
$('btn-refresh').addEventListener('click', refreshAll);
$('replay-refresh').addEventListener('click', loadReplays);
$('replay-search').addEventListener('input', () => renderReplays());
$('replay-source').addEventListener('change', () => renderReplays());
$('replay-sort').addEventListener('change', () => renderReplays());
$('replay-list').addEventListener('toggle', event => {
    const group = event.target.closest('details.replay-group');
    if (group) replayGroupOpenState.set(group.dataset.replayGroupKey, group.open);
}, true);
$('replay-summary').addEventListener('click', event => {
    const button = event.target.closest('[data-replay-source]');
    if (!button) return;
    $('replay-source').value = button.dataset.replaySource;
    renderReplays();
});
$('leaderboard-body').addEventListener('change', async event => {
    const toggle = event.target.closest('.replay-bot-toggle');
    if (!toggle) return;
    if (toggle.checked) replayBotIds.add(toggle.value);
    else replayBotIds.delete(toggle.value);
    updateReplayButton();
    try {
        await api('/api/watched', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({watched: [...replayBotIds]})});
        await loadReplays();
    } catch (error) {
        if (toggle.checked) replayBotIds.delete(toggle.value);
        else replayBotIds.add(toggle.value);
        toggle.checked = !toggle.checked;
        updateReplayButton();
        showMessage(error.message, true);
    }
});
$('leaderboard-body').addEventListener('input', event => {
    const input = event.target.closest('.deck-name-input');
    if (input) unsavedDeckNames.set(input.dataset.botId, input.value);
});
$('leaderboard-body').addEventListener('keydown', event => {
    const input = event.target.closest('.deck-name-input');
    if (input && event.key === 'Enter') { event.preventDefault(); input.blur(); }
});
$('leaderboard-body').addEventListener('blur', async event => {
    const input = event.target.closest('.deck-name-input');
    if (!input || !input.value.trim()) return;
    const botId = input.dataset.botId;
    try {
        const result = await api('/api/bot-names', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({bot_id: botId, name: input.value.trim()})});
        unsavedDeckNames.delete(botId);
        showMessage(result.message);
        await refreshAll();
    } catch (error) { showMessage(error.message, true); }
}, true);
$('btn-reset').addEventListener('click', () => {
    const confirmation = prompt("Type RESET ARENA to delete arena matches/ranking. Models, decks and evaluation results are preserved.");
    if (confirmation === 'RESET ARENA') action('/api/reset', {confirmation, include_replays: false});
});
$('btn-evaluate').addEventListener('click', () => {
    const bot_ids = [...$('evaluation-bot').selectedOptions].map(option => option.value);
    if (!bot_ids.length) return showMessage('Select at least one evaluation candidate.', true);
    action('/api/evaluation/start', {
        bot_ids, mode: $('evaluation-mode').value, games: Number($('evaluation-games').value || 100)
    });
});
$('btn-promote').addEventListener('click', () => action('/api/champion/promote', {
    min_wilson_improvement: 0.01, max_perspective_gap: 0.10
}));

async function initialize() {
    try {
        const data = await api('/api/watched');
        for (const botId of data.watched || []) replayBotIds.add(String(botId));
    } catch (error) {
        showMessage(error.message, true);
    }
    await refreshAll();
}

initialize();
setInterval(refreshAll, 5000);
