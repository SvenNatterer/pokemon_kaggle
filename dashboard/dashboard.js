const API_BASE = window.location.port === '8080' ? 'http://127.0.0.1:8050' : window.location.origin;
let busy = false;
const replayBotIds = new Set();
const unsavedDeckNames = new Map();

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
    const promotable = Boolean(winner && clearsPerspectiveGate && clearsWilsonGate);
    $('btn-promote').disabled = busy || state !== 'completed' || !evaluation.selection_file || !promotable;
    $('btn-promote').textContent = winner ? `2. ${winner.candidate} promoten` : '2. Promote champion';
    $('btn-promote').title = !winner ? 'Erst eine Validation abschließen.'
        : !clearsPerspectiveGate ? `Nicht promotierbar: Perspektiven-Differenz ${percent(perspectiveGap)} ist größer als 10,0%.`
        : !clearsWilsonGate ? `Nicht promotierbar: Wilson ${percent(winner.wilson95_score_lb)} muss mindestens ${percent(oldWilson + 0.01)} erreichen.`
        : `${winner.candidate} als Champion promoten.`;
    $('champion-status').textContent = champion && champion.candidate
        ? `Current champion: ${champion.candidate} (Wilson ${percent(champion.summary?.wilson95_score_lb)})`
        : 'No champion selected. Run validation, then promote its winner.';
    renderEvaluationResults(evaluation, winner, promotable, clearsPerspectiveGate, clearsWilsonGate, oldWilson, participants);
}

function renderEvaluationResults(evaluation, winner, promotable, clearsPerspectiveGate, clearsWilsonGate, oldWilson, participants) {
    const rows = Array.isArray(evaluation.results) ? evaluation.results : [];
    if (!rows.length) {
        $('evaluation-results').innerHTML = '<span class="muted-line">Noch keine Validation-Ergebnisse.</span>';
        return;
    }
    const winnerName = winner?.candidate || rows[0]?.candidate || '';
    const gate = !winner ? 'Kein Gewinner gespeichert.'
        : promotable ? '✅ Dieser Gewinner kann promotet werden.'
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
    const participant = participants.find(item => String(item.model_path || '').split('/').pop().replace(/\.zip$/, '') === candidate);
    if (participant?.model_path) return participant.model_path;
    const configured = evaluation.configuration?.models || [];
    const paths = Array.isArray(configured) ? configured : [configured];
    return paths.find(path => String(path).split('/').pop().replace(/\.zip$/, '') === candidate) || candidate;
}

function renderStatus(data) {
    const arena = data.arena || {};
    const state = arena.state || 'stopped';
    $('arena-state').textContent = `Arena: ${state}${arena.worker_alive ? ` · worker ${arena.worker_pid}` : ''}` +
        (arena.current_match ? ` · ${arena.current_match.bot_a} vs ${arena.current_match.bot_b}` : '') +
        (arena.error ? ` · ${arena.error}` : '');
    $('btn-start').disabled = busy || state === 'running';
    $('btn-pause').disabled = busy || state !== 'running';
    $('btn-stop').disabled = busy || state === 'stopped';
    // The periodic refresh must not replace a focused name input. Replacing the
    // leaderboard DOM here would interrupt typing and can trigger a premature save.
    const editingDeckName = document.activeElement?.classList.contains('deck-name-input');
    if (!editingDeckName) renderLeaderboard(data.leaderboard || []);
    renderEvaluation(data.evaluation || {}, data.champion || {}, data.participants || []);

    const ppoBots = (data.participants || []).filter(p => p.enabled && p.bot_type === 'ppo');
    const selected = [...$('evaluation-bot').selectedOptions].map(option => option.value);
    $('evaluation-bot').innerHTML = ppoBots.map(p => {
        const filename = String(p.model_path || '').split('/').pop();
        const tags = (p.tags || []).length ? ` [${p.tags.join(', ')}]` : '';
        const unavailable = p.load_status !== 'loadable';
        const status = unavailable ? ` — NICHT VERFÜGBAR: ${p.load_status}` : '';
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
    return `${(bytes / 1024).toFixed(1)} KB`;
}

function renderReplays(replays) {
    const watchStatus = $('replay-status').dataset.watchStatus || 'Keine Bots für automatische Replays markiert.';
    $('replay-status').textContent = `${watchStatus} ${replays.length} Replay${replays.length === 1 ? '' : 's'} verfügbar.`;
    $('replay-list').innerHTML = replays.map(replay => {
        const url = `${API_BASE}${replay.url}`;
        const meta = replay.metadata || {};
        const title = meta.p0_name || meta.p1_name ? `${meta.p0_name || 'Player 0'} vs ${meta.p1_name || 'Player 1'}` : replay.name;
        return `<article class="replay-item"><div class="replay-item-title">${escapeHtml(title)}</div>
            <div class="replay-meta"><span class="replay-pill">${escapeHtml(replay.group)}</span><span class="replay-pill">${formatBytes(replay.size)}</span></div>
            <div class="replay-path">${escapeHtml(replay.path)}</div><div class="replay-links">
            <button class="btn btn-primary" onclick="launchHeroz('${escapeHtml(url)}')">HERoz Viz</button>
            <a class="btn btn-secondary" href="${escapeHtml(url)}" target="_blank" rel="noopener">JSON</a></div></article>`;
    }).join('');
}

async function loadReplays() {
    try { renderReplays((await api('/api/replays')).replays || []); }
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

$('btn-start').addEventListener('click', () => action('/api/start'));
$('btn-pause').addEventListener('click', () => action('/api/pause'));
$('btn-stop').addEventListener('click', () => action('/api/stop'));
$('btn-refresh').addEventListener('click', refreshAll);
$('replay-refresh').addEventListener('click', loadReplays);
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
    if (!bot_ids.length) return showMessage('Select at least one PPO candidate.', true);
    action('/api/evaluation/start', {
        bot_ids, mode: $('evaluation-mode').value, games: Number($('evaluation-games').value || 30)
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
