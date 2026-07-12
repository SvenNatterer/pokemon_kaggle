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
            <td>${row.is_champion ? '<strong title="Validation champion">👑 Champion</strong><br>' : ''}${renderDeckNameField(row)}<br><small>${escapeHtml(row.bot_id)}</small></td>
            <td>${escapeHtml(row.bot_type)}</td>
            <td title="Wilson 35%, normalized Elo 25%, arena rate 15%, holdout 25%"><strong>${(row.ranking_score * 100).toFixed(1)}</strong></td>
            <td>${Math.round(row.elo)} <small>(${row.normalized_elo.toFixed(2)})</small></td>
            <td>${percent(row.arena_winrate)}</td>
            <td>${percent(row.arena_wilson)}</td>
            <td style="white-space:nowrap">${row.wins} / ${row.losses} / ${row.draws}</td>
            <td>${row.holdout_missing ? '<span title="Conservative replacement 0.35">missing ⚠</span>' : `${percent(row.holdout_winrate)}<br><small>Wilson ${percent(row.holdout_wilson)}</small>`}</td>
            <td>${row.matches}<br><small>holdout ${row.holdout_games}</small></td>
            <td><label class="replay-eye" title="Replay für ${escapeHtml(row.display_name)} erzeugen"><input class="replay-bot-toggle" type="checkbox" value="${escapeHtml(row.bot_id)}" ${replayBotIds.has(row.bot_id) ? 'checked' : ''}><span aria-hidden="true">👁️</span><span class="sr-only">Replay für ${escapeHtml(row.display_name)}</span></label></td>
        </tr>`).join('') || '<tr><td colspan="11">No participants found.</td></tr>';
    updateReplayButton();
}

function renderDeckNameField(row) {
    const modelName = String(row.display_name || '');
    const match = modelName.match(/^(V\d+|PPO)\s+(.+)$/);
    const prefix = match ? `${match[1]} ` : '';
    const deckName = unsavedDeckNames.get(row.deck_path) || (match ? match[2] : modelName);
    return `<label class="deck-name-field"><span>${escapeHtml(prefix)}</span><input class="deck-name-input" type="text" value="${escapeHtml(deckName)}" data-deck-path="${escapeHtml(row.deck_path)}" aria-label="Deckname für ${escapeHtml(row.bot_id)}" maxlength="80"></label>`;
}

function updateReplayButton() {
    const button = $('replay-generate');
    button.disabled = busy || replayBotIds.size === 0;
    button.textContent = replayBotIds.size ? `Replays erzeugen (${replayBotIds.size})` : 'Replays erzeugen';
}

function renderEvaluation(evaluation, champion) {
    const state = evaluation.state || 'idle';
    $('evaluation-progress').value = Number(evaluation.progress || 0);
    $('evaluation-status').textContent = state === 'idle'
        ? 'No evaluation running.'
        : `${state}: ${evaluation.bot_id || ''} — ${evaluation.completed_games || 0}/${evaluation.planned_games || 0} games, ` +
          `${evaluation.wins || 0} wins, ${evaluation.losses || 0} losses, ${evaluation.draws || 0} draws` +
          (evaluation.error ? ` — ${evaluation.error}` : '');
    $('btn-evaluate').disabled = busy || state === 'running';
    $('btn-promote').disabled = busy || state === 'running' || state !== 'completed' || !evaluation.selection_file;
    $('champion-status').textContent = champion && champion.candidate
        ? `Current champion: ${champion.candidate} (Wilson ${percent(champion.summary?.wilson95_score_lb)})`
        : 'No champion selected. Run validation, then promote its winner.';
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
    renderLeaderboard(data.leaderboard || []);
    renderEvaluation(data.evaluation || {}, data.champion || {});

    const ppoBots = (data.participants || []).filter(p => p.enabled && p.load_status === 'loadable' && p.bot_type === 'ppo');
    const selected = [...$('evaluation-bot').selectedOptions].map(option => option.value);
    $('evaluation-bot').innerHTML = ppoBots.map(p => `<option value="${escapeHtml(p.bot_id)}">${escapeHtml(p.display_name)}</option>`).join('');
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
    $('replay-status').textContent = `${replays.length} replay${replays.length === 1 ? '' : 's'} available`;
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
$('leaderboard-body').addEventListener('change', event => {
    const toggle = event.target.closest('.replay-bot-toggle');
    if (!toggle) return;
    if (toggle.checked) replayBotIds.add(toggle.value);
    else replayBotIds.delete(toggle.value);
    updateReplayButton();
});
$('leaderboard-body').addEventListener('input', event => {
    const input = event.target.closest('.deck-name-input');
    if (input) unsavedDeckNames.set(input.dataset.deckPath, input.value);
});
$('leaderboard-body').addEventListener('keydown', event => {
    const input = event.target.closest('.deck-name-input');
    if (input && event.key === 'Enter') { event.preventDefault(); input.blur(); }
});
$('leaderboard-body').addEventListener('blur', async event => {
    const input = event.target.closest('.deck-name-input');
    if (!input || !input.value.trim()) return;
    const deckPath = input.dataset.deckPath;
    try {
        const result = await api('/api/deck-names', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({deck_path: deckPath, name: input.value.trim()})});
        unsavedDeckNames.delete(deckPath);
        showMessage(result.message);
        await refreshAll();
    } catch (error) { showMessage(error.message, true); }
}, true);
$('replay-generate').addEventListener('click', () => {
    if (replayBotIds.size) action('/api/replays/generate', {bot_ids: [...replayBotIds]});
});
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

refreshAll();
setInterval(refreshAll, 5000);
