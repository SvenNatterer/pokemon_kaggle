const API_BASE = window.location.port === '8080' ? 'http://127.0.0.1:8050' : window.location.origin;
let busy = false;

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
            <td><strong>${escapeHtml(row.display_name)}</strong><br><small>${escapeHtml(row.bot_id)}</small></td>
            <td>${escapeHtml(row.bot_type)}</td>
            <td title="Wilson 35%, normalized Elo 25%, arena rate 15%, holdout 25%"><strong>${(row.ranking_score * 100).toFixed(1)}</strong></td>
            <td>${Math.round(row.elo)} <small>(${row.normalized_elo.toFixed(2)})</small></td>
            <td>${percent(row.arena_winrate)}</td>
            <td>${percent(row.arena_wilson)}</td>
            <td style="white-space:nowrap">${row.wins} / ${row.losses} / ${row.draws}</td>
            <td>${row.holdout_missing ? '<span title="Conservative replacement 0.35">missing ⚠</span>' : `${percent(row.holdout_winrate)}<br><small>Wilson ${percent(row.holdout_wilson)}</small>`}</td>
            <td>${row.matches}<br><small>holdout ${row.holdout_games}</small></td>
        </tr>`).join('') || '<tr><td colspan="9">No participants found.</td></tr>';
}

function renderEvaluation(evaluation) {
    const state = evaluation.state || 'idle';
    $('evaluation-progress').value = Number(evaluation.progress || 0);
    $('evaluation-status').textContent = state === 'idle'
        ? 'No evaluation running.'
        : `${state}: ${evaluation.bot_id || ''} — ${evaluation.completed_games || 0}/${evaluation.planned_games || 0} games, ` +
          `${evaluation.wins || 0} wins, ${evaluation.losses || 0} losses, ${evaluation.draws || 0} draws` +
          (evaluation.error ? ` — ${evaluation.error}` : '');
    $('btn-evaluate').disabled = busy || state === 'running';
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
    renderEvaluation(data.evaluation || {});

    const ppoBots = (data.participants || []).filter(p => p.enabled && p.load_status === 'loadable' && p.bot_type === 'ppo');
    const selected = $('evaluation-bot').value;
    $('evaluation-bot').innerHTML = ppoBots.map(p => `<option value="${escapeHtml(p.bot_id)}">${escapeHtml(p.display_name)}</option>`).join('');
    if (ppoBots.some(p => p.bot_id === selected)) $('evaluation-bot').value = selected;

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
$('btn-reset').addEventListener('click', () => {
    const confirmation = prompt("Type RESET ARENA to delete arena matches/ranking. Models, decks and evaluation results are preserved.");
    if (confirmation === 'RESET ARENA') action('/api/reset', {confirmation, include_replays: false});
});
$('btn-evaluate').addEventListener('click', () => action('/api/evaluation/start', {
    bot_id: $('evaluation-bot').value, games: Number($('evaluation-games').value || 30)
}));

refreshAll();
setInterval(refreshAll, 5000);
