/* Real-time dashboard SSE client */
let paused = false;
const liveRows = {};  // trace_id → row element

function togglePause() {
  paused = !paused;
  document.getElementById('pause-btn').textContent = paused ? '▶ Resume' : '⏸ Pause';
}

function setStatus(msg) {
  document.getElementById('status-bar').textContent = msg;
}

function badgeHtml(status) {
  return `<span class="badge badge-${status}">${status}</span>`;
}

function spinnerHtml() {
  return `<span class="spinner"></span>`;
}

function ensureRow(traceId) {
  if (!liveRows[traceId]) {
    const div = document.createElement('div');
    div.className = 'live-row';
    div.id = `live-${traceId}`;
    div.innerHTML = `
      <code style="font-size:.8rem">${traceId.substring(0, 8)}</code>
      <span class="live-project" style="color:var(--text-muted)">—</span>
      <span class="live-story truncate" style="max-width:300px">—</span>
      <span class="live-agent" style="color:var(--accent2)">—</span>
      <span class="live-tokens" style="color:var(--text-muted);font-size:.8rem">0 tokens</span>
      <span class="live-status">${spinnerHtml()}</span>
    `;
    document.getElementById('live-feed').prepend(div);
    liveRows[traceId] = div;
  }
  return liveRows[traceId];
}

const es = new EventSource('/events');

es.onopen = () => setStatus('Connected — listening for events');
es.onerror = () => setStatus('Connection lost — retrying…');

es.onmessage = (e) => {
  if (paused) return;
  let ev;
  try { ev = JSON.parse(e.data); } catch { return; }

  const traceId = ev.trace_id;
  if (!traceId) return;

  if (ev.type === 'request_started') {
    const row = ensureRow(traceId);
    if (ev.project_name) row.querySelector('.live-project').textContent = ev.project_name;
    if (ev.agent_name)   row.querySelector('.live-agent').textContent   = ev.agent_name;
  }

  if (ev.type === 'request_updated') {
    const row = ensureRow(traceId);
    if (ev.agent_name)      row.querySelector('.live-agent').textContent = ev.agent_name;
    if (ev.tool_name)       row.querySelector('.live-agent').textContent += ` → ${ev.tool_name}`;
    if (ev.output_tokens != null) {
      const total = (ev.input_tokens || 0) + (ev.output_tokens || 0);
      row.querySelector('.live-tokens').textContent = total + ' tokens';
    }
  }

  if (ev.type === 'request_completed') {
    const row = liveRows[traceId];
    if (!row) return;
    const verdict = ev.verdict || 'completed';
    row.querySelector('.live-status').innerHTML = badgeHtml(verdict);
    row.querySelector('.live-tokens').textContent = ((ev.input_tokens || 0) + (ev.output_tokens || 0)) + ' tokens';
    row.querySelector('.live-agent').textContent = '—';
  }
};
