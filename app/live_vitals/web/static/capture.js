/* Capture client for the CRVSE live_vitals demo.
   Records one clip, uploads it once, and renders the returned reading. */
const $ = id => document.getElementById(id);
const SETTINGS = { width: 640, height: 480, fps: 30, mbps: 6, seconds: 60 };
const CHIP = { ACCEPT: 'uk-label-primary', WARN: 'uk-label-secondary',
               REJECT: 'uk-label-destructive', NO_FACE: 'uk-label-secondary' };

let stream = null, framingTimer = null, lastVerdict = null;

function setChip(el, text, verdict) {
  el.textContent = text;
  el.className = 'uk-label ' + (CHIP[verdict] || 'uk-label-secondary');
}

function setConfidence(prefix, value) {
  const bar = $(`${prefix}-conf`);
  if (bar) bar.value = Math.round((value || 0) * 100);
  $(`${prefix}-conf-text`).textContent = (value || 0).toFixed(2);
}

function pickMime() {
  return ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm']
    .find(t => MediaRecorder.isTypeSupported(t)) || '';
}

/* ---------- charts, drawn by FrankenUI's uk-chart (ApexCharts) ----------
   The element is rebuilt rather than mutated: uk-chart reads its options from a
   JSON script child on upgrade, so replacing the element re-initialises it.

   ApexCharts does not inherit MonsterUI colour tokens. Resolve an accessible
   palette from the rendered page background instead of using the light-mode
   colours in every browser. */
function chartPalette() {
  const rgb = (getComputedStyle(document.body).backgroundColor.match(/\d+/g) || [])
    .slice(0, 3).map(Number);
  const dark = rgb.length === 3 && (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) < 128000;
  return dark
    ? { primary: '#60a5fa', pulse: '#2dd4bf', muted: '#64748b', grid: '#243244',
        surface: '#020617', text: '#dbeafe', trend: '#4ade80' }
    : { primary: '#1d4ed8', pulse: '#0f766e', muted: '#94a3b8', grid: '#dbe4ee',
        surface: '#ffffff', text: '#334155', trend: '#15803d' };
}

function drawChart(containerId, opts, emptyMessage) {
  const host = $(containerId);
  if (!opts) {
    host.innerHTML = `<p class="uk-text-muted uk-text-small">${emptyMessage}</p>`;
    return;
  }
  const json = JSON.stringify(opts).replace(/<\/script/gi, '<\\/script');
  host.innerHTML = `<uk-chart><script type="application/json">${json}<\/script></uk-chart>`;
}

function trendOptions(hr, kept, median, secondsPerWindow) {
  if (!hr.length) return null;
  const p = chartPalette();
  const points = hr.map((v, i) => ({ x: +(i * secondsPerWindow).toFixed(1), y: +v.toFixed(1) }));
  const discrete = hr.map((_, i) => ({
    seriesIndex: 0, dataPointIndex: i, size: 5,
    fillColor: kept[i] ? p.trend : p.surface,
    strokeColor: kept[i] ? p.trend : p.muted
  }));
  const annotations = Number.isFinite(median) ? {
    yaxis: [{
      y: median, borderColor: p.trend, strokeDashArray: 6,
      label: { text: `median ${median.toFixed(1)} bpm`, position: 'left',
               textAnchor: 'start', style: { background: p.trend, color: '#fff' } }
    }]
  } : {};
  return {
    chart: { type: 'line', height: 260, fontFamily: 'inherit',
             toolbar: { show: false }, zoom: { enabled: false },
             animations: { enabled: false }, foreColor: p.text },
    series: [{ name: 'heart rate', data: points }],
    colors: [p.trend],
    stroke: { curve: 'straight', width: 2.5 },
    markers: { size: 0, discrete },
    dataLabels: { enabled: false },
    grid: { borderColor: p.grid, strokeDashArray: 4,
            padding: { left: 10, right: 10 } },
    xaxis: { type: 'numeric', tickAmount: 6, title: { text: 'seconds' },
             labels: { formatter: v => Math.round(v) } },
    yaxis: { title: { text: 'bpm' }, decimalsInFloat: 0, forceNiceScale: true },
    annotations,
    tooltip: { x: { formatter: v => `${Math.round(v)} s` },
               y: { formatter: v => `${v.toFixed(1)} bpm` } }
  };
}

// The diagnostic strip deliberately shows all independently inferred windows.
// It is not a continuous ECG/PPG recording: vertical dividers mark joins and
// pale red regions identify the windows rejected by the quality gates.
const PX_PER_SECOND = 100;

function waveOptions(wave, seam, fps, kept) {
  if (!wave.length) return null;
  const p = chartPalette();
  const points = wave.map((v, i) => ({ x: +(i / fps).toFixed(3), y: +v.toFixed(4) }));
  const seconds = wave.length / fps;
  const regions = [];
  (kept || []).forEach((ok, windowIndex) => {
    if (ok) return;
    regions.push({
      x: +(windowIndex * seam / fps).toFixed(2),
      x2: +((windowIndex + 1) * seam / fps).toFixed(2),
      fillColor: '#ef4444', opacity: 0.08, borderColor: 'transparent', label: { text: '' }
    });
  });
  for (let windowIndex = 1; windowIndex * seam < wave.length; windowIndex++) {
    regions.push({ x: +(windowIndex * seam / fps).toFixed(2), borderColor: p.grid,
                   strokeDashArray: 0, opacity: 1 });
  }
  return {
    chart: { type: 'line', height: 220,
             width: Math.max(900, Math.round(seconds * PX_PER_SECOND)),
             fontFamily: 'inherit', toolbar: { show: false },
             zoom: { enabled: false }, animations: { enabled: false }, foreColor: p.text },
    series: [{ name: 'reconstructed BVP', data: points }],
    colors: [p.pulse],
    stroke: { curve: 'straight', width: 1.5, lineCap: 'round' },
    dataLabels: { enabled: false },
    markers: { size: 0 },
    grid: { borderColor: p.grid, strokeDashArray: 0,
            xaxis: { lines: { show: false } }, yaxis: { lines: { show: false } },
            padding: { left: 12, right: 12, top: 0, bottom: 0 } },
    xaxis: { type: 'numeric', tickAmount: 10, title: { text: 'seconds' },
             axisTicks: { show: false }, axisBorder: { show: false },
             labels: { formatter: v => Math.round(v) } },
    yaxis: { show: false },
    annotations: { xaxis: regions },
    tooltip: { enabled: false }
  };
}

/* ---------- live framing guidance ---------- */
async function checkFraming() {
  if (!stream) return;
  const video = $('preview');
  if (!video.videoWidth) return;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.7));
  const body = new FormData();
  body.append('frame', blob, 'frame.jpg');
  try {
    const d = await (await fetch('/api/framing', { method: 'POST', body })).json();
    if (!d.ok) return;
    lastVerdict = d.verdict;
    const text = d.verdict === 'NO_FACE' ? 'no face'
      : d.verdict + (d.notes && d.notes.length ? ' — ' + d.notes[0] : '');
    setChip($('framing-chip'), text, d.verdict);
    $('record').disabled = (d.verdict === 'REJECT' || d.verdict === 'NO_FACE');
  } catch (_) { /* transient failures are not worth surfacing */ }
}

/* ---------- camera ---------- */
$('start').onclick = async () => {
  const start = $('start');
  start.disabled = true;
  try {
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: SETTINGS.width }, height: { ideal: SETTINGS.height },
               frameRate: { ideal: SETTINGS.fps }, facingMode: 'user' }
    });
    $('preview').srcObject = stream;
    const s = stream.getVideoTracks()[0].getSettings();
    $('fps-chip').textContent = `${s.width}x${s.height} @ ${s.frameRate || '?'}`;
    $('state').textContent = 'camera live - check framing, then capture';
    clearInterval(framingTimer);
    framingTimer = setInterval(checkFraming, 1000);
    checkFraming();
  } catch (err) {
    stream = null;
    setChip($('framing-chip'), 'camera unavailable', 'REJECT');
    $('state').textContent = `Could not start the camera: ${err.message || err}`;
  } finally {
    start.disabled = false;
  }
};

/* ---------- capture and analyse ---------- */
$('record').onclick = () => {
  const chunks = [], mime = pickMime();
  const rec = new MediaRecorder(stream,
    { mimeType: mime, videoBitsPerSecond: SETTINGS.mbps * 1e6 });

  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  rec.onstop = async () => {
    clearInterval(framingTimer);
    $('state').textContent = 'analysing…';
    const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
    const body = new FormData();
    body.append('video', new Blob(chunks, { type: mime }), `capture.${ext}`);
    body.append('model', $('model').value);
    try {
      render(await (await fetch('/api/analyze', { method: 'POST', body })).json());
    } catch (err) {
      $('state').textContent = 'analysis failed: ' + err;
    }
    $('record').disabled = false;
    framingTimer = setInterval(checkFraming, 1000);
  };

  rec.start();
  $('record').disabled = true;
  let left = SETTINGS.seconds;
  const tick = setInterval(() => {
    $('state').textContent = `capturing ${--left}s — sit still, face the camera`;
    if (left <= 0) { clearInterval(tick); rec.stop(); }
  }, 1000);
};

/* ---------- rendering ---------- */
let lastResult = null;

/* A chart drawn inside a collapsed accordion measures zero width and comes out
   blank, so the waveform is drawn on demand and redrawn whenever the panel is
   opened. */
function drawWaveform() {
  const d = lastResult;
  if (!d) return;
  const fps = (d.quality && d.quality.effective_fps) || 30;
  $('diag-wave-note').textContent =
    'Drawn at about 100 px per second so individual beats are legible — scroll sideways to read the whole strip. ' +
    'Windows are inferred independently and concatenated; pale red stretches were rejected by the quality gates.';
  drawChart('diag-wave',
            waveOptions(d.waveform || [], 160, fps, d.window_kept || []),
            'No waveform: the model produced no output for this capture.');
}

document.addEventListener('click', event => {
  if (event.target.closest('.uk-accordion-title')) setTimeout(drawWaveform, 400);
});


/* Agreement between the neural and classical estimates. 5 bpm is the tolerance
   ANSI/AAMI EC13 allows a heart-rate meter, so it is the natural line between
   two readings that corroborate each other and two that do not. */
const AGREEMENT_BPM = 5;

function renderCrossCheck(d) {
  const spec = d.spectral;
  if (!spec) return;
  const hr = $('spec-hr');
  hr.textContent = spec.value == null ? '—' : spec.value.toFixed(1);
  hr.classList.toggle('text-muted-foreground', spec.value == null);

  const gap = (spec.value != null && d.value != null)
    ? Math.abs(spec.value - d.value) : null;
  const label = spec.value == null ? (spec.status || 'no reading')
    : gap == null ? spec.method
    : `${spec.method} · ${gap.toFixed(1)} bpm apart`;
  setChip($('spec-status'), label,
          spec.value == null ? 'REJECT'
          : gap == null ? 'WARN'
          : gap <= AGREEMENT_BPM ? 'ACCEPT' : 'WARN');
}


function render(d) {
  if (!d.ok) { $('state').textContent = 'error: ' + d.message; return; }
  lastResult = d;

  const hero = $('model-hr'), q = d.quality || {};
  const level = d.value == null ? 'REJECT'
    : (d.status === 'ok' ? 'ACCEPT'
       : (d.status === 'insufficient_quality' ? 'REJECT' : 'WARN'));

  setChip($('model-status'), d.status || 'no reading', level);
  setChip($('quality-chip'), d.status || 'no reading', level);
  setChip($('trend-chip'), `${d.n_windows ?? 0}/${d.n_total ?? 0} windows`, level);
  setChip($('diag-chip'), d.status || 'no reading', level);
  setConfidence('model', d.confidence);

  if (d.value == null) {
    hero.textContent = '—';
    hero.classList.add('text-muted-foreground');
    $('model-hr-note').textContent = 'refused by the quality gates';
    $('state').textContent = 'No reading: ' + (d.status || d.message || 'unknown') +
      '. Open Diagnostics to see which gate refused it.';
  } else {
    hero.textContent = d.value.toFixed(1);
    hero.classList.remove('text-muted-foreground');
    $('model-hr-note').textContent =
      d.status === 'ok' ? 'model estimate' : 'model estimate — treat as indicative';
    $('state').textContent = `Reading complete — ${d.value.toFixed(1)} ${d.unit}.`;
  }

  $('stat-windows').textContent = `${d.n_windows ?? 0} / ${d.n_total ?? 0}`;
  $('stat-spread').textContent =
    d.spread_bpm == null ? '—' : `${d.spread_bpm.toFixed(1)} bpm`;
  $('stat-rate').textContent =
    q.effective_fps ? `${q.effective_fps.toFixed(1)} fps` : '—';

  const fps = (d.quality && d.quality.effective_fps) || 30;
  const stride = d.n_total > 1 ? (d.quality.n_frames / d.n_total) / fps : 2.67;
  drawChart('trend',
            trendOptions(d.window_hr || [], d.window_kept || [], d.value, stride),
            'No windows to plot.');
  drawWaveform();

  const formatStats = entries => `<dl class="grid grid-cols-[auto_1fr] gap-x-5 gap-y-1 text-xs">` +
    entries.map(([key, value]) => `<dt class="text-muted-foreground">${key}</dt>` +
      `<dd class="font-medium tabular-nums">${Array.isArray(value) ? value.join('; ') : value}</dd>`)
      .join('') + '</dl>';
  $('diag-quality').innerHTML = formatStats(Object.entries(q));
  $('diag-gates').innerHTML = formatStats([
    ['usable fraction', d.usable_fraction], ['windows kept', `${d.n_windows} / ${d.n_total}`],
    ['no cardiac peak', `${d.n_no_peak ?? 0} windows`],
    ['IQR spread', d.spread_bpm == null ? '—' : `${d.spread_bpm} bpm`],
    ['status', d.status]
  ]);
  renderCrossCheck(d);
  const rows = (d.window_hr || []).map((hr, i) =>
    `<tr class="${d.window_kept[i] ? '' : 'text-gray-400'}">
       <td class="pr-6">${i}</td><td class="pr-6">${hr.toFixed(2)}</td>
       <td class="pr-6">${(d.window_confidence[i] ?? 0).toFixed(3)}</td>
       <td>${d.window_kept[i] ? 'kept' : 'dropped'}</td></tr>`).join('');
  $('diag-windows').innerHTML =
    `<table class="uk-table uk-table-small uk-table-divider text-sm"><thead><tr class="text-left">
       <th class="pr-6">#</th><th class="pr-6">HR</th>
       <th class="pr-6">conf</th><th>gate</th></tr></thead><tbody>${rows}</tbody></table>`;
}
