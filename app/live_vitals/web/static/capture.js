/* Placeholder capture client: records one clip and posts it for analysis. */
const $ = id => document.getElementById(id);
const SETTINGS = { width: 640, height: 480, fps: 30, mbps: 6, seconds: 60 };
let stream = null;

function pickMime() {
  const types = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm'];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}



$('start').onclick = async () => {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: SETTINGS.width }, height: { ideal: SETTINGS.height },
             frameRate: { ideal: SETTINGS.fps }, facingMode: 'user' }
  });
  $('preview').srcObject = stream;
  $('record').disabled = false;
  const s = stream.getVideoTracks()[0].getSettings();
  $('state').textContent = `camera live at ${s.width}x${s.height}`;
};

$('record').onclick = () => {
  const chunks = [];
  const mime = pickMime();
  const rec = new MediaRecorder(stream,
    { mimeType: mime, videoBitsPerSecond: SETTINGS.mbps * 1e6 });

  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  rec.onstop = async () => {
    $('state').textContent = 'analysing…';
    const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
    const body = new FormData();
    body.append('video', new Blob(chunks, { type: mime }), `capture.${ext}`);
    body.append('model', $('model').value);
    const res = await fetch('/api/analyze', { method: 'POST', body });
    const data = await res.json();
    $('out').textContent = JSON.stringify(
      { ...data, waveform: `[${(data.waveform || []).length} samples]` }, null, 2);
    $('state').textContent = data.value != null
      ? `${data.value} ${data.unit} (${data.status})`
      : `no reading: ${data.status || data.message}`;
    $('record').disabled = false;
  };

  rec.start();
  $('record').disabled = true;
  let left = SETTINGS.seconds;
  const tick = setInterval(() => {
    $('state').textContent = `capturing ${--left}s — sit still, face the camera`;
    if (left <= 0) { clearInterval(tick); rec.stop(); }
  }, 1000);
};

