"""FastHTML server for the CRVSE live_vitals research demo.

The browser captures a clip and uploads it once; this server runs the same
`crops_from_video` path the CLI and the validation scripts use, so preprocessing
parity with the training notebooks is structural rather than reimplemented.

Uploaded video is written to a temporary file, analysed, and deleted in a
`finally` block. No frame, crop or clip is ever persisted.

Run:
    python -m app.live_vitals.web.app
    python -m app.live_vitals.web.app --https   # for phone testing on the LAN
"""
import sys
import tempfile
import threading
import socket
from pathlib import Path

import numpy as np
import cv2
from fasthtml.common import *
from monsterui.all import *
import fasthtml.common as fh
from starlette.datastructures import UploadFile

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.capture.session import (crops_from_video, framing_verdict, stable_box)
from app.live_vitals.estimators import registry
from app.live_vitals.preprocess.face_box import make_landmarker

CERT_DIR = _REPO_ROOT / "app" / "live_hr_demo" / "certs"
DISCLAIMER = ("Research demo, not a medical device. Not validated for diagnosis "
              "or treatment decisions.")

# MediaPipe detectors are not safe to call concurrently, and FastHTML runs sync
# handlers in a threadpool, so one shared detector is guarded by a lock. Loading
# a checkpoint costs a few hundred milliseconds, so estimators are cached too.
_LANDMARKER = None
_LANDMARKER_LOCK = threading.Lock()
_ESTIMATORS = {}


def landmarker():
    """Returns the shared MediaPipe landmarker, creating it on first use."""
    global _LANDMARKER
    if _LANDMARKER is None:
        _LANDMARKER = make_landmarker()
    return _LANDMARKER


def estimator(name):
    """Returns a cached estimator instance by registered name."""
    if name not in _ESTIMATORS:
        _ESTIMATORS[name] = registry.get_estimator(name)
    return _ESTIMATORS[name]


def quality_payload(quality):
    """Serialises capture diagnostics for the browser."""
    return dict(width=quality.width, height=quality.height,
                effective_fps=round(quality.effective_fps, 2),
                n_frames=quality.n_frames,
                detections=quality.detections,
                detections_attempted=quality.detections_attempted,
                verdict=quality.verdict,
                crop_aspect=round(quality.crop_aspect, 3),
                frac_side_lost=round(quality.frac_side_lost, 3),
                notes=list(quality.notes))


def lan_address():
    """Returns this machine's LAN address, for reaching the server from a phone."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Selects a route without sending anything, which reveals the local
        # address of the interface that would carry outbound traffic.
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except Exception:
        return None
    finally:
        probe.close()


def announce(scheme, port):
    """Prints the URLs the server can be reached on."""
    print(f"\nLink: {scheme}://localhost:{port}")
    lan = lan_address()
    if lan:
        print(f"LAN : {scheme}://{lan}:{port}   (for phone testing)\n")

app, rt = fast_app(
    hdrs=(*Theme.blue.headers(),
          Link(rel="icon", type="image/x-icon", href="/static/favicon.ico")),
    live=False, static_path=str(Path(__file__).parent))

@rt("/api/models")
def api_models():
    """Lists registered estimators and whether their weights are present."""
    models = []
    for name in registry.available():
        est = registry.get_estimator(name)
        models.append(dict(name=name, vital=est.vital, unit=est.unit,
                           available=est.is_available(),
                           default=(name == registry.DEFAULT_ESTIMATOR)))
    return dict(models=models)


@rt("/api/framing", methods=["POST"])
async def api_framing(frame: UploadFile):
    """Classifies framing from a single preview frame.

    Used for live guidance before a capture starts. The frame is decoded in
    memory and discarded; nothing is written to disk.
    """
    try:
        raw = np.frombuffer(await frame.read(), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            return dict(ok=False, message="could not decode frame")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        with _LANDMARKER_LOCK:
            box, detections = stable_box([rgb], landmarker())
        if box is None:
            return dict(ok=True, verdict="NO_FACE", width=width, height=height,
                        notes=["no face detected"])

        verdict, report, notes = framing_verdict(box, width, height)
        return dict(ok=True, verdict=verdict, width=width, height=height,
                    crop_aspect=round(report["expected_aspect"], 3),
                    frac_side_lost=round(report["frac_side_lost"], 3),
                    box=[round(v, 4) for v in box], notes=notes)
    except Exception as exc:
        return dict(ok=False, message=f"{type(exc).__name__}: {exc}")


@rt("/api/analyze", methods=["POST"])
async def api_analyze(video: UploadFile, model: str = None):
    """Analyses one uploaded capture and returns a vital-sign reading.

    The upload is written to a temporary file only because video decoding needs a
    seekable path, and it is removed before this returns under every outcome.
    """
    name = model or registry.DEFAULT_ESTIMATOR
    if name not in registry.available():
        return dict(ok=False, message=f"unknown model {name!r}")

    suffix = Path(video.filename or "capture.mp4").suffix or ".mp4"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(handle.name)
    try:
        handle.write(await video.read())
        handle.close()

        with _LANDMARKER_LOCK:
            clip, fps, quality = crops_from_video(temp_path, landmarker())

        payload = dict(ok=True, model=name, quality=quality_payload(quality),
                       disclaimer=DISCLAIMER)

        if quality.verdict == "REJECT":
            payload.update(status="capture_rejected", value=None)
            return payload
        if len(clip) < config.CLIP_LEN:
            payload.update(status="insufficient_frames", value=None,
                           message=f"{len(clip)} frames, need {config.CLIP_LEN}")
            return payload

        result = estimator(name).estimate(clip, fps)
        detail = result.detail or {}
        value = None if result.value != result.value else round(float(result.value), 2)
        payload.update(
            vital=result.vital, unit=result.unit, value=value,
            confidence=round(float(result.confidence), 3), status=result.status,
            n_windows=detail.get("n_windows"), n_total=detail.get("n_total"),
            usable_fraction=round(float(detail.get("usable_fraction", 0.0)), 3),
            spread_bpm=(round(float(detail["spread_bpm"]), 2)
                        if "spread_bpm" in detail else None),
            window_hr=[round(float(x), 2) for x in detail.get("window_hr", [])],
            window_confidence=[round(float(x), 3)
                               for x in detail.get("window_confidence", [])],
            window_kept=list(detail.get("window_kept", [])),
            waveform=([round(float(x), 4) for x in result.waveform]
                      if result.waveform is not None else []))
        return payload
    except Exception as exc:
        return dict(ok=False, message=f"{type(exc).__name__}: {exc}")
    finally:
        # The capture never outlives the request, whatever happened above.
        try:
            handle.close()
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)

@rt("/")
def index():
    """Placeholder capture page; the real interface follows the UI design pass.

    Model options are rendered server-side: MonsterUI's Select is a web component
    wrapping a hidden native select, so options injected by client script are not
    read back as a value.
    """
    options = [
        fh.Option(f"{name} ({registry.get_estimator(name).vital})",
                  value=name, selected=(name == registry.DEFAULT_ESTIMATOR))
        for name in registry.available()]
    return Titled(
        "CRVSE live vitals",
        Card(
            DivVStacked(
                Video(id="preview", playsinline=True, muted=True, autoplay=True,
                      style="width:100%;max-width:640px;transform:scaleX(-1);"
                            "background:#111;border-radius:6px"),
                DivHStacked(
                    fh.Select(*options, id="model", cls="uk-select",
                              style="max-width:20rem"),
                    fh.Button("start camera", id="start", type="button",
                              cls="uk-btn uk-btn-secondary"),
                    fh.Button("capture 60 s", id="record", type="button",
                              disabled=True, cls="uk-btn uk-btn-primary"),
                    cls="space-x-2"),
                P("idle", id="state", cls=TextPresets.bold_lg),
                Pre("", id="out", cls="text-xs bg-gray-100 p-3 rounded"),
                P(DISCLAIMER, cls=TextPresets.muted_sm))),
        Script(src="/static/capture.js"))
def main():
    """Runs the server, over HTTPS when certificates are available and requested."""
    import uvicorn

    use_https = "--https" in sys.argv
    key = CERT_DIR / "qualityphys-local-key.pem"
    crt = CERT_DIR / "qualityphys-local.pem"

    # uvicorn is given the application object rather than an import string.
    # This module is `app.live_vitals.web.app` inside a package also named
    # `app`, so resolving it by name finds the package instead of this module.
    if use_https and key.exists() and crt.exists():
        announce("https", 8443)
        uvicorn.run(app, host="0.0.0.0", port=8443,
                    ssl_keyfile=str(key), ssl_certfile=str(crt))
    else:
        if use_https:
            print(f"no certificates in {CERT_DIR}; serving HTTP")
        announce("http", 8000)
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()