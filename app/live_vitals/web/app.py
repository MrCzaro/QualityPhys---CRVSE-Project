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
SPECTRAL_NAME = "hr_spectral"
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
    hdrs=(*Theme.blue.headers(apex_charts=True),
          Link(rel="icon", type="image/x-icon", href="/static/favicon.ico")),
    live=False, static_path=str(Path(__file__).parent))

# These are deliberately MonsterUI/Tailwind utility classes, not a second CSS
# system.  They make the dashboard feel like one clinical instrument in both
# the light and dark variants that Theme.blue supplies.
PANEL = (CardT.default, "rounded-2xl border border-border shadow-sm")

def chip(chip_id, text="—", tone=LabelT.secondary):
    """Renders a status pill that the client recolours in place."""
    return Label(text, id=chip_id, cls=tone)


def card_heading(title, subtitle, chip_element):
    """Card header: stacked title and subtitle on the left, a status pill right.

    A plain Div is used rather than DivVStacked because the latter centres its
    children, and a centred title reads as misaligned beside a left-aligned body.
    """
    return DivFullySpaced(
        Div(CardTitle(title), Subtitle(subtitle)),
        chip_element)


def vital_tile(title, value_id, unit, footnote, icon, chip_element=None):
    """A vitals tile: title and status on top, a large value, a muted footnote.

    The value starts muted; the client clears that class when a real reading
    arrives, so an empty tile never reads as a rendering fault.
    """
    return Card(
        DivFullySpaced(
            DivLAligned(
                Div(UkIcon(icon, cls="h-5 w-5"),
                    cls="rounded-xl bg-primary/10 p-2 text-primary"),
                H4(title, cls="text-base font-semibold"),
                cls="space-x-3"),
            chip_element or Label("pending", cls=LabelT.secondary)),
        DivLAligned(
            Span("-", id=value_id,
                 cls="text-5xl font-semibold tracking-tight tabular-nums text-muted-foreground"),
            Span(unit, cls=TextPresets.muted_sm),
            cls="items-baseline space-x-2"),
        P(footnote, id=f"{value_id}-note", cls=(TextPresets.muted_sm, "pt-1")),
        cls=PANEL)


def confidence(prefix):
    """A confidence bar with a numeric caption, driven by the client."""
    return Div(
        DivFullySpaced(P("confidence", cls=TextPresets.muted_sm),
                       P("—", id=f"{prefix}-conf-text", cls=TextPresets.muted_sm)),
        Progress(value="0", max="100", id=f"{prefix}-conf", cls="w-full"),
        cls="pt-2 space-y-1")

def cross_check_payload(result):
    """Serialises the spectral cross-check, or None when it was not run."""
    if result is None:
        return None
    detail = result.detail or {}
    value = None if result.value != result.value else round(float(result.value), 2)
    # Each method keeps its own window count. A method that agrees while surviving
    # five windows is not the same evidence as one that agrees over sixty, and the
    # bare number hides that difference.
    methods = {name: dict(hr=(None if row["hr_bpm"] != row["hr_bpm"]
                              else round(float(row["hr_bpm"]), 2)),
                          n_windows=row.get("n_windows"),
                          n_total=row.get("n_total"))
               for name, row in (detail.get("method_hr") or {}).items()}
    return dict(value=value, unit=result.unit, status=result.status,
                confidence=round(float(result.confidence), 3),
                method=detail.get("method"), method_hr=methods,
                n_windows=detail.get("n_windows"), n_total=detail.get("n_total"))



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
        
        # The classical estimator shares no weights and no training data with a
        # neural one, so where the two agree that is corroboration rather than a
        # model confirming itself. When it is the selected model it is reported
        # as its own cross-check, which surfaces the POS/CHROM/GREEN spread.
        if name == SPECTRAL_NAME:
            cross = result
        elif SPECTRAL_NAME in registry.available():
            cross = estimator(SPECTRAL_NAME).estimate(clip, fps)
        else:
            cross = None

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
            n_no_peak=detail.get("n_no_peak", 0),
            spectral=cross_check_payload(cross),
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
    """Main capture and results view, with a collapsible diagnostics panel."""
    options = [
        fh.Option(f"{name} ({registry.get_estimator(name).vital})",
                  value=name, selected=(name == registry.DEFAULT_ESTIMATOR))
        for name in registry.available()]
    spectral_installed = SPECTRAL_NAME in registry.available()

    # A vitals row gives the readings at a glance; the two unavailable vitals are
    # shown rather than hidden so the intended scope of the app is visible.
    vitals = Grid(
        vital_tile("Heart rate", "model-hr", "bpm", "awaiting capture",
                   "heart-pulse", chip("model-status", "idle")),
        vital_tile("HRV", "model-hrv", "ms", "under evaluation", "activity"),
        vital_tile("Respiratory rate", "model-rr", "br/min", "requires model",
                   "wind"),
        cols_md=3, cols_sm=1, cls="gap-5")

    capture = Card(
        Video(id="preview", playsinline=True, muted=True, autoplay=True,
              cls="w-full rounded-xl border border-border bg-black block",
              style="transform:scaleX(-1)"),
        Div(
            DivFullySpaced(
                DivLAligned(chip("framing-chip", "no camera"),
                            P("-", id="fps-chip", cls=TextPresets.muted_sm),
                            cls="space-x-3"),
                fh.Select(*options, id="model", cls="uk-select",
                          style="max-width:14rem")),
            Div(
                Button(UkIcon("video", cls="mr-2 h-4 w-4"), "Start camera",
                       id="start", type="button",
                       cls=(ButtonT.primary, ButtonT.lg,
                            "flex-1 rounded-full px-5 font-semibold whitespace-nowrap")),
                Button(UkIcon("circle", cls="mr-2 h-3.5 w-3.5"), "Capture 60 s",
                       id="record", type="button", disabled=True,
                       cls=(ButtonT.default, ButtonT.lg,
                            "flex-1 rounded-full border border-primary/60 px-5 text-primary "
                            "font-semibold whitespace-nowrap disabled:opacity-60")),
                cls="flex items-center gap-2"),
            cls="space-y-3 border-t border-border pt-4"),
        P("Press start, wait for the framing chip to read ACCEPT, then capture.",
          id="state", cls=(TextPresets.muted_sm, "pt-1")),
        header=card_heading(
            "Camera capture",
            "face a window or lamp — not with one behind you",
            chip("privacy-chip", "analysed, never stored", LabelT.primary)),
        cls=(*PANEL, "col-span-4"))

    detail = Card(
        Div(confidence("model"),
            DividerLine(),
            DivFullySpaced(P("windows used", cls=TextPresets.muted_sm),
                           P("—", id="stat-windows", cls=(TextT.sm, "tabular-nums"))),
            DivFullySpaced(P("spread", cls=TextPresets.muted_sm),
                           P("—", id="stat-spread", cls=(TextT.sm, "tabular-nums"))),
            DivFullySpaced(P("capture rate", cls=TextPresets.muted_sm),
                           P("—", id="stat-rate", cls=(TextT.sm, "tabular-nums"))),
            cls="space-y-3 rounded-xl bg-muted/30 p-4"),
        DividerLine(),
        DivFullySpaced(Div(Strong("Spectral", cls=TextT.sm),
                           P("classical cross-check", cls=TextPresets.muted_sm)),
                       chip("spec-status", "not installed")),
        DivLAligned(Span("—", id="spec-hr",
                         cls="text-2xl tabular-nums text-muted-foreground"),
                    Span("bpm", cls=TextPresets.muted_sm),
                    cls="items-baseline space-x-2"),
        None if spectral_installed else
        P("Not yet ported from the Phase-2 app, so there is no independent "
          "estimate to compare against.", cls=TextPresets.muted_sm),
        header=card_heading("Measurement quality", "how far to trust this reading",
                            chip("quality-chip", "idle")),
        cls=(*PANEL, "col-span-3"))

    # Every panel the client fills carries a placeholder, so an app that has not
    # measured anything yet reads as waiting rather than as broken.
    waiting = lambda: P("No capture yet.", cls=TextPresets.muted_sm)

    trend = Card(
        Div(waiting(), id="trend", cls="w-full"),
        P("Per-window heart rate. Hollow points were discarded by the quality "
          "gates; the dashed line is the reported median.",
          cls=TextPresets.muted_sm),
        header=card_heading("Trend", "stability across the capture",
                            chip("trend-chip", "idle")),
        cls=PANEL)

    diagnostics = Card(
        Grid(Div(H4("Capture quality"),
                 Div(waiting(), id="diag-quality", cls="font-mono text-xs pt-1")),
             Div(H4("Gates"),
                 Div(waiting(), id="diag-gates", cls="font-mono text-xs pt-1")),
             cols_md=2, cols_sm=1, cls="gap-8"),
        DividerLine(),
        H4("Windows"),
        Div(waiting(), id="diag-windows", cls="overflow-x-auto pt-1"),
        DividerLine(),
        H4("Spectral cross-check"),
        P("POS and CHROM read the same colour trace through different projections, "
          "so their agreeing says the trace itself is sound and their diverging "
          "says the cross-check should not be trusted on this capture, however "
          "close either one lands to the model. GREEN carries no specular "
          "rejection and is shown for reference only.", cls=TextPresets.muted_sm),
        Div(waiting(), id="diag-spectral", cls="overflow-x-auto pt-1"),
        DividerLine(),
        H4("Reconstructed BVP waveform"),
        P("Drawn at about 100 px per second so individual beats are legible — "
          "scroll sideways to read the whole strip. Windows are inferred "
          "independently and concatenated; pale red stretches were rejected by "
          "the quality gates.", id="diag-wave-note",
          cls=TextPresets.muted_sm),
        Div(waiting(), id="diag-wave", cls="w-full pt-2 overflow-x-auto"),
        header=card_heading("Diagnostics", "every number behind the reading",
                            chip("diag-chip", "idle")),
        cls=PANEL)

    # `cols` is omitted deliberately: passing it propagates to every breakpoint
    # and defeats the per-breakpoint values that make col-span layouts work.
    return Container(
        Title("CRVSE live vitals"),
        NavBar(chip("build-chip", "research only · not a medical device",
                    LabelT.secondary),
               brand=Div(H3("CRVSE live vitals"),
                         Subtitle("camera heart-rate research demo")),
               cls="rounded-2xl border border-border bg-card/80 px-3 shadow-sm"),
        Div(vitals,
            Grid(capture, detail, cols_xl=7, cols_lg=7, cols_md=1, cols_sm=1),
            trend,
            Accordion(AccordionItem("Diagnostics", diagnostics, open=False),
                      multiple=False),
            cls="space-y-6 pt-2"),
        P(DISCLAIMER, cls=(TextPresets.muted_sm, "pt-8 pb-4")),
        Script(src="/static/capture.js?v=20260901-4"),
        cls=("space-y-4", ContainerT.xl))

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
