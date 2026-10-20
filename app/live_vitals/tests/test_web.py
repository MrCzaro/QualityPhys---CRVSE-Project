"""The web server's endpoints, through Starlette's TestClient: no server, no browser."""
import itertools
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import pytest
from starlette.testclient import TestClient

from app.live_vitals import config
from app.live_vitals.capture import session
from app.live_vitals.estimators import registry
from app.live_vitals.tests import synth

MIME = {".avi": "video/x-msvideo", ".mp4": "video/mp4", ".webm": "video/webm"}
WINDOWS_IN_480_FRAMES = len(range(0, 480 - config.CLIP_LEN + 1, config.WINDOW_STRIDE))
# A file of a type FastHTML's catch-all static route served, outside web/static.
OUTSIDE = Path(__file__).resolve().parents[3] / "requirements.txt"


@pytest.fixture(scope="module")
def web():
    from app.live_vitals.web import app as web_app
    return web_app


@pytest.fixture(scope="module")
def client(web):
    return TestClient(web.app)


def analyze(client, path, model="hr_spectral"):
    """Uploads a capture the way the page does, and returns the payload."""
    path = Path(path)
    with open(path, "rb") as handle:
        response = client.post("/api/analyze",
                               files={"video": (f"capture{path.suffix}", handle, MIME[path.suffix])},
                               data={"model": model})
    assert response.status_code == 200
    return response.json()


def jpeg(rgb):
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buffer.tobytes()


def record_captures(web, monkeypatch):
    """Returns the list of temporary files the server creates from now on."""
    made = []
    real = tempfile.NamedTemporaryFile

    def recording(*args, **kwargs):
        handle = real(*args, **kwargs)
        made.append(Path(handle.name))
        return handle

    monkeypatch.setattr(web.tempfile, "NamedTemporaryFile", recording)
    return made


def failing_after(n, real):
    """Wraps real so that its call number n raises, as a read failing midway would."""
    calls = itertools.count()

    def wrapped(*args):
        if next(calls) == n:
            raise RuntimeError("read failed")
        return real(*args)

    return wrapped


# ----------------------------------------------------------- page, static --

def test_the_page_loads_the_versioned_script(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "/static/capture.js?v=" in response.text


def test_the_script_is_served(client):
    response = client.get("/static/capture.js")
    assert response.status_code == 200 and "clearResults" in response.text


@pytest.mark.parametrize("url", [
    "/%2e%2e%2f%2e%2e%2f%2e%2e%2frequirements.txt",
    "/..%2f..%2f..%2frequirements.txt",
    "/static/%2e%2e%2f%2e%2e%2f%2e%2e%2f%2e%2e%2frequirements.txt",
    "/static/..%2f..%2f..%2f..%2frequirements.txt"])
def test_nothing_outside_the_static_folder_is_served(client, url):
    # FastHTML's catch-all static route once served files from anywhere on the drive.
    assert OUTSIDE.exists()
    response = client.get(url)
    assert response.status_code == 404
    assert OUTSIDE.read_bytes()[:60] not in response.content


def test_models_lists_the_registered_estimators(client):
    names = [m["name"] for m in client.get("/api/models").json()["models"]]
    assert names == registry.available()


# ---------------------------------------------------------------- framing --

def post_frame(client, data):
    return client.post("/api/framing",
                       files={"frame": ("frame.jpg", data, "image/jpeg")}).json()


def test_framing_accepts_a_centred_face(client):
    result = post_frame(client, jpeg(synth.Scene().frame(0)))
    assert result["ok"] and result["verdict"] == "ACCEPT"
    assert len(result["box"]) == 4 and result["notes"] == []


def test_framing_without_a_face(client):
    result = post_frame(client, jpeg(synth.Scene(face=False).frame(0)))
    assert result["ok"] and result["verdict"] == "NO_FACE"


def test_framing_flags_a_portrait_frame(client):
    result = post_frame(client, jpeg(synth.Scene(width=480, height=640).frame(0)))
    assert result["verdict"] == "WARN" and "record in landscape" in result["notes"][0]


def test_framing_refuses_bytes_that_are_not_an_image(client):
    assert post_frame(client, b"not an image")["ok"] is False


def test_a_framing_failure_is_logged_once_while_it_lasts(client, web, monkeypatch, caplog):
    """The page asks once a second: a failure that lasts is one log entry, and a failure
    after a check that succeeded is a new one."""
    frame = jpeg(synth.Scene().frame(0))
    real = web.stable_box
    broken = dict(on=True)

    def flaky(*args):
        if broken["on"]:
            raise RuntimeError("detector failed")
        return real(*args)

    monkeypatch.setattr(web, "stable_box", flaky)
    monkeypatch.setattr(web, "_FRAMING_FAILURE", None)
    with caplog.at_level(logging.ERROR, logger="live_vitals.web"):
        for _ in range(3):
            assert post_frame(client, frame) == dict(ok=False,
                                                     message="RuntimeError: detector failed")
        broken["on"] = False
        assert post_frame(client, frame)["verdict"] == "ACCEPT"
        broken["on"] = True
        post_frame(client, frame)
    logged = [r for r in caplog.records if r.name == "live_vitals.web"]
    assert len(logged) == 2 and all(r.exc_info for r in logged)


# ---------------------------------------------------------------- analyze --

def test_a_still_face_is_read(client, video):
    payload = analyze(client, video("still_avi"))
    assert payload["ok"] and payload["status"] == "ok"
    assert payload["value"] == pytest.approx(72.0, abs=2.0)
    quality = payload["quality"]
    assert quality["verdict"] == "ACCEPT" and quality["reasons"] == []
    assert quality["n_frames"] == 480 and quality["detections"] == config.N_DETECT_FRAMES
    assert payload["n_total"] == WINDOWS_IN_480_FRAMES
    assert payload["spectral"]["value"] == pytest.approx(72.0, abs=2.0)


def test_a_browser_webm_is_read(client, video):
    """The Firefox case end to end: declared 1000 fps, read at its real rate."""
    payload = analyze(client, video("browser_vp8"))
    assert payload["ok"] and payload["status"] == "ok"
    assert payload["value"] == pytest.approx(72.0, abs=2.0)
    assert payload["quality"]["declared_fps"] == 1000.0
    assert 29.0 < payload["quality"]["effective_fps"] < 31.0
    assert payload["quality"]["n_frames"] == 480


def test_a_moving_head_is_refused_with_its_reason(client, video):
    payload = analyze(client, video("bob_avi"))
    assert payload["ok"] and payload["status"] == "capture_rejected"
    assert payload["value"] is None and "spectral" not in payload
    reasons = payload["quality"]["reasons"]
    assert len(reasons) == 1 and reasons[0].startswith("the head moved")


def test_a_capture_too_short_to_read(client, video):
    payload = analyze(client, video("short_avi"))
    assert payload["status"] == "insufficient_frames" and payload["value"] is None
    assert payload["message"] == f"100 frames, need {config.CLIP_LEN}"


def test_no_face_is_an_error_not_a_reading(client, video):
    payload = analyze(client, video("noface_avi"))
    # The message names the capture, not the temporary file in the server's user folder.
    assert payload == dict(ok=False, message="RuntimeError: no face detected in the capture")


def test_an_unknown_model_is_refused(client, video):
    payload = analyze(client, video("short_avi"), model="no_such_model")
    assert payload["ok"] is False and "unknown model" in payload["message"]


def test_the_upload_never_outlives_the_request(client, web, video, monkeypatch):
    """The capture is written to a temporary file only so it can be decoded, and that
    file is removed under every outcome, a reading and an error alike."""
    made = record_captures(web, monkeypatch)
    analyze(client, video("still_avi"))
    analyze(client, video("noface_avi"))
    assert len(made) == 2
    assert not any(path.exists() for path in made)
    assert all(path.name.startswith(web.CAPTURE_PREFIX) for path in made)


def test_a_read_that_fails_midway_still_deletes_the_upload(client, web, video, monkeypatch,
                                                            caplog):
    """The decode raises with the file open, and the logged traceback keeps the failing
    frames alive. The file must still be closed and deleted: on Windows an open file
    cannot be."""
    made = record_captures(web, monkeypatch)
    monkeypatch.setattr(session, "crop_resize", failing_after(20, session.crop_resize))
    with caplog.at_level(logging.ERROR, logger="live_vitals.web"):
        payload = analyze(client, video("still_avi"))
    assert payload == dict(ok=False, message="RuntimeError: read failed")
    assert len(made) == 1 and not made[0].exists()


def test_a_failed_analysis_is_logged_with_its_traceback(client, web, video, monkeypatch,
                                                        caplog):
    """The page gets one line; the server's log gets the traceback."""
    def broken(*args, **kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(web, "crops_from_video", broken)
    with caplog.at_level(logging.ERROR, logger="live_vitals.web"):
        payload = analyze(client, video("short_avi"))
    assert payload == dict(ok=False, message="RuntimeError: decoder failed")
    [record] = [r for r in caplog.records if r.name == "live_vitals.web"]
    assert record.levelno == logging.ERROR and record.exc_info is not None
    assert "Traceback" in caplog.text and "decoder failed" in caplog.text


def test_a_capture_longer_than_the_frame_limit_is_refused(client, web, video, monkeypatch):
    # The size limit does not bound a capture's length; this does, before any cropping.
    monkeypatch.setattr(web, "MAX_CAPTURE_FRAMES", 50)
    payload = analyze(client, video("short_avi"))
    assert payload == dict(ok=False, message="ValueError: the capture has more than 50 frames")


@pytest.mark.parametrize("name", ["hr_physnet_v2", "hr_physnet_v3"])
def test_the_neural_estimators_run_end_to_end(client, video, name):
    """Smoke test: the checkpoint loads and every window is analysed. The synthetic face
    is not a test of accuracy; the regression check is."""
    if not registry.get_estimator(name).is_available():
        pytest.skip(f"{name}: checkpoint not present")
    payload = analyze(client, video("still_avi"), model=name)
    assert payload["ok"] and payload["model"] == name
    assert payload["status"] in {"ok", "degraded_capture", "unstable",
                                 "insufficient_quality", "no_estimate"}
    assert payload["n_total"] == WINDOWS_IN_480_FRAMES
    assert len(payload["window_hr"]) == len(payload["window_start"])


# ---------------------------------------------------------- body limits --

@pytest.mark.parametrize("route, field", [("/api/analyze", "video"), ("/api/framing", "frame")])
def test_an_upload_over_its_limit_is_refused_unread(client, web, monkeypatch, route, field):
    """A body that declares its length is refused before any of it is read, so nothing
    is parsed, copied or analysed."""
    monkeypatch.setitem(web.UPLOAD_LIMITS, route, web.MB)
    made = record_captures(web, monkeypatch)
    response = client.post(route, files={field: ("capture.webm", b"\0" * (2 * web.MB),
                                                 "video/webm")})
    assert response.status_code == 413
    assert response.json() == dict(ok=False, message="the upload is over the 1 MB limit")
    assert made == []


@pytest.mark.parametrize("method, route", [("POST", "/api/analyze"), ("GET", "/")])
def test_a_body_sent_in_chunks_is_counted_as_it_arrives(client, web, monkeypatch, method,
                                                        route):
    """Without a declared length the body is counted while it is read, and refused at
    the limit. FastHTML reads any body that is not a form, on every route."""
    monkeypatch.setitem(web.UPLOAD_LIMITS, "/api/analyze", web.MB)
    monkeypatch.setattr(web, "OTHER_BODY_LIMIT", web.MB)

    def chunks():
        for _ in range(4):
            yield b"\0" * (web.MB // 2)

    response = client.request(method, route, content=chunks(),
                              headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 413
    assert response.json() == dict(ok=False, message="the upload is over the 1 MB limit")


def test_an_upload_within_its_limit_is_read(client, web, video, monkeypatch):
    path = Path(video("short_avi"))
    monkeypatch.setitem(web.UPLOAD_LIMITS, "/api/analyze", path.stat().st_size + 4096)
    assert analyze(client, path)["status"] == "insufficient_frames"


def test_only_this_apps_stale_captures_are_swept(web, tmp_path):
    """A capture left by a stopped server is removed at the next start, once it is old
    enough that no other running instance can still be reading it."""
    long_ago = time.time() - web.STALE_CAPTURE_SECONDS - 60
    stale = tmp_path / f"{web.CAPTURE_PREFIX}stale.webm"
    recent = tmp_path / f"{web.CAPTURE_PREFIX}recent.webm"
    other = tmp_path / "another_programs_file.webm"
    for path in (stale, recent, other):
        path.write_bytes(b"\0")
    for path in (stale, other):
        os.utime(path, (long_ago, long_ago))
    assert web.sweep_stale_captures(tmp_path) == 1
    assert not stale.exists() and recent.exists() and other.exists()