"""The web server's endpoints, through Starlette's TestClient: no server, no browser."""
import tempfile
from pathlib import Path

import cv2
import pytest
from starlette.testclient import TestClient

from app.live_vitals import config
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
    assert payload["ok"] is False and "no face detected" in payload["message"]


def test_an_unknown_model_is_refused(client, video):
    payload = analyze(client, video("short_avi"), model="no_such_model")
    assert payload["ok"] is False and "unknown model" in payload["message"]


def test_the_upload_never_outlives_the_request(client, web, video, monkeypatch):
    """The capture is written to a temporary file only so it can be decoded, and that
    file is removed under every outcome, a reading and an error alike."""
    made = []
    real = tempfile.NamedTemporaryFile

    def recording(*args, **kwargs):
        handle = real(*args, **kwargs)
        made.append(Path(handle.name))
        return handle

    monkeypatch.setattr(web.tempfile, "NamedTemporaryFile", recording)
    analyze(client, video("still_avi"))
    analyze(client, video("noface_avi"))
    assert len(made) == 2
    assert not any(path.exists() for path in made)


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