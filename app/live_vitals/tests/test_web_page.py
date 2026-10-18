"""The capture page (capture.js) in headless Chromium, with a synthetic face as the camera.

The app is served on a free local port, and Chromium's fake camera plays a Y4M file of
the synthetic face in a loop, so the page records, uploads and renders exactly as it
does for a person. Captures are shortened by setting SETTINGS.seconds in the page.

Needs Playwright and its Chromium (pip install playwright; python -m playwright install
chromium). Every test here carries the `browser` mark: `-m "not browser"` leaves them out
of a quick run, `-m browser` runs them alone.
"""
import json
import re
import socket
import threading
import time

import pytest

from app.live_vitals.tests import synth

pytestmark = pytest.mark.browser
sync_api = pytest.importorskip("playwright.sync_api",
                               reason="pip install playwright; python -m playwright install chromium")

READING_SECONDS = 16   # five analysis windows, enough for a reading
SHORT_SECONDS = 6      # for captures that are meant to fail
DASH = "\u2014"       # what an empty panel shows

# Exposes each MediaRecorder the page builds, so a test can fire its error event.
EXPOSE_RECORDER = """
(() => {
  const Real = window.MediaRecorder;
  if (!Real) return;
  class Exposed extends Real {
    constructor(...args) { super(...args); (window.__recorders ||= []).push(this); }
  }
  Exposed.isTypeSupported = Real.isTypeSupported.bind(Real);
  window.MediaRecorder = Exposed;
})();
"""

SNAPSHOT = """() => {
  const text = id => document.getElementById(id).textContent;
  return {
    hero: text('model-hr'), hero_note: text('model-hr-note'), model_chip: text('model-status'),
    spec_chip: text('spec-status'), spec_hr: text('spec-hr'), conf: text('model-conf-text'),
    state: text('state'), start_disabled: document.getElementById('start').disabled,
    hud: document.getElementById('rec-hud').style.display,
    stage: document.getElementById('stage').style.display,
  };
}"""


@pytest.fixture(scope="module")
def base_url():
    """The app, served by uvicorn on a free port in a background thread."""
    import uvicorn
    from app.live_vitals.web import app as web
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(web.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("the app did not start")
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def cameras(tmp_path_factory):
    """Ten-second loops for the fake camera: a whole number of pulse and motion cycles."""
    folder = tmp_path_factory.mktemp("cameras")
    return {"still": synth.write_y4m(folder / "still.y4m", synth.Scene(), 300),
            "moving": synth.write_y4m(folder / "moving.y4m",
                                      synth.Scene(bob_px=14, bob_hz=1.5), 300)}


@pytest.fixture(scope="module")
def playwright():
    with sync_api.sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="module")
def browsers(playwright, cameras):
    """One Chromium per fake camera, launched on first use."""
    launched = {}

    def get(camera):
        if camera not in launched:
            try:
                launched[camera] = playwright.chromium.launch(args=[
                    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                    f"--use-file-for-fake-video-capture={cameras[camera]}"])
            except Exception as exc:
                pytest.skip(f"Chromium for Playwright is missing ({type(exc).__name__}); "
                            f"run: python -m playwright install chromium")
        return launched[camera]

    yield get
    for browser in launched.values():
        browser.close()


class Page:
    """One page with the camera started, and a record of its requests and script errors."""

    def __init__(self, browser, base_url, model="hr_spectral"):
        self.context = browser.new_context()
        self.page = self.context.new_page()
        self.requests, self.script_errors = [], []
        self.page.on("request", lambda r: self.requests.append((time.time(), r.url)))
        # Errors raised in capture.js; the page's CDN libraries are not under test.
        self.page.on("pageerror", lambda e: "capture.js" in (getattr(e, "stack", "") or "")
                     and self.script_errors.append(str(e)))
        self.page.add_init_script(EXPOSE_RECORDER)
        self.page.goto(base_url)
        self.page.click("#start")
        self.page.wait_for_function(
            "document.getElementById('framing-chip').textContent.startsWith('ACCEPT')",
            timeout=30000)
        self.page.wait_for_function("!document.getElementById('record').disabled",
                                    timeout=10000)
        self.page.select_option("#model", model)

    def seconds(self, n):
        self.page.evaluate(f"SETTINGS.seconds = {n}")

    def begin(self):
        self.page.click("#record")
        self.page.wait_for_function("capturing === true", timeout=5000)

    def finish(self, timeout=180000):
        self.page.wait_for_function("capturing === false", timeout=timeout)
        self.wait(300)
        return self.snap()

    def wait(self, ms):
        self.page.wait_for_timeout(ms)   # lets request and error events through

    def snap(self):
        return self.page.evaluate(SNAPSHOT)

    def uploads(self):
        return sum(1 for _, url in self.requests if url.endswith("/api/analyze"))

    def framing_polls_per_second(self, seconds=4):
        start = time.time()
        self.wait(seconds * 1000)
        return sum(1 for t, url in self.requests
                   if t >= start and url.endswith("/api/framing")) / seconds

    def close(self):
        self.context.close()


@pytest.fixture
def page(browsers, base_url):
    pages = []

    def open_page(camera="still", model="hr_spectral"):
        pages.append(Page(browsers(camera), base_url, model))
        return pages[-1]

    yield open_page
    errors = [error for p in pages for error in p.script_errors]
    for p in pages:
        p.close()
    assert errors == [], errors


def test_a_capture_is_read_and_the_next_one_starts_clean(page):
    p = page("still")
    p.seconds(READING_SECONDS)
    p.begin()
    during = p.snap()
    assert during["hero"] == DASH and during["hero_note"] == "Waiting for this capture."
    assert during["start_disabled"] and during["model_chip"] == "capturing"
    after = p.finish()
    assert after["state"].startswith("Reading complete")
    assert float(after["hero"]) == pytest.approx(72.0, abs=2.0)
    assert not after["start_disabled"] and after["hud"] == "none" and after["stage"] == "none"
    assert p.uploads() == 1

    # The next capture takes the last reading off the screen as it starts.
    p.begin()
    restart = p.snap()
    assert restart["hero"] == DASH and restart["spec_hr"] == DASH
    assert restart["hero_note"] == "Waiting for this capture."
    assert restart["state"].startswith("get ready")


def test_a_lost_camera_discards_the_recording(page):
    p = page("still")
    p.seconds(SHORT_SECONDS)
    p.begin()
    p.wait((3 + 2) * 1000)               # the countdown, then two seconds of recording
    p.page.evaluate("stream.getVideoTracks()[0].stop()")
    p.wait(1500)
    stopped = p.snap()
    assert re.match(r"Recording stopped after \d+ s because the camera stopped delivering",
                    stopped["state"])
    assert stopped["hero_note"] == "No reading for this capture."
    assert stopped["hud"] == "none" and not stopped["start_disabled"]
    p.wait(SHORT_SECONDS * 1000)         # past where the countdown would have ended
    assert p.snap()["state"] == stopped["state"]
    assert p.uploads() == 0
    # One framing poll a second at most: a leaked second timer would double it. (With
    # the camera gone the poll may find no frame to send, so there is no lower bound.)
    assert p.framing_polls_per_second() <= 1.5


def test_a_recorder_error_discards_the_recording(page):
    p = page("still")
    p.seconds(SHORT_SECONDS)
    p.begin()
    p.wait((3 + 2) * 1000)
    p.page.evaluate("window.__recorders.at(-1).dispatchEvent(new Event('error'))")
    p.wait(1500)
    failed = p.snap()
    assert failed["state"] == "Recording failed: unknown recorder error"
    assert failed["hero_note"] == "No reading for this capture."
    p.wait(SHORT_SECONDS * 1000)
    assert p.snap()["state"] == failed["state"]
    assert p.uploads() == 0
    assert 0.5 <= p.framing_polls_per_second() <= 1.5


def test_a_server_error_and_a_lost_connection_leave_no_reading(page):
    p = page("still")
    p.seconds(SHORT_SECONDS)
    p.page.route("**/api/analyze", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"ok": False, "message": "RuntimeError: test failure"})))
    p.begin()
    error = p.finish()
    assert error["state"] == "error: RuntimeError: test failure"
    assert error["hero"] == DASH and error["hero_note"] == "No reading for this capture."
    assert error["model_chip"] == "error"

    p.page.unroute("**/api/analyze")
    p.page.route("**/api/analyze", lambda route: route.abort())
    p.begin()
    lost = p.finish()
    assert lost["state"].startswith("analysis failed")
    assert lost["hero_note"] == "No reading for this capture."


def test_a_refused_capture_says_why(page):
    p = page("moving")
    p.seconds(READING_SECONDS)
    p.begin()
    refused = p.finish()
    assert refused["state"].startswith("No reading: the head moved")
    assert "hold still" in refused["state"]
    assert refused["hero"] == DASH and refused["spec_chip"] == "not run"
    assert refused["conf"] == DASH