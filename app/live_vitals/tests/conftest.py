"""Shared fixtures: synthetic videos written once per session, and one face detector.

Videos are written to pytest's temporary directory on first use and reused by every
test in the session; nothing is written inside the repository. Results of the capture
path are cached the same way, because several tests read the same capture.
"""
import sys

import pytest

from app.live_vitals.tests import synth

# name: (writer, Scene arguments, frame count, writer arguments)
VIDEOS = {
    "still_avi": ("opencv", {}, 480, {}),
    "still_mp4": ("opencv", {}, 480, {"fourcc": "mp4v"}),
    "fps60_avi": ("opencv", {"fps": 60}, 960, {}),
    "fps24_avi": ("opencv", {"fps": 24}, 384, {}),
    "small_motion_avi": ("opencv", {"bob_px": 2, "bob_hz": 1.2}, 360, {}),
    "rhythm_avi": ("opencv", {"bob_px": 5, "bob_hz": 1.2}, 360, {}),
    "bob_avi": ("opencv", {"bob_px": 14, "bob_hz": 1.5}, 360, {}),
    "portrait_bob_avi": ("opencv", {"width": 480, "height": 640, "bob_px": 14,
                                    "bob_hz": 1.5}, 360, {}),
    "noface_avi": ("opencv", {"face": False}, 120, {}),
    "short_avi": ("opencv", {}, 100, {}),
    "browser_vp8": ("webm", {}, 480, {}),
    "browser_vp9": ("webm", {}, 360, {"codec": "libvpx-vp9"}),
}


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    """Returns a function giving the path of a named synthetic video, writing it once."""
    folder = tmp_path_factory.mktemp("videos")
    made = {}

    def get(name):
        if name not in made:
            kind, scene_args, n, writer_args = VIDEOS[name]
            scene = synth.Scene(**scene_args)
            if kind == "webm":
                pytest.importorskip("av", reason="PyAV writes the browser-style WebM")
                made[name] = synth.write_browser_webm(folder / f"{name}.webm", scene, n,
                                                      **writer_args)
            else:
                suffix = ".mp4" if writer_args.get("fourcc") == "mp4v" else ".avi"
                made[name] = synth.write_opencv(folder / f"{name}{suffix}", scene, n,
                                                **writer_args)
        return made[name]

    return get


@pytest.fixture(scope="session")
def frame_count():
    """The number of frames each named video was written with."""
    return lambda name: VIDEOS[name][2]


@pytest.fixture(scope="session")
def landmarker():
    from app.live_vitals.preprocess.face_box import make_landmarker
    detector = make_landmarker()
    yield detector
    # Closed here rather than left to the interpreter's exit, where MediaPipe's own
    # cleanup can hang once a failed test has kept a reference to the detector.
    detector.close()


@pytest.fixture(scope="session")
def captured(video, landmarker):
    """Returns a function giving crops_from_video's (clip, fps, quality) for a named video."""
    from app.live_vitals.capture.session import crops_from_video
    results = {}

    def get(name):
        if name not in results:
            results[name] = crops_from_video(video(name), landmarker)
        return results[name]

    return get


@pytest.fixture(scope="session", autouse=True)
def close_the_app_detector():
    """Closes the web app's own detector, if a test created it, for the same reason."""
    yield
    web = sys.modules.get("app.live_vitals.web.app")
    if web is not None and web._LANDMARKER is not None:
        web._LANDMARKER.close()
        web._LANDMARKER = None