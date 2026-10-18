"""Synthetic captures for the test suite: a real face photograph composed into video.

The face is the portrait of Grace Hopper that ships with matplotlib (a US Navy
photograph in the public domain), which MediaPipe detects reliably. Nothing here is
a recording of a person, and every file is written to pytest's temporary directory.

A Scene renders frame i deterministically, so the same capture can be written in
several containers and compared frame for frame. Each writer reproduces one way a
capture reaches the app: OpenCV-written AVI and MP4 with an honest frame count and
rate, WebM written the way browser MediaRecorder writes it, and Y4M for Chromium's
fake camera in the browser tests.
"""
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

# The colour gain of a pulsing face: blood absorbs green most, so the pulse is a
# chroma change rather than a brightness one.
PULSE_GAIN = np.array([0.004, 0.015, 0.007], dtype=np.float32)


def portrait():
    """Returns the bundled portrait as RGB uint8."""
    import matplotlib.cbook as cbook
    path = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class Scene:
    """A face on a plain background, with a pulse and optional head motion.

    The face photograph spans 70% of the frame's shorter side, centred, as in a
    seated capture. Its colour gain follows a PPG-like wave at `bpm`, and the whole
    frame shifts vertically by `bob_px` at `bob_hz`, the way a head moves during
    exercise. Frames are a pure function of their index, so every writer receives
    identical frames.
    """

    def __init__(self, width=640, height=480, fps=30.0, bpm=72.0, bob_px=0.0,
                 bob_hz=1.5, face=True):
        self.width, self.height, self.fps = width, height, float(fps)
        self.bpm, self.bob_px, self.bob_hz = bpm, bob_px, bob_hz
        background = np.full((height, width, 3), 96.0, dtype=np.float32)
        if face:
            image = portrait()
            face_h = int(0.7 * min(height, width))
            face_w = int(face_h * image.shape[1] / image.shape[0])
            resized = cv2.resize(image, (face_w, face_h), interpolation=cv2.INTER_AREA)
            y0, x0 = (height - face_h) // 2, (width - face_w) // 2
            background[y0:y0 + face_h, x0:x0 + face_w] = resized
        self.base = background

    def frame(self, i):
        """Returns frame i as RGB uint8."""
        t = i / self.fps
        f = self.bpm / 60.0
        ppg = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t + 0.8)
        gain = tuple(float(g) for g in 1.0 - PULSE_GAIN * ppg) + (0.0,)
        image = cv2.convertScaleAbs(cv2.multiply(self.base, gain))
        if self.bob_px:
            shift = np.float32([[1, 0, 0],
                                [0, 1, self.bob_px * np.sin(2 * np.pi * self.bob_hz * t)]])
            image = cv2.warpAffine(image, shift, (self.width, self.height),
                                   borderMode=cv2.BORDER_REFLECT)
        return image

    def frames(self, n):
        for i in range(n):
            yield self.frame(i)


def write_opencv(path, scene, n, fourcc="MJPG"):
    """Writes n frames with OpenCV; the container declares the scene's rate and count."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), scene.fps,
                             (scene.width, scene.height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write {fourcc} to {Path(path).suffix}")
    for rgb in scene.frames(n):
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return Path(path)


def write_browser_webm(path, scene, n, codec="libvpx", jitter_ms=6.0, seed=0):
    """Writes n frames as WebM the way browser MediaRecorder writes it.

    Streamed (no seek index, no duration), a 1 ms timebase, and frame times that
    wander a few ms around 1/fps as a camera's do. OpenCV then reports the timebase
    as a 1000 fps rate and a frame count derived from it, as it did for Firefox's
    captures; a capture path that seeks, or trusts that rate, fails on this file.
    """
    import av  # PyAV: a test-only dependency
    rng = np.random.default_rng(seed)
    options = {"deadline": "realtime", "cpu-used": "8"} if codec == "libvpx-vp9" else {}
    with av.open(str(path), "w", format="webm", options={"live": "1"}) as container:
        stream = container.add_stream(codec, rate=1000, options=options)
        stream.width, stream.height, stream.pix_fmt = scene.width, scene.height, "yuv420p"
        stream.bit_rate = 4_000_000
        stream.time_base = Fraction(1, 1000)
        for i, rgb in enumerate(scene.frames(n)):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            wander = rng.uniform(-jitter_ms, jitter_ms) if i else 0.0
            frame.pts = int(round(1000.0 * i / scene.fps + wander))
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return Path(path)


def write_y4m(path, scene, n):
    """Writes n frames as Y4M (4:2:0), the format Chromium's fake camera plays in a loop."""
    with open(path, "wb") as handle:
        handle.write(f"YUV4MPEG2 W{scene.width} H{scene.height} F{int(scene.fps)}:1 "
                     f"Ip A1:1 C420jpeg XYSCSS=420JPEG\n".encode("ascii"))
        for rgb in scene.frames(n):
            handle.write(b"FRAME\n")
            handle.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420).tobytes())
    return Path(path)


def sequential_frames(path):
    """Every frame of a plain front-to-back decode, as RGB, with its timestamp in ms."""
    cap = cv2.VideoCapture(str(path))
    frames, stamps = [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    return frames, stamps