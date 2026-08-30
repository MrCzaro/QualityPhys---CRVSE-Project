"""Webcam capture for the live_vitals app.

Frames are converted, cropped and discarded as they arrive: no full frame is
retained beyond the current one and nothing is written to disk, matching the
project's no-frame-storage rule.
"""
import sys
import time

import cv2


class Camera:
    """Context manager around an OpenCV capture device."""

    def __init__(self, index=0, width=1280, height=720, fps=30.0):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None

    def __enter__(self):
        # Media Foundation negotiates the full frame rate on Windows. DirectShow
        # capped this machine's camera at 10 fps at 720p regardless of the pixel
        # format requested (measured 2026-08-29), which silently violates the
        # model's 30 fps contract. DirectShow is kept only as a fallback.
        if sys.platform == "win32":
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
        else:
            backends = [cv2.CAP_ANY]
        cap = None
        for backend in backends:
            candidate = cv2.VideoCapture(self.index, backend)
            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()
        if cap is None:
            raise RuntimeError(f"could not open camera {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        return self

    def __exit__(self, *exc_info):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        return False

    @property
    def frame_size(self):
        """Returns the (width, height) the device actually delivers."""
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        """Returns (ok, rgb_frame, timestamp_seconds)."""
        ok, bgr = self._cap.read()
        if not ok:
            return False, None, None
        return True, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), time.perf_counter()

    def warm_up(self, frames=15, measure=20):
        """Discards initial frames so exposure settles, then measures the rate.

        Returns the delivered frames per second, which is what the model contract
        depends on — the rate a device reports is not reliable.
        """
        for _ in range(frames):
            self._cap.read()
        start = time.perf_counter()
        got = sum(1 for _ in range(measure) if self._cap.read()[0])
        elapsed = time.perf_counter() - start
        return got / elapsed if elapsed > 0 else float("nan")