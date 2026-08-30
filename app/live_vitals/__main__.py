"""Command-line entry point for the CRVSE live_vitals research demo.

Usage:
    python -m app.live_vitals models
    python -m app.live_vitals analyze <video> [--model NAME]
    python -m app.live_vitals camera [--seconds 60] [--model NAME] [--device 0]
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.capture.camera import Camera
from app.live_vitals.capture.session import crops_from_video, crops_from_camera
from app.live_vitals.estimators import registry

DISCLAIMER = ("Research demo, not a medical device. Not validated for diagnosis "
              "or treatment decisions.")


def confidence_bar(value, width=10):
    """Renders a confidence value as a fixed-width text bar."""
    filled = 0 if value != value else int(round(max(0.0, min(1.0, value)) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def report(result, quality, model_name, source):
    """Prints one reading with its acquisition diagnostics."""
    print(f"\nmodel      : {model_name}")
    print(f"source     : {source}")
    print(f"format     : {quality.width}x{quality.height} @ "
          f"{quality.effective_fps:.2f} fps  (crop aspect {quality.crop_aspect:.3f})")
    print(f"detections : {quality.detections}/{quality.detections_attempted} "
          f"sampled frames")
    print(f"framing    : {quality.verdict}")
    for note in quality.notes:
        print(f"             - {note}")

    if result is None:
        print("\n  no reading: capture rejected")
        print(f"\n{DISCLAIMER}")
        return 1

    detail = result.detail or {}
    used = detail.get("n_windows", 0)
    total = detail.get("n_total", 0)
    print(f"windows    : {used}/{total} usable "
          f"({detail.get('usable_fraction', float('nan')):.2f})")
    print(f"\n  {result.vital.replace('_', ' ')}   {result.value:.2f} {result.unit}")
    print(f"  confidence   {result.confidence:.2f} {confidence_bar(result.confidence)}")
    print(f"  status       {result.status}")
    if quality.verdict == "WARN":
        print("  note         capture is off-distribution; treat as indicative")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_models(args):
    """Lists the registered estimators and whether their weights are present."""
    for name in registry.available():
        estimator = registry.get_estimator(name)
        state = "ready" if estimator.is_available() else "checkpoint missing"
        default = " (default)" if name == registry.DEFAULT_ESTIMATOR else ""
        print(f"{name:24s} {estimator.vital:12s} {state}{default}")
    return 0


def cmd_analyze(args):
    """Estimates a vital from a recorded video."""
    estimator = registry.get_estimator(args.model)
    if not estimator.is_available():
        print(f"ERROR: checkpoint missing for {args.model}")
        return 2
    clip, fps, quality = crops_from_video(args.video)
    if quality.verdict == "REJECT" and not args.force:
        return report(None, quality, args.model, args.video)
    if len(clip) < config.CLIP_LEN:
        print(f"ERROR: {len(clip)} frames, need at least {config.CLIP_LEN}")
        return 1
    return report(estimator.estimate(clip, fps), quality, args.model, args.video)


def cmd_camera(args):
    """Captures from the webcam for a fixed duration, then estimates."""
    estimator = registry.get_estimator(args.model)
    if not estimator.is_available():
        print(f"ERROR: checkpoint missing for {args.model}")
        return 2

    def progress(elapsed, total):
        bar = int(round(elapsed / total * 30))
        print(f"\r  capturing {elapsed:5.1f}/{total:.0f}s "
              f"[{'#' * bar}{'-' * (30 - bar)}]", end="", flush=True)

    print(f"Sit still, facing the camera, head filling about a third of the frame.")
    with Camera(index=args.device) as camera:
        clip, fps, quality = crops_from_camera(camera, args.seconds,
                                               on_progress=progress)
    print()
    if quality.verdict == "REJECT" and not args.force:
        return report(None, quality, args.model, f"camera {args.device}")
    if len(clip) < config.CLIP_LEN:
        print(f"ERROR: {len(clip)} frames, need at least {config.CLIP_LEN}")
        return 1
    return report(estimator.estimate(clip, fps), quality, args.model,
                  f"camera {args.device}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m app.live_vitals", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="list registered estimators")

    p_analyze = sub.add_parser("analyze", help="estimate from a recorded video")
    p_analyze.add_argument("video")
    p_analyze.add_argument("--model", default=registry.DEFAULT_ESTIMATOR,
                           choices=registry.available())
    p_analyze.add_argument("--force", action="store_true",
                           help="report a reading even when framing is rejected")

    p_camera = sub.add_parser("camera", help="capture from a webcam and estimate")
    p_camera.add_argument("--seconds", type=float, default=60.0)
    p_camera.add_argument("--device", type=int, default=0)
    p_camera.add_argument("--model", default=registry.DEFAULT_ESTIMATOR,
                          choices=registry.available())
    p_camera.add_argument("--force", action="store_true")

    args = parser.parse_args()
    return {"models": cmd_models, "analyze": cmd_analyze,
            "camera": cmd_camera}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())