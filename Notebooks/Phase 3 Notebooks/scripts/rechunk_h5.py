#!/usr/bin/env python3
"""Rewrites Phase-3 rPPG HDF5 stores with a clip-friendly chunk layout.

Why
---
Four of the five Phase-3 stores were first written with h5py auto-chunking, which
produced chunks long in time and tiny in space, for example (1135, 9, 9, 1) for a
72x72x3 frame. Reading one 160-frame clip then touched up to 384 chunks and
decompressed 14x more bytes than the clip needed. Measured on the ZBook:
PhysDrive served 10.5 clips/s against UBFC-Phys at 23.2 clips/s under identical
settings, with read amplification of 14.2x against 1.2x (Pearson r = -0.93 across
the five datasets).

The script rewrites every store with frame chunks shaped (FRAMES_PER_CHUNK, H, W, C),
so a clip needs ceil(clip_len / FRAMES_PER_CHUNK) chunk reads and no wasted bytes.

The Phase-3 corpus was rewritten on 2026-09-12 without compression, producing the
`*_phase3_rc_none.h5` stores. Reading one clip fell from 98.9 ms (gzip:1, auto
chunks) to 12.7 ms (lzf) and 3.1 ms (uncompressed); cold and warm reads matched, so
the cost had been decompression rather than I/O. In the full training pipeline this
step took PhysNet from 6.34 to 8.83 steps/s and the pipeline's share of step time
from 30% to 3%. Those settings are the defaults below.

Safety
------
- Never writes in place. Output goes to <stem><suffix>.h5 next to the source, via a
  temporary .h5.part file that is renamed only on completion.
- Runs as an inspection by default: it prints the plan and changes nothing until
  `--apply` is given.
- Layout-agnostic: walks the file and copies every group, dataset and attribute,
  without assuming dataset names.
- Skips any store that already carries the `rechunked_by` marker, so a store is
  never rechunked twice.
- `--verify` reopens the result and compares shapes, dtypes, attributes and randomly
  chosen data blocks against the source.

Usage
-----
  python rechunk_h5.py <data_dir>                                  # inspect only
  python rechunk_h5.py <data_dir> --only physdrive --apply         # one store
  python rechunk_h5.py <data_dir> --apply --verify                 # whole corpus
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------- parameters --

DEFAULT_FRAMES_PER_CHUNK = 32     # the layout of ubfc_phys_phase3.h5, the one fast store
DEFAULT_CLIP_LEN = 160            # used only to report read amplification
SIG_CHUNK = 32768                 # elements per chunk for large 1-D reference signals
COPY_BLOCK_FRAMES = 512           # frames copied per read/write; bounds RAM use


def human(n: float) -> str:
    """Formats a byte count with a binary unit suffix."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:6.1f} {unit}"
        n /= 1024.0
    return f"{n:6.1f} PB"


def parse_compression(spec: str):
    """Parses 'none' | 'lzf' | 'gzip' | 'gzip:N' into (compression, compression_opts)."""
    spec = (spec or "none").strip().lower()
    if spec in ("none", "off", ""):
        return None, None
    if spec == "lzf":
        return "lzf", None
    if spec.startswith("gzip"):
        level = 4
        if ":" in spec:
            level = int(spec.split(":", 1)[1])
        return "gzip", level
    raise SystemExit(f"unknown --compression value: {spec!r}")


# ------------------------------------------------------------ chunk planning --

def looks_like_frames(ds: h5py.Dataset) -> bool:
    """Reports whether a dataset is a frame stack: >=3 dims, time first, uint8 pixels."""
    return ds.ndim >= 3 and ds.dtype == np.uint8 and ds.shape[0] > 1


def plan_chunks(ds: h5py.Dataset, frames_per_chunk: int):
    """Returns (new_chunks, reason); new_chunks of None means contiguous storage."""
    if ds.shape == () or ds.size == 0:
        return None, "scalar/empty - contiguous"

    if looks_like_frames(ds):
        t = min(frames_per_chunk, ds.shape[0])
        return (t,) + tuple(ds.shape[1:]), f"frames: {t} frames/chunk, full spatial extent"

    if ds.ndim == 1:
        n = min(SIG_CHUNK, ds.shape[0])
        if ds.nbytes < 1 << 20:
            return None, "small 1-D - contiguous"
        return (n,), f"1-D signal: {n} elements/chunk"

    # Anything else keeps the source chunking, or stays contiguous.
    return (ds.chunks, "unrecognised shape - keeping source chunking") if ds.chunks \
        else (None, "unrecognised shape - contiguous")


def amplification(ds_shape, chunks, clip_len: int, itemsize: int):
    """Returns (bytes read / bytes needed, chunks touched) for one clip_len clip."""
    if not chunks:
        return 1.0, 0
    needed = clip_len
    for d in ds_shape[1:]:
        needed *= d
    needed *= itemsize

    n_chunks_time = math.ceil(clip_len / chunks[0]) + 1   # +1: a clip rarely aligns
    per_chunk = itemsize
    for d in chunks:
        per_chunk *= d
    # Chunks spanning the full spatial extent are read once per time block;
    # narrower ones multiply by the number of tiles covering the spatial dims.
    spatial = 1
    for cd, sd in zip(chunks[1:], ds_shape[1:]):
        spatial *= math.ceil(sd / cd)
    read = n_chunks_time * spatial * per_chunk
    return (read / needed if needed else 1.0), n_chunks_time * spatial


# -------------------------------------------------------------------- walking --

def collect(src: h5py.File):
    """Returns flat lists of group and dataset names, in visit order."""
    groups, datasets = [], []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)
        elif isinstance(obj, h5py.Group):
            groups.append(name)

    src.visititems(visit)
    return groups, datasets


def copy_attrs(a_src, a_dst):
    """Copies attributes one by one, reporting any that cannot be written."""
    for k, v in a_src.items():
        try:
            a_dst[k] = v
        except Exception as e:                    # noqa: BLE001
            print(f"      ! attribute {k!r} not copied: {e!r}")


def copy_dataset(src_ds: h5py.Dataset, dst_parent: h5py.Group, name: str,
                 chunks, compression, comp_opts) -> None:
    """Copies one dataset in blocks under the planned chunk layout."""
    kwargs = dict(shape=src_ds.shape, dtype=src_ds.dtype)
    if chunks:
        kwargs["chunks"] = tuple(chunks)
        if compression:
            kwargs["compression"] = compression
            if comp_opts is not None:
                kwargs["compression_opts"] = comp_opts
    if src_ds.maxshape != src_ds.shape:
        kwargs["maxshape"] = src_ds.maxshape

    dst_ds = dst_parent.create_dataset(name, **kwargs)

    if src_ds.size:
        if src_ds.ndim == 0:
            dst_ds[()] = src_ds[()]
        else:
            step = max(1, COPY_BLOCK_FRAMES if src_ds.ndim >= 3 else 1 << 20)
            for i in range(0, src_ds.shape[0], step):
                j = min(i + step, src_ds.shape[0])
                dst_ds[i:j] = src_ds[i:j]

    copy_attrs(src_ds.attrs, dst_ds.attrs)


# ------------------------------------------------------------------ per file --

def inspect_file(path: Path, frames_per_chunk: int, clip_len: int, top: int):
    """Prints the current layout of a store and the planned rechunking."""
    print(f"\n=== {path.name}  ({human(path.stat().st_size)}) ===")
    with h5py.File(path, "r") as f:
        groups, datasets = collect(f)
        print(f"  groups: {len(groups)}   datasets: {len(datasets)}")
        if f.attrs:
            print(f"  file attributes: {dict(f.attrs)}")

        frames_like = [n for n in datasets if looks_like_frames(f[n])]
        print(f"  datasets that look like frame stacks: {len(frames_like)}")

        # Frame stacks come first because they dominate read cost; after them, one
        # example of every other distinct shape/chunk/compression combination.
        order = frames_like + [n for n in datasets if n not in set(frames_like)]
        shown = 0
        seen_shapes = set()
        for name in order:
            ds = f[name]
            key = (ds.shape[1:], ds.dtype.str, ds.chunks, ds.compression)
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            new, reason = plan_chunks(ds, frames_per_chunk)
            amp_old, n_old = amplification(ds.shape, ds.chunks, clip_len, ds.dtype.itemsize)
            amp_new, n_new = amplification(ds.shape, new, clip_len, ds.dtype.itemsize)
            print(f"  - {name}")
            print(f"      shape={ds.shape} dtype={ds.dtype} compression={ds.compression}"
                  f"{'' if ds.compression_opts is None else f':{ds.compression_opts}'}")
            print(f"      chunks: {ds.chunks}  ->  {new}      ({reason})")
            if looks_like_frames(ds):
                print(f"      {clip_len}-frame clip: {n_old} chunks / amplification {amp_old:5.1f}x"
                      f"   ->   {n_new} chunks / {amp_new:5.1f}x")
            shown += 1
            if shown >= top:
                print(f"      ... ({len(datasets) - shown} datasets omitted - "
                      f"they repeat one of the layouts above)")
                break


def rechunk_file(path: Path, out: Path, frames_per_chunk: int, compression, comp_opts,
                 sample_records: int | None, verify: bool, clip_len: int) -> bool:
    """Writes a rechunked copy of one store and optionally verifies it."""
    if out.exists():
        print(f"  SKIPPED: {out.name} already exists (delete it to rewrite)")
        return True

    tmp = out.with_suffix(".h5.part")
    if tmp.exists():
        tmp.unlink()

    t0 = time.perf_counter()
    copied_bytes = 0
    n_ds = 0

    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)
        # The marker stops a later run from treating this output as a new input.
        dst.attrs["rechunked_by"] = "rechunk_h5.py"
        dst.attrs["rechunked_from"] = path.name
        dst.attrs["rechunk_frames_per_chunk"] = frames_per_chunk

        top_names = list(src.keys())
        if sample_records:
            top_names = top_names[:sample_records]
            print(f"  SAMPLE MODE: copying only {len(top_names)} of {len(src.keys())} "
                  f"top-level objects - the output WILL BE INCOMPLETE")

        total_ds = 0
        for tn in top_names:
            obj = src[tn]
            if isinstance(obj, h5py.Dataset):
                total_ds += 1
            else:
                total_ds += sum(1 for _ in _iter_datasets(obj))

        for tn in top_names:
            obj = src[tn]
            if isinstance(obj, h5py.Dataset):
                new, _ = plan_chunks(obj, frames_per_chunk)
                copy_dataset(obj, dst, tn, new, compression, comp_opts)
                copied_bytes += obj.nbytes
                n_ds += 1
            else:
                g = dst.create_group(tn)
                copy_attrs(obj.attrs, g.attrs)
                for rel, sub in _iter_all(obj):
                    if isinstance(sub, h5py.Group):
                        gg = g.require_group(rel)
                        copy_attrs(sub.attrs, gg.attrs)
                    else:
                        # HDF5 paths always use '/'. os.path would also split on '\\'
                        # on Windows, so the path is split explicitly.
                        parent_rel, _, leaf = rel.rpartition("/")
                        gg = g.require_group(parent_rel) if parent_rel else g
                        new, _ = plan_chunks(sub, frames_per_chunk)
                        copy_dataset(sub, gg, leaf, new, compression, comp_opts)
                        copied_bytes += sub.nbytes
                        n_ds += 1
                        if n_ds % 200 == 0:
                            el = time.perf_counter() - t0
                            rate = copied_bytes / el if el else 0
                            print(f"    {n_ds}/{total_ds} datasets | {human(copied_bytes)} "
                                  f"| {human(rate)}/s | {el/60:5.1f} min")

    tmp.rename(out)
    el = time.perf_counter() - t0
    print(f"  done: {n_ds} datasets, {human(copied_bytes)} of data in {el/60:.1f} min")
    print(f"  file size: {human(path.stat().st_size)}  ->  {human(out.stat().st_size)}"
          f"   ({100 * out.stat().st_size / path.stat().st_size:.0f}%)")

    if verify:
        ok = verify_file(path, out, clip_len, partial=bool(sample_records))
        print(f"  verification: {'OK' if ok else 'FAILED'}")
        return ok
    return True


def _iter_datasets(group: h5py.Group):
    """Yields every dataset under a group."""
    for _, obj in _iter_all(group):
        if isinstance(obj, h5py.Dataset):
            yield obj


def _iter_all(group: h5py.Group):
    """Returns (relative_name, object) for everything under a group, groups first."""
    out = []
    group.visititems(lambda n, o: out.append((n, o)))
    out.sort(key=lambda t: (not isinstance(t[1], h5py.Group), t[0]))
    return out


# ------------------------------------------------------------------- verify --

def verify_file(src_path: Path, dst_path: Path, clip_len: int, n_spot: int = 40,
                partial: bool = False) -> bool:
    """Compares a rechunked store against its source.

    Checks the dataset inventory, file and dataset attributes, shapes and dtypes,
    and the contents of n_spot randomly chosen clip-length blocks.
    """
    rng = random.Random(0)
    ok = True
    with h5py.File(src_path, "r") as a, h5py.File(dst_path, "r") as b:
        ga, da = collect(a)
        gb, db = collect(b)

        missing = set(da) - set(db)
        extra = set(db) - set(da)
        if extra:
            print(f"    ! unexpected datasets in the output: {len(extra)}")
            ok = False
        if missing:
            if partial:
                print(f"    (sample mode: {len(missing)} datasets deliberately not copied)")
            else:
                print(f"    ! {len(missing)} datasets missing, e.g.: {sorted(missing)[:3]}")
                ok = False

        marker = {"rechunked_by", "rechunked_from", "rechunk_frames_per_chunk"}
        attrs_a = {k: v for k, v in a.attrs.items() if k not in marker}
        attrs_b = {k: v for k, v in b.attrs.items() if k not in marker}
        if str(attrs_a) != str(attrs_b):
            print("    ! file attributes differ")
            print(f"      source: {attrs_a}")
            print(f"      output: {attrs_b}")
            ok = False

        common = sorted(set(da) & set(db))
        for name in common:
            if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
                print(f"    ! {name}: shape/dtype {a[name].shape}/{a[name].dtype} "
                      f"vs {b[name].shape}/{b[name].dtype}")
                ok = False
            if dict(a[name].attrs) != dict(b[name].attrs):
                print(f"    ! {name}: attributes differ")
                ok = False

        frames = [n for n in common if looks_like_frames(a[n])] or common
        for name in rng.sample(frames, min(n_spot, len(frames))):
            ds_a, ds_b = a[name], b[name]
            if ds_a.ndim == 0 or ds_a.shape[0] == 0:
                continue
            t = ds_a.shape[0]
            i = rng.randrange(0, max(1, t - 1))
            j = min(t, i + min(clip_len, t))
            if not np.array_equal(ds_a[i:j], ds_b[i:j]):
                print(f"    ! {name}[{i}:{j}] - contents differ")
                ok = False
    return ok


# ---------------------------------------------------------------------- main --

def main() -> int:
    """Parses arguments, inspects the stores, and rechunks them when asked."""
    p = argparse.ArgumentParser(description="Rechunks rPPG HDF5 stores for clip reads.")
    p.add_argument("data_dir", type=Path, help="directory holding the .h5 stores")
    p.add_argument("--only", default=None,
                   help="process only stores whose name contains this substring")
    p.add_argument("--apply", action="store_true",
                   help="write the rechunked stores (without it: inspection only)")
    p.add_argument("--compression", default="none",
                   help="none | lzf | gzip | gzip:N   (default: none, as used for *_rc_none)")
    p.add_argument("--frames-per-chunk", type=int, default=DEFAULT_FRAMES_PER_CHUNK)
    p.add_argument("--clip-len", type=int, default=DEFAULT_CLIP_LEN,
                   help="clip length used only to report read amplification")
    p.add_argument("--suffix", default="_rc_none", help="suffix for the output file name")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default: the source directory)")
    p.add_argument("--sample-records", type=int, default=None,
                   help="copy only the first N top-level objects - a smoke test; "
                        "the output is incomplete")
    p.add_argument("--verify", action="store_true", help="check each output after writing")
    p.add_argument("--top", type=int, default=6,
                   help="number of datasets to print per store during inspection")
    args = p.parse_args()

    if not args.data_dir.is_dir():
        print(f"directory not found: {args.data_dir}")
        return 2

    files = sorted(args.data_dir.glob("*.h5"))
    if args.only:
        files = [f for f in files if args.only.lower() in f.name.lower()]

    # A previous run's output is never reprocessed, whatever suffix it used.
    kept = []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                if "rechunked_by" in h.attrs:
                    print(f"skipping {f.name} - output of an earlier run")
                    continue
        except Exception as e:                    # noqa: BLE001
            print(f"skipping {f.name} - cannot be opened: {e!r}")
            continue
        kept.append(f)
    files = kept
    if not files:
        print("no .h5 stores to process")
        return 2

    compression, comp_opts = parse_compression(args.compression)
    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"h5py {h5py.__version__} | stores: {len(files)} | frames per chunk: "
          f"{args.frames_per_chunk} | compression: {args.compression}")
    print(f"output: {out_dir}")

    total_in = sum(f.stat().st_size for f in files)
    try:
        free = os.statvfs(out_dir).f_bavail * os.statvfs(out_dir).f_frsize
    except AttributeError:
        import shutil
        free = shutil.disk_usage(out_dir).free
    print(f"input total: {human(total_in)} | free on target drive: {human(free)}")
    if args.apply and free < total_in * 1.3:
        print("! low disk space. Without compression an output can be much larger "
              "than its source (the gzip:1 Phase-3 stores grew 1.3-2.1x).")

    for f in files:
        inspect_file(f, args.frames_per_chunk, args.clip_len, args.top)

    if not args.apply:
        print("\n--- INSPECTION ONLY. Nothing was written. Add --apply to rewrite. ---")
        return 0

    print("\n" + "=" * 74)
    all_ok = True
    for f in files:
        out = out_dir / f"{f.stem}{args.suffix}.h5"
        print(f"\n>>> {f.name}  ->  {out.name}")
        all_ok &= rechunk_file(f, out, args.frames_per_chunk, compression, comp_opts,
                               args.sample_records, args.verify, args.clip_len)
    print("\n" + "=" * 74)
    print("all OK" if all_ok else "problems found - read the messages above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
