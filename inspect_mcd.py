import h5py, numpy as np, random
from pathlib import Path

OUT_H5 = Path('G:/rppg/phase3/mcd_phase3.h5')
MCD_DIR = Path('G:/rppg/MCD_rPPG_dataset')
TARGET = 72

def load_col0(p):
    out = []
    with open(p) as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(float(s.split()[0]))
    return np.asarray(out, dtype=np.float32)

issues, n_list, sqis, not_usable = [], [], [], []
with h5py.File(OUT_H5, 'r') as store:
    groups = sorted(store.keys())
    for name in groups:
        g = store[name]
        if 'frames' not in g or 'bvp' not in g: issues.append(name + ' missing ds')
        if bool(g.attrs.get('has_ecg')) and 'ecg' not in g: issues.append(name + ' missing ecg')
        fr, bvp = g['frames'], g['bvp']
        na = int(g.attrs['n_frames'])
        if fr.shape[1:] != (TARGET, TARGET, 3) or fr.dtype != np.uint8: issues.append(name + ' frame ' + str(fr.shape))
        if fr.shape[0] != na or bvp.shape[0] != na: issues.append(name + ' length mismatch')
        n_list.append(fr.shape[0]); sqis.append(float(g.attrs['cardiac_sqi']))
        if not bool(g.attrs['usable']): not_usable.append(name)

sq = np.array(sqis)
print('groups:', len(groups), '| hard issues:', len(issues))
for m in issues[:15]: print('  ', m)
print('n_frames: min', min(n_list), 'median', int(np.median(n_list)), 'max', max(n_list))
print('cardiac_sqi pct:', {p: round(float(np.percentile(sq, p)), 3) for p in (0, 10, 25, 50, 75, 90)})
print('sqi<0.10:', int((sq < 0.10).sum()), '| sqi<0.05:', int((sq < 0.05).sum()), '| not usable:', len(not_usable))

rng = random.Random(0)
with h5py.File(OUT_H5, 'r') as store:
    for name in rng.sample(groups, 5):
        g = store[name]
        src = load_col0(MCD_DIR / 'ppg_sync' / g.attrs['source_file'].replace('.avi', '.txt'))
        n = g['bvp'].shape[0]
        print('bvp roundtrip', name, np.allclose(g['bvp'][:], src[:n], atol=1e-2))


import pandas as pd

def sqi(sig, fps, low=0.7, high=3.5):
    x = np.asarray(sig, np.float64)
    if x.size < 16 or not np.all(np.isfinite(x)) or np.std(x) < 1e-9: return 0.0
    x = (x - x.mean()) * np.hanning(x.size)
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    b = (f >= low) & (f <= high)
    if not b.any() or p[b].sum() <= 0: return 0.0
    k = int(np.argmax(p[b]))
    return float(p[b][max(0, k - 1):k + 2].sum() / p[b].sum())

def load_two_col(p):
    v, t = [], []
    with open(p) as f:
        for line in f:
            a = line.strip().split(maxsplit=1)
            if len(a) == 2:
                v.append(float(a[0])); t.append(pd.Timestamp(a[1]).value)
    return np.array(v, np.float64), np.array(t, np.float64)

rng = random.Random(1)
with h5py.File(OUT_H5, 'r') as store:
    for name in rng.sample(sorted(store.keys()), 5):
        g = store[name]
        fps = float(g.attrs['fps']); sync = g['bvp'][:]
        pid, _, state = g.attrs['source_file'].replace('.avi', '').split('_')
        ppg_v, ppg_t = load_two_col(MCD_DIR / 'ppg' / (pid + '_' + state + '.PW'))
        meta_t, _ = load_two_col(MCD_DIR / 'meta' / g.attrs['source_file'].replace('.avi', '.txt'))
        interp = np.interp(meta_t, ppg_t, ppg_v)
        print(name, '| ppg_sync', round(sqi(sync, fps), 3),
              '| raw100Hz', round(sqi(ppg_v, 100.0), 3),
              '| raw->frames interp', round(sqi(interp[:len(sync)], fps), 3))