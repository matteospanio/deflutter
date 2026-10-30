"""Signal analysis (doc §3.1-3.3): STFT, reference tracking, nominal f0, anomaly detection and profiles.

Deviations are relative: e(t) = f_obs(t) / f0 - 1, and the correction ratio is r(t) = 1 + e_hat(t).
Pure NumPy/SciPy (no Qt), so the same code runs in GUI worker threads and in the tests.
`progress` arguments are optional callables taking a 0..1 fraction; they may raise to cancel.
"""
import warnings

import numpy as np
from scipy import fft, ndimage, signal
from scipy.interpolate import make_smoothing_spline

MAD_TO_SIGMA = 1.4826


def cents(e):
    return 1200 * np.log2(1 + np.asarray(e))


def stft_db(x, sr, n_fft=8192, hop=1024, window="hann", t0=0.0, t1=None, fmin=0.0, fmax=None,
            max_cells=200_000_000, block=1024, progress=None):
    """Magnitude STFT (dB re full-scale sine) of mono x between t0 and t1, keeping bins in [fmin, fmax].

    Returns (times, freqs, db[frames, bins]) with times = absolute frame centers in seconds; db is float16
    (0.06 dB resolution, plenty for peak picking) and the hop is raised if frames*bins would exceed
    max_cells, which bounds memory on long selections. FFTs are computed in blocks of frames, each
    block multi-threaded (scipy.fft workers=-1).
    """
    a = int(round(t0 * sr))
    seg = x[a:len(x) if t1 is None else int(round(t1 * sr))]
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    k0 = np.searchsorted(freqs, fmin)
    k1 = np.searchsorted(freqs, sr / 2 if fmax is None else fmax, side="right")
    hop = max(hop, int(np.ceil((len(seg) - n_fft) * (k1 - k0) / max_cells)))
    frames = np.lib.stride_tricks.sliding_window_view(seg, n_fft)[::hop]
    win = signal.get_window(window, n_fft).astype(np.float32)
    scale = 2 / win.sum()
    db = np.empty((len(frames), k1 - k0), np.float16)
    for i in range(0, len(frames), block):
        spec = fft.rfft(frames[i:i + block] * win, axis=-1, workers=-1)[:, k0:k1]
        db[i:i + block] = 20 * np.log10(np.abs(spec) * scale + 1e-9)
        if progress:
            progress(min(1.0, (i + block) / len(frames)))
    times = (a + np.arange(len(frames)) * hop + n_fft / 2) / sr
    return times, freqs[k0:k1], db


def display_spectrogram(x, sr, n_fft=4096, max_cols=6000, progress=None):
    """Full-band spectrogram for display; the hop grows so long files stay under max_cols columns."""
    hop = max(n_fft // 4, int(np.ceil(len(x) / max_cols)))
    return stft_db(x, sr, n_fft, hop, progress=progress)


def _parabolic(a, b, c):
    """Sub-bin offset of a peak from three log-magnitude samples (b is the local maximum)."""
    d = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d < 0, 0.5 * (a - c) / d, 0.0)


def find_candidates(freqs, db, prominence_db=6.0, max_n=12):
    """Persistent tonal components (Hz): peaks of the 90th-percentile spectrum over a smoothed floor.

    The 90th percentile (rather than the median) still shows tones that sit at their nominal
    frequency only part of the time, or drift slowly across a few bins.
    """
    spec = np.percentile(db[::max(1, len(db) // 4000)].astype(np.float32), 90, axis=0)
    floor = ndimage.median_filter(spec, size=31, mode="nearest")
    peaks, props = signal.find_peaks(spec - floor, height=prominence_db, distance=3)
    peaks = peaks[np.argsort(props["peak_heights"])[::-1][:max_n]]
    peaks = peaks[(peaks > 0) & (peaks < len(spec) - 1)]
    k = peaks + _parabolic(spec[peaks - 1], spec[peaks], spec[peaks + 1])
    return sorted(np.interp(k, np.arange(len(freqs)), freqs).tolist())


def track(freqs, db, center, search_pct=15.0, step_pct=0.5, min_conf_db=6.0, stick_db=0.5, relock_db=3.0,
          seed=None):
    """Follow the tone near `center` frame by frame (doc §3.1). Returns (f_obs Hz, confidence dB).

    Seeds at frame `seed` (default: the most confident frame within +-step_pct of center), then walks
    forward and backward,
    looking for the best peak within +-step_pct of the last good estimate and never leaving
    +-search_pct of center. Following the tone instead of searching one wide window copes with large
    deviations without jumping to neighbouring tones. To keep a weak tone from random-walking onto
    noise peaks, candidates pay stick_db per bin of distance from the last estimate, and after a
    dropout (window widened up to 4x) re-locking needs relock_db more confidence. Peaks are refined by
    parabolic interpolation on the dB values. Confidence = peak minus the median level of the search
    band; frames below min_conf_db are NaN.
    """
    n, nb = db.shape
    df, f_lo = freqs[1] - freqs[0], freqs[0]
    kc = int((center - f_lo) / df)  # search band: +-search_pct, but at least +-8 bins
    g0 = max(min(int((center * (1 - search_pct / 100) - f_lo) / df), kc - 8), 1)
    g1 = min(max(int(np.ceil((center * (1 + search_pct / 100) - f_lo) / df)), kc + 8), nb - 2)
    f_obs, conf = np.full(n, np.nan), np.zeros(n)
    if g1 - g0 < 4 or n == 0:
        return f_obs, conf
    band = db[:, g0:g1 + 1:max(1, (g1 - g0) // 64)]  # <= ~64 bins are plenty for a noise floor
    noise = np.median(band.astype(np.float32), axis=1)

    def peak(m, est, miss):
        half = max(est * step_pct / 100 * min(1 + miss, 4), 4 * df)
        a = max(int((est - half - f_lo) / df), g0)
        b = min(int(np.ceil((est + half - f_lo) / df)), g1)
        bins = np.arange(a, b + 1)
        k = a + int(np.argmax(db[m, a:b + 1] - stick_db * np.abs(bins - (est - f_lo) / df)))
        y0, y1, y2 = db[m, k - 1:k + 2].astype(np.float32)
        conf[m] = y1 - noise[m]
        if conf[m] >= min_conf_db + (relock_db if miss else 0) and a < k < b:
            f_obs[m] = f_lo + (k + float(_parabolic(y0, y1, y2))) * df

    half = max(center * step_pct / 100, 4 * df)
    a, b = max(int((center - half - f_lo) / df), g0), min(int((center + half - f_lo) / df), g1)
    if seed is None:
        seed = int(np.argmax(db[:, a:b + 1].max(axis=1).astype(np.float32) - noise))
    for frames in (range(seed, n), range(seed - 1, -1, -1)):
        est, miss = center, 0
        for m in frames:
            peak(m, est, miss)
            if np.isfinite(f_obs[m]):
                est, miss = f_obs[m], 0
            else:
                miss += 1
    return f_obs, conf


def classify(f_obs, f0, min_valid=0.6, max_jitter=5e-4, min_dev=2e-3):
    """'unstable', 'moving' or 'fixed'.

    A usable reference is tracked in most frames and moves smoothly: its median frame-to-frame change
    stays below max_jitter (0.05 %). Musical partials fail this (they jump between notes), even when a
    real reference deviates a lot over the analysed range. A smooth track is 'moving' when its
    excursions exceed both min_dev and 6x the tracker noise, else 'fixed'.
    """
    e = f_obs / f0 - 1
    v = e[np.isfinite(e)]
    d = np.abs(np.diff(e))
    d = d[np.isfinite(d)]
    if len(v) < min_valid * len(e) or not len(d) or np.median(d) > max_jitter:
        return "unstable"
    noise = MAD_TO_SIGMA * float(np.median(d)) / np.sqrt(2)  # frame-to-frame jitter -> per-frame noise
    excursion = float(np.percentile(np.abs(v - np.median(v)), 99))
    return "moving" if excursion > max(6 * noise, min_dev) else "fixed"


def propose_references(freqs, db, prominence_db=6.0, search_pct=15.0, max_n=12, max_candidates=32):
    """Track candidate tones and keep up to max_n usable references, sorted by frequency.

    Unstable tracks (e.g. musical partials) and duplicates (two candidates ending up on the same tone)
    are dropped. Returns [dict(center, f0, f_obs, conf, kind)].
    """
    refs = []
    for c in find_candidates(freqs, db, prominence_db, max_candidates):
        f_obs, conf = track(freqs, db, c, search_pct)
        f0 = estimate_f0(f_obs)
        if not np.isfinite(f0) or any(abs(f0 / r["f0"] - 1) < 5e-3 for r in refs):
            continue
        kind = classify(f_obs, f0)
        if kind != "unstable":
            refs.append(dict(center=c, f0=f0, f_obs=f_obs, conf=conf, kind=kind))
    return sorted(refs[:max_n], key=lambda r: r["f0"])


def robust_sigma(e):
    e = e[np.isfinite(e)]
    return MAD_TO_SIGMA * float(np.median(np.abs(e - np.median(e)))) if len(e) else 0.0


def estimate_f0(f_obs, k_off=2.0):
    """Robust nominal frequency: median, then median of the frames that are stable around it."""
    f = f_obs[np.isfinite(f_obs)]
    if not len(f):
        return float("nan")
    f0 = float(np.median(f))
    e = f / f0 - 1
    stable = np.abs(e) <= max(k_off * robust_sigma(e), 1e-5)
    return float(np.median(f[stable])) if stable.any() else f0


def fill_gaps(e):
    """Linearly interpolate NaN frames (all-NaN -> zeros)."""
    ok = np.isfinite(e)
    if ok.all() or not ok.any():
        return np.where(ok, e, 0.0)
    idx = np.arange(len(e))
    return np.interp(idx, idx[ok], e[ok])


def combine(es):
    """Per-frame median deviation across references, with gaps interpolated."""
    # ponytail: plain median; weight by confidence if references often disagree
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames
        e = np.nanmedian(np.vstack(es), axis=0)
    return fill_gaps(e)


def _runs(mask):
    """(start, stop) index pairs of the True runs in a boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def thresholds(e, k_on=4.0, k_off=2.0, min_dev=5e-4):
    """Trigger / release thresholds on |e|: k * robust sigma, never below min_dev (resp. min_dev/2)."""
    s = robust_sigma(e)
    return max(k_on * s, min_dev), max(k_off * s, min_dev / 2)


def detect_events(t, e, thr_on, thr_off, min_dur=0.1, gap=0.2, roll=0.15):
    """Hysteresis detection on |e| (doc §3.2).

    An event is a run of |e| > thr_off containing at least min_dur of consecutive frames above thr_on.
    Events are padded by `roll` (pre/post-roll), clipped to the analysed range and merged when closer
    than `gap`. Returns [(start_s, end_s)].
    """
    a = np.abs(e)
    min_frames = max(1, int(round(min_dur / (t[1] - t[0]))))
    out = []
    for i0, i1 in _runs(a > thr_off):
        if max((s1 - s0 for s0, s1 in _runs(a[i0:i1] > thr_on)), default=0) < min_frames:
            continue
        s, en = max(t[i0] - roll, t[0]), min(t[i1 - 1] + roll, t[-1])
        if out and s - out[-1][1] <= gap:
            out[-1][1] = en
        else:
            out.append([s, en])
    return [tuple(map(float, ev)) for ev in out]


def fit_profile(t, e, start, end, roll, smooth=0.0, open_ends=(False, False), span=(0.0, np.inf)):
    """Smooth deviation curve e_hat over [start, end], tapered to 0 at the edges (doc §3.3).

    A smoothing spline keeps the profile from chasing tracker noise: smooth=0 lets GCV choose the
    amount, otherwise `smooth` is the time scale in seconds that gets smoothed out. A raised-cosine
    taper over the pre/post-roll forces r(t) = 1 at the event edges, except at an "open" end (the
    event reaches the start/end of the file): there the profile is held flat out to `span` instead.
    Returns (t_ev, e_hat).
    """
    m = (t >= start) & (t <= end)
    tt, ee = t[m], e[m]
    if len(tt) >= 5:
        dt = tt[1] - tt[0]
        try:  # lam ~ h^4 / dt gives an equivalent kernel width of about h
            ee = make_smoothing_spline(tt, ee, lam=smooth ** 4 / dt if smooth > 0 else None)(tt)
        except (ValueError, np.linalg.LinAlgError):
            pass
    if not len(tt):
        return tt, ee
    left = np.inf if open_ends[0] else tt - start
    right = np.inf if open_ends[1] else end - tt
    edge = np.clip(np.minimum(left, right) / max(roll, 1e-3), 0, 1)
    ee = ee * (0.5 - 0.5 * np.cos(np.pi * edge))
    if open_ends[0]:
        tt, ee = np.r_[span[0], tt], np.r_[ee[0], ee]
    if open_ends[1]:
        tt, ee = np.r_[tt, span[1]], np.r_[ee, ee[-1]]
    return tt, ee


def coherence(t, es, labels, e_comb, start, end, thr_off, tol=0.25, abs_tol=5e-4):
    """Do all references deviate by the same ratio during the event? (global-model check, doc §2, §9)

    Compares each reference's mean deviation over the event core (|e_comb| > thr_off).
    Returns (status, message) with status 'coherent', 'incoherent' or 'unverified'.
    """
    m = (t >= start) & (t <= end)
    core = m & (np.abs(e_comb) > thr_off)
    core = core if core.any() else m
    means = {}
    for lab, e in zip(labels, es):
        v = e[core & np.isfinite(e)]
        if len(v) >= max(3, core.sum() // 4):  # needs to be tracked over a good part of the event
            means[lab] = float(v.mean())
    if len(means) < 2:
        return "unverified", "only one usable reference: global deviation not verified"
    vals = np.array(list(means.values()))
    spread, ref = vals.max() - vals.min(), np.median(vals)
    if spread <= max(abs_tol, tol * abs(ref)):
        return "coherent", f"{len(means)} references agree ({cents(ref):+.1f} cents)"
    detail = ", ".join(f"{lab}: {cents(v):+.1f}" for lab, v in means.items())
    return "incoherent", f"references disagree (cents) - {detail}"


def ratio_curve(t, profiles):
    """r(t) = 1 + sum of the event profiles (already scaled by their intensity); 1 elsewhere."""
    r = np.ones_like(t, dtype=float)
    for tt, ee in profiles:
        r += np.interp(t, tt, ee, left=0, right=0)
    return r


def residual(t, e_before, e_after, spans):
    """RMS deviation before/after correction over the event spans and the reduction in dB."""
    m = np.zeros(len(t), bool)
    for s, en in spans:
        m |= (t >= s) & (t <= en)
    rms = lambda e: float(np.sqrt(np.nanmean(e[m] ** 2))) if m.any() else float("nan")  # noqa: E731
    before, after = rms(e_before), rms(e_after)
    return before, after, 20 * np.log10(before / max(after, 1e-12))
