"""Constant-duration, time-varying pitch correction with a phase vocoder (doc §3.4).

Every STFT frame m is pitch-shifted in the frequency domain by 1/r_m: output bin k takes the magnitude
found at source bin q = k * r_m and a phase advanced by the corrected instantaneous frequency, so
f_out = f_in / r(t) while the frames stay on the original time grid (same sample count, events stay put).

- Identity phase locking (Laroche & Dolson): only spectral peaks are phase-propagated; the bins around
  each peak keep their original phase relation to it, which avoids most of the "phasiness".
- Transient preservation: a peak whose magnitude jumps up resets to the input phase.
- Resync: after an event, the accumulated phase offset is faded out over `tail_s` seconds so the output
  returns exactly to the original signal (frequency error during the fade <= 1 / (2 * tail_s) Hz).
- Stereo: one phase trajectory computed on the channel sum; every channel keeps its own phase offset
  relative to it, so the stereo image is preserved and all channels get the same correction.

No Qt imports: regions are rendered in worker processes.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import fft, signal


def _wrap(p):
    return p - 2 * np.pi * np.rint(p / (2 * np.pi))


def correct_region(x, r, keep, n_fft, hop, transient_db=6.0, progress=None):
    """Phase-vocoder pitch correction of x [samples, channels].

    Frame m covers x[m*hop : m*hop + n_fft]. r[m] is its frequency ratio; keep[m] the fraction of the
    phase-vocoder phase offset kept at that frame (0 = pass the input frame through untouched).
    Returns y, same shape as x; only y[n_fft:-n_fft] has full overlap-add support.
    """
    x = np.asarray(x, np.float64)
    n_frames = 1 + (len(x) - n_fft) // hop
    nb = n_fft // 2 + 1
    k = np.arange(nb)
    omega = 2 * np.pi * k / n_fft  # bin center frequencies, rad/sample
    win = signal.get_window("hann", n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft, axis=0)[::hop]  # (frames, ch, n_fft)
    rise = 10 ** (transient_db / 20) if transient_db else np.inf
    y = np.zeros_like(x)
    wsum = np.zeros(len(x))
    phi_prev = mag_prev = psi = None
    block = 256
    for b0 in range(0, n_frames, block):
        X = fft.rfft(frames[b0:b0 + block] * win, axis=-1, workers=-1)  # (B, ch, nb)
        Y = X.copy()
        M = X.sum(axis=1)  # channel sum, (B, nb)
        PHI, MAG = np.angle(M), np.abs(M)  # vectorized per block, sliced per frame below
        OFFS = np.angle(X) - PHI[:, None] if X.shape[1] > 1 else None
        AMP = np.abs(X)
        for j in range(len(X)):
            m = b0 + j
            phi, mag = PHI[j], MAG[j]
            if psi is None or keep[m] == 0:  # outside events: identity, keep the phase state in sync
                psi, phi_prev, mag_prev = phi, phi, mag
                continue
            # instantaneous frequency of every source bin from the phase advance over one hop
            w_inst = omega + _wrap(phi - phi_prev - omega * hop) / hop
            q = k * r[m]  # source position of each output bin
            qi = np.minimum(np.rint(q).astype(int), nb - 1)
            mag_out = np.interp(q, k, mag, right=0.0)
            w_out = np.interp(q, k, w_inst) / r[m]
            peaks = np.flatnonzero((mag_out[1:-1] > mag_out[:-2]) & (mag_out[1:-1] >= mag_out[2:])) + 1
            if len(peaks):
                psi_pk = psi[peaks] + hop * w_out[peaks]
                onset = mag[qi[peaks]] > rise * mag_prev[qi[peaks]]  # ponytail: per-peak rise heuristic
                psi_pk = np.where(onset, phi[qi[peaks]], psi_pk)
                own = np.searchsorted((peaks[:-1] + peaks[1:]) / 2, k)  # nearest peak of every bin
                psi_new = psi_pk[own] + phi[qi] - phi[qi[peaks]][own]
            else:
                psi_new = psi + hop * w_out
            psi_new = phi[qi] + keep[m] * _wrap(psi_new - phi[qi])
            # every channel: remapped magnitude, shared phase trajectory, own phase offset to the sum
            mags = np.stack([np.interp(q, k, a, right=0.0) for a in AMP[j]])
            Y[j] = mags * np.exp(1j * (psi_new if OFFS is None else psi_new + OFFS[j][:, qi]))
            psi, phi_prev, mag_prev = _wrap(psi_new), phi, mag
        out = fft.irfft(Y, n=n_fft, axis=-1, workers=-1) * win
        for j in range(len(out)):
            s = (b0 + j) * hop
            y[s:s + n_fft] += out[j].T
            wsum[s:s + n_fft] += win ** 2
        if progress:
            progress(min(1.0, (b0 + block) / n_frames))
    return y / np.maximum(wsum, 1e-9)[:, None]


def _keep_curve(tc, spans, tail_s):
    """Per-frame fraction of the phase offset to keep: 1 inside events, a ramp to 0 after each.

    The ramp g(t) is turned into per-frame factors g[m] / g[m-1] so that the *product* over the tail
    (what the recursion applies to the offset) is the linear fade itself.
    """
    g = np.zeros_like(tc)
    for s, e in spans:
        g = np.maximum(g, ((tc >= s) & (tc <= e)).astype(float))
        g = np.maximum(g, np.where(tc > e, 1 - (tc - e) / tail_s, 0).clip(0, 1))
    prev = np.concatenate(([0.0], g[:-1]))
    return np.where(prev > 0, g / np.where(prev > 0, prev, 1), g)


def render(x, sr, t_r, r, spans, n_fft=4096, overlap=8, tail_s=1.0, transient_db=6.0, progress=None):
    """Corrected copy of x [samples, channels]; samples outside the processed regions are untouched.

    r(t) is given on the time grid t_r (1 outside the events); spans are the (start, end) seconds of the
    enabled events. Independent regions are rendered in parallel worker processes.
    Returns (y, regions) with regions = [(a, b)] sample ranges that were rewritten.
    """
    hop = n_fft // overlap
    pad = 2 * n_fft
    xp = np.pad(x, ((pad, pad), (0, 0)))  # so regions touching the file edges still have full support
    regions = []
    for s, e in sorted(spans):
        a = max(int(s * sr), 0)
        b = min(int((e + tail_s) * sr) + 2 * pad, len(xp))
        if regions and a <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], b)
        else:
            regions.append([a, b])
    jobs = []
    for a, b in regions:
        tc = (a - pad + np.arange(1 + (b - a - n_fft) // hop) * hop + n_fft / 2) / sr
        rm = np.interp(tc, t_r, r, left=1.0, right=1.0)
        jobs.append((xp[a:b], rm, _keep_curve(tc, spans, tail_s), n_fft, hop, transient_db))
    if len(jobs) == 1:  # no process start-up cost for the common single-event case
        results = [correct_region(*jobs[0], progress=progress)]
    else:
        # ponytail: a single long region stays sequential (~0.2 ms/frame); numba the frame loop if needed
        # spawn: never fork the (multi-threaded) GUI process
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1), mp_context=ctx) as ex:
            futures = [ex.submit(correct_region, *job) for job in jobs]
            try:
                results = []
                for i, f in enumerate(futures):
                    results.append(f.result())
                    if progress:
                        progress((i + 1) / len(futures))
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    y = xp.copy()
    for (a, b), yr in zip(regions, results):
        y[a + n_fft:b - n_fft] = yr[n_fft:-n_fft]
    y = y[pad:-pad].astype(x.dtype)
    assert y.shape == x.shape  # duration preserved by construction (doc requirement 8)
    return y, [(max(a - pad, 0), min(b - pad, len(x))) for a, b in regions]
