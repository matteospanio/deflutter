"""Audio I/O, hashing, export and the deterministic synthetic test-signal generator (doc §7.1)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SUBTYPES = {"16": "PCM_16", "24": "PCM_24", "32f": "FLOAT"}


def load(path):
    """Return (samples float32 [frames, channels], sample_rate)."""
    return sf.read(str(path), dtype="float32", always_2d=True)


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def export(path, x, sr, bit_depth="24"):
    """Write x [frames, channels] (format from the extension) and re-open it to verify it.

    Returns the number of samples that had to be clipped (always 0 for 32-bit float).
    """
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN/Inf, refusing to export")
    clipped = 0
    if bit_depth != "32f":
        clipped = int(np.count_nonzero(np.abs(x) > 1.0))
        x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x, sr, subtype=SUBTYPES[bit_depth])
    info = sf.info(str(path))
    if (info.frames, info.samplerate, info.channels) != (len(x), sr, x.shape[1]):
        raise IOError(f"exported file does not match the render: {info}")
    return clipped


# ---- synthetic signals -------------------------------------------------------------------------

def bump(t, t0, t1, edge):
    """0 -> 1 -> 0 window over [t0, t1] with raised-cosine edges of `edge` seconds."""
    w = np.clip(np.minimum(t - t0, t1 - t) / edge, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * w)


# r(t) profiles with known shape (doc §6): smoothed step, gaussian, ramp, several separated events
PROFILES = {
    "step": lambda t: 1 + 0.02 * bump(t, 6, 12, 0.3),
    "gaussian": lambda t: 1 - 0.03 * np.exp(-0.5 * ((t - 10) / 1.0) ** 2),
    "ramp": lambda t: 1 + 0.025 * np.clip((t - 6) / 6, 0, 1) * bump(t, 6, 12.5, 0.5),
    "multi": lambda t: (1 + 0.02 * bump(t, 3, 6, 0.3) - 0.015 * np.exp(-0.5 * ((t - 10) / 0.6) ** 2)
                        + 0.01 * bump(t, 14, 17, 0.4)),
}


def synth(profile=None, sr=48000, dur=20.0, freqs=(440, 880, 1500, 3000), amps=(0.2, 0.15, 0.1, 0.08),
          deviating=None, hum=60.0, noise_db=-60.0, channels=1, seed=0):
    """Sum of sinusoids whose instantaneous frequency is f_i * r(t), i.e. phase 2*pi*f_i*integral(r dt).

    profile: callable t -> r(t) (None = no anomaly). deviating: indices of the components that follow r(t)
    (None = all of them, a global anomaly). The `hum` tone never deviates: it is an additive machine tone.
    Deterministic for a given seed. Returns float32 [frames, channels].
    """
    t = np.arange(int(sr * dur)) / sr
    warp = np.cumsum(profile(t)) / sr if profile else t
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    for i, (f, a) in enumerate(zip(freqs, amps)):
        tau = warp if deviating is None or i in deviating else t
        x += a * np.sin(2 * np.pi * f * tau + rng.uniform(0, 2 * np.pi))
    if hum:
        x += 0.05 * np.sin(2 * np.pi * hum * t)
    noise = 10 ** (noise_db / 20)
    out = [x * (1 - 0.3 * c) + noise * rng.standard_normal(len(t)) for c in range(channels)]
    return np.stack(out, axis=1).astype(np.float32)


if __name__ == "__main__":
    # python -m deflutter.audio <dir>: write demo WAVs for every profile plus a non-global case
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
    out.mkdir(parents=True, exist_ok=True)
    for name, prof in PROFILES.items():
        sf.write(out / f"{name}.wav", synth(prof, channels=2), 48000, subtype="PCM_24")
    sf.write(out / "nonglobal.wav", synth(PROFILES["gaussian"], deviating={2}), 48000, subtype="PCM_24")
    print(f"wrote demo files to {out}/")
