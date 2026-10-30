"""Numerical tests on deterministic synthetic signals (doc §7): tracking, stability, detection, profile,
correction, duration, determinism, the non-global case, export and persistence."""
from functools import lru_cache

import numpy as np
import pytest
import soundfile as sf

from deflutter import analysis as A
from deflutter import audio, correction
from deflutter.store import Store, cache_key

SR, N_FFT, HOP = 48000, 8192, 512
FREQS = (440, 880, 1500, 3000)


@lru_cache(maxsize=None)
def pipeline(name, deviating=None, channels=1):
    """Analyse + detect + correct a synthetic signal, the way the GUI does (all references enabled)."""
    prof = audio.PROFILES[name]
    x = audio.synth(prof, deviating=deviating and set(deviating), channels=channels)
    t, f, db = A.stft_db(x.mean(1), SR, N_FFT, HOP, fmin=20)
    refs = {round(r["f0"]): r for r in A.propose_references(f, db)}
    moving = [r for r in refs.values() if r["kind"] == "moving"]
    es = [r["f_obs"] / r["f0"] - 1 for r in moving]
    e = A.combine(es)
    on, off = A.thresholds(e)
    events = A.detect_events(t, e, on, off, roll=0.15)
    profiles = [A.fit_profile(t, e, s, en, 0.15) for s, en in events]
    y, regions = correction.render(x, SR, t, A.ratio_curve(t, profiles), events)
    _, _, db2 = A.stft_db(y.mean(1), SR, N_FFT, HOP, fmin=20)
    e2 = A.combine([A.track(f, db2, r["center"])[0] / r["f0"] - 1 for r in moving])
    return dict(x=x, y=y, t=t, f=f, db=db, refs=refs, moving=moving, es=es, e=e, thr=(on, off), events=events,
                profiles=profiles, e2=e2, regions=regions, true_e=prof(t) - 1)


def test_tracking_accuracy_on_stable_tones():
    x = audio.synth(None)
    t, f, db = A.stft_db(x[:, 0], SR, N_FFT, HOP, fmin=20)
    for f_true in FREQS:
        f_obs, _ = A.track(f, db, f_true * 1.004)  # start slightly off: the tracker must find the tone
        err = np.abs(f_obs - f_true)
        assert np.isfinite(f_obs).mean() > 0.99
        assert np.nanmean(err) < 0.2 and np.nanmax(err) < 1.0, f_true


def test_f0_is_robust_to_the_anomaly():
    p = pipeline("gaussian")
    for f_true in FREQS:
        assert abs(p["refs"][f_true]["f0"] - f_true) < 0.5


def test_references_are_classified():
    p = pipeline("gaussian")
    assert {round(r["f0"]) for r in p["moving"]} == set(FREQS)
    assert p["refs"][60]["kind"] == "fixed"  # the additive machine hum does not follow the deviation


@pytest.mark.parametrize("name, n_events", [("step", 1), ("gaussian", 1), ("ramp", 1), ("multi", 3)])
def test_detection_profile_and_correction(name, n_events):
    p = pipeline(name)
    t, e, true_e = p["t"], p["e"], p["true_e"]
    assert len(p["events"]) == n_events
    # every frame where the true deviation is clearly above the release threshold is inside an event
    inside = np.zeros(len(t), bool)
    for s, en in p["events"]:
        inside |= (t >= s) & (t <= en)
    assert inside[np.abs(true_e) > 2 * p["thr"][1]].all()
    # edges: event bounds (minus the 150 ms roll) within a few hops of where |true e| crosses release
    for s, en in p["events"]:
        m = (t >= s) & (t <= en) & (np.abs(true_e) > p["thr"][1])
        assert abs(t[m][0] - (s + 0.15)) < 4 * HOP / SR + 0.05
        assert abs(t[m][-1] - (en - 0.15)) < 4 * HOP / SR + 0.05
    # profile: RMS error below 10 % of the peak deviation
    for tt, ee in p["profiles"]:
        ref = np.interp(tt, t, true_e)
        assert np.sqrt(np.mean((ee - ref) ** 2)) < 0.1 * np.abs(ref).max()
    # correction: residual deviation on the references reduced by at least 10 dB (doc §7.3)
    before, after, red = A.residual(t, e, p["e2"], p["events"])
    assert red > 10, (before, after)


def test_duration_and_integrity_stereo():
    p = pipeline("step", channels=2)
    x, y = p["x"], p["y"]
    assert y.shape == x.shape and y.dtype == x.dtype
    assert np.isfinite(y).all()
    # outside the rewritten regions the signal is bit-identical; inside, both channels were corrected
    untouched = np.ones(len(x), bool)
    for a, b in p["regions"]:
        untouched[a:b] = False
    assert np.array_equal(x[untouched], y[untouched])
    assert np.abs(y[~untouched] - x[~untouched]).max(axis=0).min() > 1e-3
    # stereo image: channel 1 is 0.7 * channel 0 (+ independent noise) and must stay so
    a, b = p["regions"][0]
    ratio = np.sqrt(np.mean(y[a:b, 1] ** 2) / np.mean(y[a:b, 0] ** 2))
    assert abs(ratio - 0.7) < 0.01


def test_regions_merge_and_render_is_deterministic():
    x = audio.synth(audio.PROFILES["multi"], dur=20.0)
    t = np.linspace(0, 20, 2000)
    r = audio.PROFILES["multi"](t)
    spans = [(2.5, 6.5), (8, 12), (13.5, 17.5)]
    y1, regions = correction.render(x, SR, t, r, spans, tail_s=0.5)
    y2, _ = correction.render(x, SR, t, r, spans, tail_s=0.5)
    assert len(regions) == 3 and np.array_equal(y1, y2)


def test_non_global_deviation_is_flagged():
    """Only the 1500 Hz component deviates: not a global speed change (doc §6, §7.1)."""
    p = pipeline("gaussian", deviating=(2,))
    assert [round(r["f0"]) for r in p["moving"]] == [1500]
    s, en = p["events"][0]
    status, _ = A.coherence(p["t"], p["es"], ["1500"], p["e"], s, en, p["thr"][1])
    assert status == "unverified"  # one moving reference only ...
    every = [r["f_obs"] / r["f0"] - 1 for r in p["refs"].values()]
    status, msg = A.coherence(p["t"], every, list(map(str, p["refs"])), p["e"], s, en, p["thr"][1])
    assert status == "incoherent"  # ... and the other components contradict it
    assert "1500" in msg and "440" in msg


def test_export_roundtrip(tmp_path):
    x = audio.synth(None, dur=1.0, channels=2) * 6  # deliberately clipping
    clipped = audio.export(tmp_path / "out.wav", x, SR, "24")
    y, sr = audio.load(tmp_path / "out.wav")
    assert clipped > 0 and sr == SR and y.shape == x.shape and np.abs(y).max() <= 1.0
    assert audio.export(tmp_path / "out.flac", x / 8, SR, "16") == 0
    assert sf.info(str(tmp_path / "out.flac")).frames == len(x)


def test_store_roundtrip(tmp_path):
    st = Store(tmp_path)
    state = dict(params=dict(k_on=4.0), selection=[1.0, 2.0], tracks=[dict(f0=440.0)], events=[])
    pid = st.create_project("demo", tmp_path / "a.wav", "abc", SR, 2, 1000, state)
    state["selection"] = None
    st.save_state(pid, state)
    st = Store(tmp_path)  # reopen: schema migration must be a no-op
    row = st.project(pid)
    assert row["name"] == "demo" and '"selection": null' in row["state_json"]
    key = cache_key("abc", np.arange(3), dict(a=1))
    assert key == cache_key("abc", np.arange(3), dict(a=1)) != cache_key("abc", np.arange(4), dict(a=1))
    st.save_arrays(key, a=np.arange(5))
    assert np.array_equal(st.load_arrays(key)["a"], np.arange(5))
    c1 = st.add_correction(pid, key, {}, st.cache_path(key))
    c2 = st.add_correction(pid, key, {}, st.cache_path(key))
    assert st.active_correction(pid)["id"] == c2 != c1
    st.add_export(pid, c2, tmp_path / "o.wav", "wav", SR, "24")
    st.delete_project(pid)
    assert st.project(pid) is None and not st.db.execute("SELECT * FROM exports").fetchall()
