# -*- coding: utf-8 -*-
"""
Audio clean-up for the session recordings, NumPy only (no scipy: Windows Smart App
Control on this machine blocks scipy/torch DLLs at times).

  gain(x)      loudness to a fixed level (-20 dBFS RMS over speech frames) + soft limiter
  highpass(x)  remove rumble / handling noise below 80 Hz
  denoise(x)   spectral subtraction against a noise profile taken from the quietest
               frames of the clip (classroom hum, fan, hiss)

Variants used by the evals: "none", "gain", "clean" (= highpass -> denoise -> gain).
Note: wav2vec2 models normalise every input to zero mean / unit variance, so pure
gain is close to a no-op for them; it can matter for services like SpeechAce.
"""
import numpy as np

SR = 16000


def _frames_db(x, n):
    fr = x[:len(x) // n * n].reshape(-1, n)
    return 10 * np.log10((fr ** 2).mean(1) + 1e-12)


def gain(x, target_db=-20.0, sr=SR):
    """Scale so the loudest 30% of 30 ms frames (the speech) average target_db,
    then soft-limit peaks with tanh so nothing clips."""
    e = _frames_db(x, int(0.03 * sr))
    speech = e[e >= np.percentile(e, 70)]
    level = 10 * np.log10(np.mean(10 ** (speech / 10)))
    y = x * 10 ** ((target_db - level) / 20)
    return np.tanh(y / 0.95) * 0.95


def highpass(x, cutoff=80.0, sr=SR):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X *= 1 / (1 + (cutoff / np.maximum(f, 1e-3)) ** 8) ** 0.5     # 4th-order Butterworth magnitude
    return np.fft.irfft(X, n=len(x))


def _stft(x, n=512, hop=128):
    w = np.hanning(n)
    pad = np.pad(x, (n, n))
    idx = np.arange(0, len(pad) - n, hop)
    return np.fft.rfft(np.stack([pad[i:i + n] * w for i in idx]), axis=1), w, idx, len(pad)


def _istft(S, w, idx, total, n=512):
    out, norm = np.zeros(total), np.zeros(total)
    fr = np.fft.irfft(S, n=n, axis=1)
    for k, i in enumerate(idx):
        out[i:i + n] += fr[k] * w
        norm[i:i + n] += w ** 2
    return (out / np.maximum(norm, 1e-8))[n:total - n]


def denoise(x, strength=1.5, floor=0.1):
    """Spectral subtraction. Noise profile = per-frequency 15th percentile of magnitude
    (the level present even between words). Gains are smoothed over time to limit
    'musical noise' artefacts; floor keeps some residual so speech onsets aren't cut."""
    S, w, idx, total = _stft(x)
    mag = np.abs(S)
    noise = np.percentile(mag, 15, axis=0, keepdims=True)
    g = np.clip(1 - strength * noise / np.maximum(mag, 1e-10), floor, 1.0)
    k = np.ones(5) / 5                                   # ~40 ms smoothing along time
    g = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, g)
    return _istft(S * g, w, idx, total)[:len(x)]


def apply(x, variant):
    if variant in ("", "none", None):
        return x
    if variant == "gain":
        return gain(x)
    if variant == "clean":
        return gain(denoise(highpass(x)))
    raise ValueError("unknown PREPROC variant %r" % variant)
