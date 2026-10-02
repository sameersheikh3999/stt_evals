# -*- coding: utf-8 -*-
"""
Score a PWR_ENG reading with OpenPronounce (pip install openpronounce).

OpenPronounce combines wav2vec2-large-960h (words), wav2vec2-lv-60-espeak-cv-ft
(phones) and a DTW distance to a gTTS reference reading of the expected text.
Note: gTTS sends the TEXT (not the audio) to Google to synthesise the reference.

espeak-ng (needed by phonemizer) comes from the espeakng-loader wheel, so no
system install is required on Windows.

Usage: .venv/Scripts/python run_openpronounce.py [audio]   (default sample_audio.ogg)
  -> transcripts/json/<stem>_openpronounce.json
"""
import json, os, sys, pathlib

import espeakng_loader

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
os.environ.setdefault("OPENPRONOUNCE_DEVICE", "cpu")   # GPU driver here is too old for this torch

import av, numpy as np
from openpronounce import compare_audio_with_text

ROOT = pathlib.Path(__file__).parent
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_audio.ogg"
GOLD = ("maz ver lut paf nom yod fut et zib mib dag fol san leb huz teb ved bif lef vom "
        "ret nep riz lus rop dit nup kad hig yag wix tob tib gax jod nad gof sig ral reg "
        "tup fim peb sen kib sim fid sal zon tat")

# decode with PyAV (no ffmpeg on this machine)
with av.open(str(SRC)) as c:
    rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
    x = np.concatenate([o.to_ndarray().reshape(-1) for f in c.decode(audio=0) for o in rs.resample(f)])

res = compare_audio_with_text(x, GOLD)
res.pop("prosody", None)
out = ROOT / "transcripts" / "json" / (SRC.stem + "_openpronounce.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

d = res["differences"]
bad = {e["word"] for e in d["errors"]}
print("score          : %s/100" % res["score"])
print("transcription  : %s" % res["transcribe"])
print("word error rate: %s   phone error rate: %s" % (d.get("word_error_rate"), d.get("phoneme_error_rate")))
print("\n%-4s %-10s %-12s %s" % ("word", "expected", "heard", "confidence wrong"))
for e in d["errors"]:
    print("%-4s /%s/  /%s/  %s" % (e["word"], e.get("expected", ""), e.get("actual", ""), e.get("confidence")))
ok = [w for w in GOLD.split() if w not in bad]
print("\ncorrect: %d / 50   (flagged: %d)" % (len(ok), len(bad)))
print("correct words:", " ".join(ok))
print("-> %s" % out.relative_to(ROOT))
