# -*- coding: utf-8 -*-
"""
Transcribe an audio file with facebook/wav2vec2-large-960h (English character CTC,
no language model) and save it in Deepgram's JSON shape, so g2p_cosine.py can
score it the same way.

Usage: .venv/Scripts/python transcribe_w2v_960h.py [audio]    (default sample_audio.ogg)
  -> transcripts/json/<stem>_w2v960h.json, transcripts/txt/<stem>_w2v960h.txt
"""
import json, sys, pathlib
import av, numpy as np
from transformers import pipeline

ROOT = pathlib.Path(__file__).parent
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_audio.ogg"
SR = 16000

# no ffmpeg on this machine, so decode with PyAV and hand the pipeline an array
with av.open(str(SRC)) as c:
    rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
    x = np.concatenate([o.to_ndarray().reshape(-1) for f in c.decode(audio=0) for o in rs.resample(f)])

# cpu: the GPU driver here (528, CUDA 12.0) is too old for the installed torch build
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h", device="cpu")
res = asr({"raw": x, "sampling_rate": SR}, return_timestamps="word", chunk_length_s=30, stride_length_s=5)
text = res["text"].lower()
words = [{"word": ch["text"].lower(), "start": ch["timestamp"][0], "end": ch["timestamp"][1]}
         for ch in res["chunks"]]

stem = SRC.stem + "_w2v960h"
(ROOT / "transcripts" / "json").mkdir(parents=True, exist_ok=True)
(ROOT / "transcripts" / "txt").mkdir(parents=True, exist_ok=True)
(ROOT / "transcripts" / "json" / (stem + ".json")).write_text(json.dumps(
    {"model": "facebook/wav2vec2-large-960h",
     "results": {"channels": [{"alternatives": [{"transcript": text, "words": words}]}]}},
    ensure_ascii=False, indent=1), encoding="utf-8")
(ROOT / "transcripts" / "txt" / (stem + ".txt")).write_text(text + "\n", encoding="utf-8")
print(text)
print("-> transcripts/json/%s.json, transcripts/txt/%s.txt" % (stem, stem))
