# -*- coding: utf-8 -*-
"""
Compare SpeechAce and OpenPronounce against the enumerator on PWR_ENG (English
pseudowords) for a few session recordings.

Per child: cut the PWR_ENG window from the session m4a (manifest offsets + PAD),
score it against the words the child attempted, and compare each word's verdict
with the enumerator's tap (gold_items_long.csv).

SpeechAce accepts at most 30 s, so windows longer than that are split into
overlapping chunks; every chunk is scored against the full attempted text and a
word keeps its best score over chunks (a word read in chunk 1 scores low in chunk 2).
OpenPronounce takes the whole window.

Usage: .venv/Scripts/python eval_pwr_speechace_openpronounce.py uuid8 [uuid8 ...]
Env: SA_THR (SpeechAce "correct" cut-off, default 80), PAD (s, default 2),
     PREPROC (none | gain | clean, see preproc.py; default none)
Outputs: eval_pwr_eng/sa_op[_<preproc>]/<uuid8>.json and per_item.csv
"""
import csv, json, os, re, sys, time, wave, pathlib, urllib.error, urllib.parse, urllib.request, uuid, collections

import espeakng_loader

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
os.environ.setdefault("OPENPRONOUNCE_DEVICE", "cpu")   # GPU driver here is too old for this torch

import av, numpy as np

import preproc

ROOT = pathlib.Path(__file__).parent
AUDIO_DIR = ROOT / "EGMA audio files - path-20260918T004653Z-1-001"
PREPROC = os.environ.get("PREPROC", "none")
OUT = ROOT / "eval_pwr_eng" / ("sa_op" if PREPROC == "none" else "sa_op_" + PREPROC)
OUT.mkdir(parents=True, exist_ok=True)
SA_URL = "https://api5.speechace.com/api/scoring/text/v9/json"   # the key's region
SA_THR = float(os.environ.get("SA_THR", 80))
PAD = float(os.environ.get("PAD", 2.0))
SR = 16000
CHUNK, HOP = 29.0, 20.0       # SpeechAce max is 30 s; chunks overlap by 9 s

GOLD = ("maz ver lut paf nom yod fut et zib mib dag fol san leb huz teb ved bif lef vom "
        "ret nep riz lus rop dit nup kad hig yag wix tob tib gax jod nad gof sig ral reg "
        "tup fim peb sen kib sim fid sal zon tat").split()


def env_key(name):
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("%s missing from .env" % name)


def load_window(path, t0, t1):
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        c.seek(int(max(t0 - 1, 0) / s.time_base), stream=s, backward=True)
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks, first = [], None
        for fr in c.decode(s):
            if fr.time is None:
                continue
            first = fr.time if first is None else first
            if fr.time > t1:
                break
            chunks += [o.to_ndarray().reshape(-1) for o in rs.resample(fr)]
    x = np.concatenate(chunks)
    return x[max(0, int((t0 - first) * SR)):int((t1 - first) * SR)]


def wav_bytes(x):
    import io
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()


def speechace(x, text, key):
    """POST one <=30 s clip; returns SpeechAce's JSON."""
    bnd = uuid.uuid4().hex
    body = b"".join([
        b"--%s\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n%s\r\n" % (bnd.encode(), text.encode()),
        b"--%s\r\nContent-Disposition: form-data; name=\"user_audio_file\"; filename=\"a.wav\"\r\n"
        b"Content-Type: audio/wav\r\n\r\n" % bnd.encode(), wav_bytes(x), b"\r\n--%s--\r\n" % bnd.encode()])
    url = SA_URL + "?" + urllib.parse.urlencode({"key": key, "dialect": "en-us", "user_id": "stt_evals"})
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
    for attempt in range(3):          # retry network drops; a failed upload is not billed
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 2:
                raise
            print("  SpeechAce network error (%s), retrying in 30 s" % e, flush=True)
            time.sleep(30)


def speechace_scores(x, words, key):
    """best SpeechAce quality score per word position over overlapping chunks"""
    text = " ".join(words)
    starts = [0.0] if len(x) <= 30 * SR else list(np.arange(0, len(x) / SR - CHUNK + HOP, HOP))
    best, raw = [0.0] * len(words), []
    for s in starts:
        seg = x[int(s * SR):int((s + CHUNK) * SR)]
        j = speechace(seg, text, key)
        raw.append({"chunk_start": float(s), "response": j})
        if j.get("short_message") == "error_no_speech":
            continue                    # a chunk of silence (e.g. after the child finished)
        if j.get("status") != "success":
            raise SystemExit("SpeechAce error: %s" % j)
        for i, w in enumerate(j["text_score"]["word_score_list"][:len(words)]):
            best[i] = max(best[i], w["quality_score"])
    return best, raw


# ---------------------------------------------------------------- data
man = {r["KEY"]: r for r in csv.DictReader(open(ROOT / "eval_manifest.csv", encoding="utf-8-sig"))
       if r["task_id"] == "PWR_ENG"}
lab = collections.defaultdict(dict)
for r in csv.DictReader(open(ROOT / "gold_items_long.csv", encoding="utf-8-sig")):
    if r["task_id"] == "PWR_ENG":
        lab[r["KEY"]][int(r["spoken_position"])] = int(r["is_correct"])
audio = {}
for p in sorted(AUDIO_DIR.rglob("*.m4a")):
    m = re.search(r"AA_([0-9a-f-]{36})_enumerator", p.name)
    audio.setdefault(m.group(1)[:8], ("uuid:" + m.group(1), p))

try:
    from openpronounce import compare_audio_with_text
except (OSError, ImportError) as e:   # e.g. Windows Smart App Control blocking torch/scipy DLLs
    print("OpenPronounce unavailable, running SpeechAce only: %s" % str(e).splitlines()[0])
    compare_audio_with_text = None
key = env_key("speechace_api_key")
rows = []
for u8 in sys.argv[1:] or ["52be51a8", "7e464fe2", "d99c7b13"]:
    k, path = audio[u8]
    m = man[k]
    n = int(float(m["attempted"]))
    words = GOLD[:n]
    x = load_window(path, max(float(m["offset_start_sec"]) - PAD, 0), float(m["offset_end_sec"]) + PAD)
    x = preproc.apply(x, PREPROC).astype(np.float32)
    print("\n== %s  %d words attempted, window %.0fs" % (u8, n, len(x) / SR), flush=True)

    prev = OUT / (u8 + ".json")
    saved = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    if saved.get("speechace"):
        # reuse saved SpeechAce responses: re-scoring the same clip only spends quota
        sa_raw = saved["speechace"]
        sa = [max([c["response"]["text_score"]["word_score_list"][i]["quality_score"] for c in sa_raw
                   if c["response"].get("status") == "success"] or [0.0])
              for i in range(len(words))]
    else:
        sa, sa_raw = speechace_scores(x, words, key)
    op = saved.get("openpronounce")                    # reuse: OpenPronounce takes minutes on CPU
    if not op and compare_audio_with_text:
        op = compare_audio_with_text(x, " ".join(words))
    if op:
        op.pop("prosody", None)
    # OpenPronounce reports errors by word text; attempted words are unique, so text identifies position
    op_err = {e["word"]: e for e in op["differences"]["errors"]} if op else {}
    (OUT / (u8 + ".json")).write_text(json.dumps({"speechace": sa_raw, "openpronounce": op}, ensure_ascii=False,
                                                 indent=1, default=str), encoding="utf-8")
    for i, w in enumerate(words, 1):
        e = op_err.get(w)
        rows.append({"uuid": u8, "item": i, "word": w, "enumerator": lab[k].get(i),
                     "speechace_score": sa[i - 1], "speechace_ok": int(sa[i - 1] >= SA_THR),
                     "openpronounce_ok": int(e is None) if op else "",
                     "op_expected": e.get("expected", "") if e else "", "op_heard": e.get("actual", "") if e else ""})
    print("  enumerator %d/%d correct | SpeechAce %d | OpenPronounce %d   (OP score %s, transcript: %s)" % (
        sum(r["enumerator"] for r in rows if r["uuid"] == u8), n,
        sum(r["speechace_ok"] for r in rows if r["uuid"] == u8),
        sum(r["openpronounce_ok"] or 0 for r in rows if r["uuid"] == u8),
        op["score"] if op else "-", op["transcribe"][:120] if op else "-"))

with open(OUT / "per_item.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def stats(h, p):
    tp = sum(a == 0 and b == 0 for a, b in zip(h, p)); tn = sum(a == 1 and b == 1 for a, b in zip(h, p))
    fp = sum(a == 1 and b == 0 for a, b in zip(h, p)); fn = sum(a == 0 and b == 1 for a, b in zip(h, p))
    N = len(h); po = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / N / N
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return po, (po - pe) / (1 - pe) if pe < 1 else 0.0, rec, spec


def auc(score, h):
    pos = [s for s, y in zip(score, h) if y == 1]; neg = [s for s, y in zip(score, h) if y == 0]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / len(pos) / len(neg)


h = [r["enumerator"] for r in rows]
print("\nPOOLED %d items, enumerator: %d correct / %d wrong" % (len(h), sum(h), len(h) - sum(h)))
print("  %-24s %6s %6s %12s %12s" % ("", "agree", "kappa", "catches-wrong", "passes-right"))
for name, col in (("SpeechAce (>= %.0f)" % SA_THR, "speechace_ok"), ("OpenPronounce", "openpronounce_ok")):
    if rows[0][col] == "":
        print("  %-24s (not run)" % name)
        continue
    po, k, rec, spec = stats(h, [r[col] for r in rows])
    print("  %-24s %6.2f %6.2f %12.2f %12.2f" % (name, po, k, rec, spec))
print("  SpeechAce score AUC (threshold-free): %.2f" % auc([r["speechace_score"] for r in rows], h))
print("-> %s" % (OUT / "per_item.csv").relative_to(ROOT))
