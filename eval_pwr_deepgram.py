# -*- coding: utf-8 -*-
"""
Deepgram nova-3 (English) on an EGRA English reading subtask, original vs cleaned audio,
(TASK=PWR_ENG pseudowords, the default, or TASK=FWR_ENG familiar words)
scored against the enumerator.

Per child: cut the PWR_ENG window (as eval_pwr_speechace_openpronounce.py does),
optionally clean it (preproc.py), transcribe with Deepgram, align the attempted
pseudowords to the transcript, and judge each word two ways:
  exact   transcript word spelled exactly like the pseudoword
  sound   g2p_en phonemes of both, cosine similarity >= 0.8 (as g2p_cosine.py)

Usage: .venv/Scripts/python eval_pwr_deepgram.py uuid8 [uuid8 ...]
Transcripts cached in eval_<task>/deepgram/ and eval_<task>/deepgram_clean/.
  -> eval_<task>/deepgram_per_item.csv
"""
import collections, csv, io, json, math, os, re, sys, time, wave, pathlib, urllib.error, urllib.request

import av, numpy as np
from g2p_en import G2p

import preproc

ROOT = pathlib.Path(__file__).parent
AUDIO_DIR = ROOT / "EGMA audio files - path-20260918T004653Z-1-001"
DG_URL = "https://api.deepgram.com/v1/listen?model=nova-3&language=en&punctuate=false"
SR, PAD, THR = 16000, 2.0, 0.8
VARIANTS = (("original", "none", "deepgram"), ("cleaned", "clean", "deepgram_clean"))
TASK = os.environ.get("TASK", "PWR_ENG")            # PWR_ENG (pseudowords) or FWR_ENG (familiar words)
OUTBASE = ROOT / ("eval_" + TASK.lower())
import openpyxl
GOLD = [str(r[5]).lower() for r in openpyxl.load_workbook(ROOT / "EGRA_STT_eval_key.xlsx", read_only=True)
        ["Item_Key"].iter_rows(values_only=True) if r[0] == TASK]


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
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()


def deepgram(x, key):
    req = urllib.request.Request(DG_URL, data=wav_bytes(x),
                                 headers={"Authorization": "Token " + key, "Content-Type": "audio/wav"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 2:
                raise
            print("  Deepgram network error (%s), retrying in 30 s" % e, flush=True)
            time.sleep(30)


# ---------------------------------------------------------------- sound match (as g2p_cosine.py)
g2p = G2p()
_ph = {}


def phones(word):
    if word not in _ph:
        _ph[word] = [re.sub(r"\d", "", p) for p in g2p(word) if p.strip() and p[0].isalpha()]
    return _ph[word]


def cosine(a, b):
    def vec(ph):
        seq = ["<"] + ph + [">"]
        v = collections.Counter(ph)
        v.update(x + "_" + y for x, y in zip(seq, seq[1:]))
        return v
    va, vb = vec(a), vec(b)
    dot = sum(va[k] * vb[k] for k in va)
    na, nb = math.sqrt(sum(x * x for x in va.values())), math.sqrt(sum(x * x for x in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def align(gp, hp):
    """gold word -> transcript index (or None), maximising total similarity in order."""
    n, m = len(gp), len(hp)
    S = [[cosine(g, h) for h in hp] for g in gp]
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = max(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1] + S[i - 1][j - 1])
    out, i, j = [None] * n, n, m
    while i and j:
        if d[i][j] == d[i - 1][j - 1] + S[i - 1][j - 1]:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return out, S


# ---------------------------------------------------------------- run
man = {r["KEY"]: r for r in csv.DictReader(open(ROOT / "eval_manifest.csv", encoding="utf-8-sig"))
       if r["task_id"] == TASK}
lab = collections.defaultdict(dict)
for r in csv.DictReader(open(ROOT / "gold_items_long.csv", encoding="utf-8-sig")):
    if r["task_id"] == TASK:
        lab[r["KEY"]][int(r["spoken_position"])] = int(r["is_correct"])
audio = {}
for p in sorted(AUDIO_DIR.rglob("*.m4a")):
    m = re.search(r"AA_([0-9a-f-]{36})_enumerator", p.name)
    audio.setdefault(m.group(1)[:8], ("uuid:" + m.group(1), p))

key = env_key("deepgram_api_key")
rows = []
for u8 in sys.argv[1:]:
    k, path = audio[u8]
    m = man[k]
    n = int(float(m["attempted"]))
    words = GOLD[:n]
    raw = load_window(path, max(float(m["offset_start_sec"]) - PAD, 0), float(m["offset_end_sec"]) + PAD)
    per = {}
    for vname, pre, d in VARIANTS:
        f = OUTBASE / d / (u8 + ".json")
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            f.write_text(json.dumps(deepgram(preproc.apply(raw, pre).astype(np.float32), key), ensure_ascii=False),
                         encoding="utf-8")
        alt = json.loads(f.read_text(encoding="utf-8"))["results"]["channels"][0]["alternatives"][0]
        hyp = [re.sub(r"[^a-z']", "", w["word"].lower()) for w in alt["words"]]
        hyp = [h for h in hyp if h]
        match, S = align([phones(w) for w in words], [phones(h) for h in hyp])
        per[vname] = (hyp, match, S, alt.get("confidence", 0))
    for i, w in enumerate(words):
        row = {"uuid": u8, "grade": m["class_grade_a"][:1], "item": i + 1, "word": w, "enumerator": lab[k].get(i + 1)}
        for vname, (hyp, match, S, _) in per.items():
            j = match[i]
            row["heard_" + vname] = hyp[j] if j is not None else ""
            row["cos_" + vname] = round(S[i][j], 3) if j is not None else 0.0
            row["exact_" + vname] = int(j is not None and hyp[j] == w)
            row["sound_" + vname] = int(j is not None and S[i][j] >= THR)
        rows.append(row)
    print("%s grade %s  enumerator %2d/%-2d | exact orig/clean %2d/%-2d | sound orig/clean %2d/%-2d | words heard %d/%d"
          % (u8, m["class_grade_a"][:1], sum(lab[k].get(i, 0) for i in range(1, n + 1)), n,
             sum(r["exact_original"] for r in rows if r["uuid"] == u8), sum(r["exact_cleaned"] for r in rows if r["uuid"] == u8),
             sum(r["sound_original"] for r in rows if r["uuid"] == u8), sum(r["sound_cleaned"] for r in rows if r["uuid"] == u8),
             len(per["original"][0]), len(per["cleaned"][0])), flush=True)


def stats(h, p):
    tp = sum(a == 0 and b == 0 for a, b in zip(h, p)); tn = sum(a == 1 and b == 1 for a, b in zip(h, p))
    fp = sum(a == 1 and b == 0 for a, b in zip(h, p)); fn = sum(a == 0 and b == 1 for a, b in zip(h, p))
    N = len(h); po = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / N / N
    return po, (po - pe) / (1 - pe) if pe < 1 else 0.0, tn / (tn + fp), tp / (tp + fn)


def auc(score, h):
    pos = [s for s, y in zip(score, h) if y == 1]; neg = [s for s, y in zip(score, h) if y == 0]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / len(pos) / len(neg)


h = [r["enumerator"] for r in rows]
print("\n%d children, %d words; enumerator %d correct / %d wrong" % (len(set(r["uuid"] for r in rows)), len(h), sum(h), len(h) - sum(h)))
print("  %-24s %6s %6s %13s %13s" % ("", "agree", "kappa", "passes-right", "catches-wrong"))
for v in ("original", "cleaned"):
    for how in ("exact", "sound"):
        po, k, pr, cw = stats(h, [r["%s_%s" % (how, v)] for r in rows])
        print("  %-24s %6.2f %6.2f %13.2f %13.2f" % ("%s, %s match" % (v, how), po, k, pr, cw))
    print("  %-24s AUC of cosine %.2f" % (v, auc([r["cos_" + v] for r in rows], h)))
print("  (kappa 0 = chance)")
out = OUTBASE / "deepgram_per_item.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("-> %s" % out.relative_to(ROOT))
