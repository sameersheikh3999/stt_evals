# -*- coding: utf-8 -*-
"""
Score a PWR_ENG reading by comparing SOUNDS, not spellings.

Deepgram transcribes the audio into words; g2p_en turns both the transcript
words and the 50 gold pseudowords into ARPAbet phonemes; each gold word is
scored by cosine similarity with the transcript word aligned to it.

Why sounds: a word ASR writes "maz" as "Mars" and "tob" as "top". Spelled out
these share little; as phonemes (M AA Z vs M AA R Z) they are close.

Vectors are counts of phoneme unigrams + bigrams with word-boundary markers,
stress digits dropped. Rough guide for a 3-phoneme word:
  identical -> 1.00,  one extra phoneme -> ~0.76,  one phoneme swapped -> ~0.57

Usage: .venv/Scripts/python g2p_cosine.py [deepgram.json] [threshold]
  defaults: transcripts/json/sample_audio.json  0.8
"""
import collections, csv, json, math, re, sys, pathlib

from g2p_en import G2p

ROOT = pathlib.Path(__file__).parent
SRC = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "transcripts" / "json" / "sample_audio.json"
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
GOLD = ("maz ver lut paf nom yod fut et zib mib dag fol san leb huz teb ved bif lef vom "
        "ret nep riz lus rop dit nup kad hig yag wix tob tib gax jod nad gof sig ral reg "
        "tup fim peb sen kib sim fid sal zon tat").split()

g2p = G2p()


def phones(word):
    return [re.sub(r"\d", "", p) for p in g2p(word) if p.strip() and p[0].isalpha()]


def vec(ph):
    seq = ["<"] + ph + [">"]
    v = collections.Counter(ph)
    v.update(a + "_" + b for a, b in zip(seq, seq[1:]))
    return v


def cosine(a, b):
    va, vb = vec(a), vec(b)
    dot = sum(va[k] * vb[k] for k in va)
    na, nb = math.sqrt(sum(x * x for x in va.values())), math.sqrt(sum(x * x for x in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def align(gp, hp):
    """Word alignment maximising total similarity (gaps free), so a skipped or
    extra word shifts nothing after it. Returns hyp index (or None) per gold word."""
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
    return out


words = json.loads(SRC.read_text(encoding="utf-8"))["results"]["channels"][0]["alternatives"][0]["words"]
hyp = [w["word"] for w in words]
gp = [phones(w) for w in GOLD]
hp = [phones(w) for w in hyp]
match = align(gp, hp)

rows = []
for i, (g, j) in enumerate(zip(GOLD, match)):
    sim = cosine(gp[i], hp[j]) if j is not None else 0.0
    rows.append({"item": i + 1, "gold_word": g, "gold_phones": " ".join(gp[i]),
                 "heard_word": hyp[j] if j is not None else "", "heard_phones": " ".join(hp[j]) if j is not None else "",
                 "cosine": round(sim, 3), "correct": int(sim >= THR),
                 "start": words[j]["start"] if j is not None else None})

for r in rows:
    print("%2d %-4s %-10s  heard %-7s %-12s  cos %.2f  %s" % (
        r["item"], r["gold_word"], r["gold_phones"], r["heard_word"] or "-", r["heard_phones"],
        r["cosine"], "correct" if r["correct"] else "WRONG"))
exact = sum(r["gold_word"] == re.sub(r"[^a-z]", "", r["heard_word"].lower()) for r in rows)
print("\n%s: %d gold words, %d transcript words" % (SRC.name, len(GOLD), len(hyp)))
print("correct at cosine >= %.2f : %d / %d" % (THR, sum(r["correct"] for r in rows), len(rows)))
for t in (1.0, 0.9, 0.8, 0.7, 0.6):
    print("  at >= %.1f : %d" % (t, sum(r["cosine"] >= t - 1e-9 for r in rows)))
print("  exact spelling match (for comparison): %d" % exact)

out = SRC.parent.parent / "txt" / (SRC.stem + "_g2p_cosine.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("-> %s" % out.relative_to(ROOT))
