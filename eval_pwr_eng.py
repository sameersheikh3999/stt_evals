# -*- coding: utf-8 -*-
"""
STT evaluation for EGRA PWR_ENG (English pseudoword / non-word decoding = phonics).

Pseudowords (maz, ver, lut ...) are not in any word ASR's vocabulary, so a word
model snaps them to real words. Instead we use a wav2vec2 PHONEME recogniser
(CTC over espeak IPA, no language model) and score each item on its phonemes.

The audio is 8 kHz AAC from a phone-grade session recorder, far-field, children,
classroom noise. On it, free (greedy) phoneme decoding is close to noise: >97% of
CTC frames are blank. The acoustic evidence is still there - the true pseudoword
sequence gets a far better CTC likelihood than shuffled orderings - so the main
scorer is CONSTRAINED, as the eval key's protocol suggests:

  GOP  forced-align the expected phonemes of items 1..attempted to the window with
       a Viterbi pass over CTC states, with a "filler" state between items that
       absorbs enumerator speech ('chalein', 'aagay') at a per-frame penalty.
       Per phone: GOP = mean over its frames of  log p(target class) - log max p.
       Item score = its worst phone's GOP. Low score -> predicted "incorrect".
       The threshold is cross-fitted (chosen on one half of the children, applied
       to the other), so the reported P/R/kappa are out-of-sample. AUC needs none.

  GREEDY  argmax CTC -> phonemes -> infix alignment -> item correct iff its
       phonemes all match. Kept as the naive baseline.

As with ORF_URD, the attempted cut-off is taken from the human, so this scores
per-item judgement only, not how far the child got.

Run with the project venv:  .venv/Scripts/python eval_pwr_eng.py
Env: MODEL (default xlsr-53-espeak-cv-ft), PAD (s, default 3), FILLER_PEN (default 1.0),
     LIMIT (first N children), DEVICE=cuda (default cpu)
Log-probs are cached per model under eval_pwr_eng/<model>/, so rescoring is fast.
"""
import csv, json, os, re, math, pathlib, collections
import numpy as np

ROOT = pathlib.Path(__file__).parent
TASK = "PWR_ENG"
MODEL = os.environ.get("MODEL", "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
PAD = float(os.environ.get("PAD", 3.0))
FILLER_PEN = float(os.environ.get("FILLER_PEN", 1.0))
LIMIT = int(os.environ.get("LIMIT", 0))
SR = 16000
FRAME = 0.02                                       # one CTC frame = 20 ms
OUT = ROOT / "eval_pwr_eng"
CACHE = OUT / MODEL.split("/")[-1]
CACHE.mkdir(parents=True, exist_ok=True)
AUDIO_DIRS = [ROOT / "audios_selected", ROOT / "EGMA audio files - path-20260918T004653Z-1-001"]

# ---------------------------------------------------------------- reference phonemes
# Hand-written, not espeak: espeak's letter-to-sound rules guess at non-words.
# EGRA pseudowords are decoded with short vowels: a=/æ/ e=/ɛ/ i=/ɪ/ o=/ɒ/ u=/ʌ/.
LETTER = {"a": "æ", "e": "ɛ", "i": "ɪ", "o": "ɒ", "u": "ʌ",
          "b": "b", "d": "d", "f": "f", "g": "ɡ", "h": "h", "j": "dʒ", "k": "k",
          "l": "l", "m": "m", "n": "n", "p": "p", "r": "ɹ", "s": "s", "t": "t",
          "v": "v", "w": "w", "x": "k s", "y": "j", "z": "z"}
SPECIAL = {"ver": "v ɜː ɹ"}          # -er is r-coloured, not /ɛ/ + /r/


def ref_phones(word):
    if word in SPECIAL:
        return SPECIAL[word].split()
    return " ".join(LETTER[c] for c in word).split()


# Broad classes. The recogniser was trained on adult read speech (CommonVoice)
# and these are Pakistani children, so vowel quality and rhoticity vary a lot
# around each target. Collapse near-identical realisations; keep the five
# short-vowel contrasts that the task actually tests apart.
CLASS = {}
for _c, _members in {
    "A": "æ a ä",
    "E": "ɛ e eɪ ə ɜ",       # 'ver': /ɜː/ /ə/ /ɛ/ all accepted
    "I": "ɪ i ᵻ ɨ",
    "O": "ɒ ɑ ɔ o oʊ əʊ",
    "U": "ʌ ɐ ʊ u ʉ",
    "R": "ɹ r ɾ ɻ ɽ ʁ",
    "T": "t ʈ",
    "D": "d ɖ",
    "G": "ɡ g",
    "K": "k q c",
    "P": "p",
    "J": "dʒ ɟ dZ", "CH": "tʃ tS",
    "Y": "j",
    "V": "v ʋ w β",          # Urdu speakers merge /v/ and /w/
    "Z": "z", "S": "s", "H": "h ɦ", "F": "f", "B": "b", "M": "m",
    "N": "n ŋ", "L": "l ɫ ɭ",
    "ER": "ɚ ɝ",             # r-coloured vowel = vowel + r
}.items():
    for _p in _members.split():
        CLASS[_p] = _c
# the model's vocab (392 tokens, multilingual CommonVoice) carries tone digits,
# length, aspiration, palatalisation and nasal marks - none contrastive here
_STRIP = re.compile("[0-9ː:.\\^\\[\"ʲʰ̪̞̩̝̃̊]")


def cls(p):
    """IPA token -> list of broad classes. Multi-phone tokens ('ɑːɹ', 'ɛɹ') split."""
    s, out, i = _STRIP.sub("", p), [], 0
    while i < len(s):
        k = s[i:i + 2] if s[i:i + 2] in CLASS else s[i]
        c = CLASS.get(k, k)
        out += ["E", "R"] if c == "ER" else [c]
        i += len(k)
    return [c for j, c in enumerate(out) if j == 0 or c != out[j - 1]]  # 'ää', 'ɐɐ'


# ---------------------------------------------------------------- gold
import openpyxl

wb = openpyxl.load_workbook(ROOT / "EGRA_STT_eval_key.xlsx", read_only=True)
GOLD = [r[5] for r in wb["Item_Key"].iter_rows(values_only=True) if r[0] == TASK]
assert len(GOLD) == 50, len(GOLD)
GOLD_C = [[c for p in ref_phones(w) for c in cls(p)] for w in GOLD]

human = collections.defaultdict(dict)
with open(ROOT / "gold_items_long.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["task_id"] == TASK:
            human[r["KEY"]][int(float(r["spoken_position"]))] = int(r["is_correct"])

man = {}
with open(ROOT / "eval_manifest.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["task_id"] == TASK and r["offset_usable"] == "YES":
            man[r["KEY"]] = r

# full 36-char uuid join, as in eval_orf_urd.py; duplicate "Copy of ...(1)" files are identical
_UUID = re.compile(r"AA_([0-9a-fA-F-]{36})_enumerator")
AUDIO = {}
for d in AUDIO_DIRS:
    for p in sorted(d.rglob("*.m4a")):
        m = _UUID.search(p.name)
        if m:
            AUDIO.setdefault("uuid:" + m.group(1).lower(), p)
KEYS = sorted(k for k in AUDIO if k in man and human.get(k))
if LIMIT:
    KEYS = KEYS[:LIMIT]


def window(key):
    m = man[key]
    return max(float(m["offset_start_sec"]) - PAD, 0.0), float(m["offset_end_sec"]) + PAD


# ---------------------------------------------------------------- audio + model
def load_window(path, t0, t1):
    """Decode [t0, t1) seconds of an m4a to 16 kHz mono float32."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        c.seek(int(max(t0 - 1, 0) / s.time_base), stream=s, any_frame=False, backward=True)
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks, first = [], None
        for fr in c.decode(s):
            if fr.time is None:
                continue
            if first is None:
                first = fr.time
            if fr.time > t1:
                break
            for o in rs.resample(fr):
                chunks.append(o.to_ndarray().reshape(-1))
        for o in rs.resample(None):
            chunks.append(o.to_ndarray().reshape(-1))
    if not chunks:
        return None
    x = np.concatenate(chunks)
    a = max(0, int((t0 - first) * SR))
    return x[a:int((t1 - first) * SR)]


_model = None


def load_model():
    global _model
    if _model is None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer
        dev = os.environ.get("DEVICE") or "cpu"
        if dev == "cuda":
            try:
                torch.ones(1).cuda()
            except Exception as e:      # e.g. driver too old for the wheel's CUDA
                print("cuda unusable (%s), falling back to cpu" % str(e).splitlines()[0])
                dev = "cpu"
        torch.set_num_threads(os.cpu_count())
        fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
        # do_phonemize=False: we only decode, so espeak/phonemizer are not needed
        tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL, do_phonemize=False)
        mdl = Wav2Vec2ForCTC.from_pretrained(MODEL).to(dev).eval()
        _model = (fe, tok, mdl, dev)
        print("model %s on %s" % (MODEL, dev))
    return _model


def log_probs(key):
    """(T, V) CTC log-probs for the child's window, cached as float16."""
    t0, t1 = window(key)
    f = CACHE / ("%s_%.1f_%.1f.npy" % (key[5:], t0, t1))
    if f.exists():
        return np.load(f).astype(np.float32)
    import torch
    fe, tok, mdl, dev = load_model()
    x = load_window(AUDIO[key], t0, t1)
    if x is None or len(x) < SR:
        return None
    iv = fe(x, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    with torch.no_grad():
        lp = mdl(iv).logits[0].log_softmax(-1).cpu().numpy()
    np.save(f, lp.astype(np.float16))
    return lp


# vocab: needs only the tokenizer files, not the model weights
from transformers import Wav2Vec2PhonemeCTCTokenizer
_tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL, do_phonemize=False)
ID2TOK = {i: t for t, i in _tok.get_vocab().items()}
BLANK = _tok.pad_token_id
SKIP_IDS = {BLANK} | {i for t, i in _tok.get_vocab().items() if t.startswith("<") or t in ("|",)}
CLASS_IDS = collections.defaultdict(list)          # class -> token ids that realise it
for i, t in ID2TOK.items():
    if i not in SKIP_IDS:
        for c in set(cls(t)):
            CLASS_IDS[c].append(i)
for w in GOLD_C:
    for c in w:
        assert CLASS_IDS[c], "no model token for class %s" % c


# ---------------------------------------------------------------- GOP scorer
def lse(a, axis=-1):
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


def gop_items(lp, n_items):
    """Constrained Viterbi over [F0 item1 F1 item2 ... itemN FN]; returns per-item
    (min phone GOP, mean phone GOP, t_start, t_end) in frames."""
    T = lp.shape[0]
    best = lp.max(-1)                                          # free phone loop
    emis, kind, owner, phone = [], [], [], []                 # per state
    pred = []

    def add(e, k, it, ph, pr):
        emis.append(e); kind.append(k); owner.append(it); phone.append(ph); pred.append(pr)
        return len(emis) - 1

    filler = best - FILLER_PEN
    prev_exit = [add(filler, "F", 0, -1, [])]                  # F0
    pred[prev_exit[0]].append(prev_exit[0])
    starts = [prev_exit[0]]
    for it in range(1, n_items + 1):
        entry_from = prev_exit                                 # previous filler + previous item exits
        cl = GOLD_C[it - 1]
        b = add(lp[:, BLANK], "B", it, -1, entry_from + [])
        pred[b].append(b)
        last_p, last_b = None, b
        for k, c in enumerate(cl):
            p = add(lse(lp[:, CLASS_IDS[c]]), "P", it, k, [last_b])
            pred[p].append(p)
            if last_p is None:
                pred[p] += entry_from
            elif cl[k - 1] != c:
                pred[p].append(last_p)
            nb = add(lp[:, BLANK], "B", it, -1, [p])
            pred[nb].append(nb)
            last_p, last_b = p, nb
        if it == 1:
            starts += [b, b + 1]
        f = add(filler, "F", it, -1, [last_p, last_b])
        pred[f].append(f)
        prev_exit = [f, last_p, last_b]
    S = len(emis)
    E = np.stack(emis, 1)                                      # (T, S)
    P = np.full((S, max(len(p) for p in pred)), -1)
    for s, pr in enumerate(pred):
        P[s, :len(pr)] = pr
    valid = P >= 0
    delta = np.full(S, -np.inf)
    delta[starts] = E[0, starts]
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        cand = np.where(valid, delta[np.where(valid, P, 0)], -np.inf)
        a = cand.argmax(1)
        back[t] = P[np.arange(S), a]
        delta = cand[np.arange(S), a] + E[t]
    s = max(prev_exit, key=lambda q: delta[q])
    path = np.empty(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    res = {}
    kind, owner, phone = np.array(kind), np.array(owner), np.array(phone)
    for it in range(1, n_items + 1):
        fr = np.where(owner[path] == it)[0]
        fr = fr[kind[path[fr]] != "F"]
        g = []
        for k, c in enumerate(GOLD_C[it - 1]):
            pf = fr[(kind[path[fr]] == "P") & (phone[path[fr]] == k)]
            g.append(float(np.mean(lse(lp[pf][:, CLASS_IDS[c]]) - best[pf])) if len(pf) else -99.0)
        res[it] = (min(g), float(np.mean(g)), int(fr[0]) if len(fr) else None, int(fr[-1]) if len(fr) else None)
    return res


# ---------------------------------------------------------------- greedy baseline
def align_infix(ref, hyp):
    """ref aligned as an infix of hyp (same DP as eval_orf_urd.align_infix)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    j = min(range(m + 1), key=lambda jj: d[n][jj])
    ops, i = [], n
    while i > 0:
        if j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("eq" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def greedy_items(lp, n_items):
    ids = lp.argmax(-1)
    toks = [ID2TOK[i] for k, i in enumerate(ids) if i not in SKIP_IDS and (k == 0 or i != ids[k - 1])]
    hyp, src = [], []              # src[j] = token that class hyp[j] came from
    for k, t in enumerate(toks):
        for c in cls(t):
            hyp.append(c)
            src.append(k)
    ref, own = [], []
    for it in range(1, n_items + 1):
        ref += GOLD_C[it - 1]
        own += [it] * len(GOLD_C[it - 1])
    err, heard, last = collections.Counter(), collections.defaultdict(list), None
    for op, ri, hj in align_infix(ref, hyp):
        it = own[ri] if ri is not None else last
        last = own[ri] if ri is not None else last
        if it is not None:
            err[it] += op != "eq"
            if hj is not None and src[hj] not in heard[it][-1:]:
                heard[it].append(src[hj])
    return {it: (int(err[it] == 0), " ".join(toks[k] for k in heard[it]), err[it])
            for it in range(1, n_items + 1)}, " ".join(toks)


# ---------------------------------------------------------------- run
rows, kids = [], []
for n, key in enumerate(KEYS, 1):
    lp = log_probs(key)
    m = man[key]
    attempted = int(float(m["attempted"]))
    if lp is None or not attempted:
        continue
    g = gop_items(lp, attempted)
    gr, toks = greedy_items(lp, attempted)
    t0 = window(key)[0]
    for i in range(1, attempted + 1):
        if i not in human[key]:
            continue
        mn, mean, a, b = g[i]
        rows.append({"uuid": key[5:13], "grade": m["class_grade_a"], "enumerator": m["enumerator"],
                     "item_index": i, "gold_word": GOLD[i - 1],
                     "expected_ipa": " ".join(ref_phones(GOLD[i - 1])), "heard_ipa": gr[i][1],
                     "sound_errors": gr[i][2], "n_sounds": len(GOLD_C[i - 1]),
                     "human_correct": human[key][i], "gop_min": round(mn, 3), "gop_mean": round(mean, 3),
                     "t_start": round(t0 + a * FRAME, 2) if a is not None else None,
                     "t_end": round(t0 + (b + 1) * FRAME, 2) if b is not None else None,
                     "greedy_correct": gr[i][0]})
    kids.append({"uuid": key[5:13], "attempted": attempted, "greedy_ipa": toks,
                 "blank_frac": round(float((lp.argmax(-1) == BLANK).mean()), 3)})
    if n % 20 == 0:
        print("[%d/%d]" % (n, len(KEYS)), flush=True)


# ---------------------------------------------------------------- metrics
def conf(h, p):
    """positive class = 'incorrect' (h/p == 0), as in eval_orf_urd."""
    tp = sum(a == 0 and b == 0 for a, b in zip(h, p))
    fp = sum(a == 1 and b == 0 for a, b in zip(h, p))
    fn = sum(a == 0 and b == 1 for a, b in zip(h, p))
    tn = sum(a == 1 and b == 1 for a, b in zip(h, p))
    return tp, fp, fn, tn


def report(tag, h, p):
    tp, fp, fn, tn = conf(h, p)
    N = tp + fp + fn + tn
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    sp = tn / (tn + fp) if tn + fp else 0
    po = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / N / N
    k = (po - pe) / (1 - pe) if pe < 1 else 0
    print("  %-22s agree %.3f  P(inc) %.3f  R(inc) %.3f  F1 %.3f  bal.acc %.3f  kappa %.3f"
          % (tag, po, P, R, 2 * P * R / (P + R) if P + R else 0, (R + sp) / 2, k))
    return k


def auc(score, h):
    """P(score of a human-correct item > score of a human-incorrect item)."""
    pos = sorted(s for s, y in zip(score, h) if y == 1)
    neg = [s for s, y in zip(score, h) if y == 0]
    if not pos or not neg:
        return float("nan")
    import bisect
    w = sum(bisect.bisect_left(pos, s) + 0.5 * (bisect.bisect_right(pos, s) - bisect.bisect_left(pos, s)) for s in neg)
    return 1 - w / len(pos) / len(neg)


def best_thr(score, h):
    """threshold maximising balanced accuracy (predict correct iff score > thr)."""
    cands = sorted(set(score))
    best = (-1, None)
    for t in cands[:: max(1, len(cands) // 400)]:
        tp, fp, fn, tn = conf(h, [int(s > t) for s in score])
        R = tp / (tp + fn) if tp + fn else 0
        sp = tn / (tn + fp) if tn + fp else 0
        best = max(best, ((R + sp) / 2, t))
    return best[1]


def crossfit(rs, key="gop_min"):
    """choose the threshold on odd children, apply to even, and vice versa."""
    ids = sorted({r["uuid"] for r in rs})
    fold = {u: i % 2 for i, u in enumerate(ids)}
    pred = {}
    for f in (0, 1):
        tr = [r for r in rs if fold[r["uuid"]] != f]
        t = best_thr([r[key] for r in tr], [r["human_correct"] for r in tr])
        for r in rs:
            if fold[r["uuid"]] == f:
                pred[id(r)] = int(r[key] > t)
    return [pred[id(r)] for r in rs]


h = [r["human_correct"] for r in rows]
for r, p in zip(rows, crossfit(rows)):
    r["gop_correct"] = p
print("\nEGRA PWR_ENG - wav2vec2 phoneme eval (%s)  PAD=%.0fs FILLER_PEN=%.1f" % (MODEL, PAD, FILLER_PEN))
print("%d children / %d items   human: %.1f%% incorrect   median CTC blank frames %.1f%%"
      % (len(kids), len(rows), 100 * (1 - sum(h) / len(h)),
         100 * float(np.median([k["blank_frac"] for k in kids]))))
print("\nITEM LEVEL (positive class = incorrect)")
print("  AUC  gop_min %.3f   gop_mean %.3f   (0.5 = chance)"
      % (auc([r["gop_min"] for r in rows], h), auc([r["gop_mean"] for r in rows], h)))
report("greedy decode", h, [r["greedy_correct"] for r in rows])
_per = lambda rs: sum(r["sound_errors"] for r in rs) / sum(r["n_sounds"] for r in rs)
print("\nSOUND LEVEL (greedy: expected sounds vs what wav2vec heard)")
print("  sound error rate, all items               %.1f%%" % (100 * _per(rows)))
print("  ... items enumerator marked CORRECT       %.1f%%" % (100 * _per([r for r in rows if r["human_correct"]])))
print("  ... items enumerator marked INCORRECT     %.1f%%" % (100 * _per([r for r in rows if not r["human_correct"]])))
print("  items where wav2vec heard every sound     %.1f%%" % (100 * np.mean([r["sound_errors"] == 0 for r in rows])))
print("  items where wav2vec heard nothing         %.1f%%" % (100 * np.mean([not r["heard_ipa"] for r in rows])))
report("GOP (cross-fit thr)", h, [r["gop_correct"] for r in rows])
report("all 'incorrect'", h, [0] * len(h))

print("\nCHILD LEVEL (correct-item count)")
by = collections.defaultdict(list)
for r in rows:
    by[r["uuid"]].append(r)
for tag, k in (("greedy", "greedy_correct"), ("GOP", "gop_correct")):
    hc = [sum(x["human_correct"] for x in v) for v in by.values()]
    ac = [sum(x[k] for x in v) for v in by.values()]
    d = [a - b for a, b in zip(ac, hc)]
    r = np.corrcoef(hc, ac)[0, 1] if np.std(ac) else float("nan")
    print("  %-7s MAE %.2f  bias %+.2f  r %.3f   zero-scorers human %d / model %d (of %d)"
          % (tag, np.mean(np.abs(d)), np.mean(d), r, sum(x == 0 for x in hc), sum(x == 0 for x in ac), len(hc)))

print("\nBY GRADE (GOP)")
for g in sorted({r["grade"] for r in rows}):
    rs = [r for r in rows if r["grade"] == g]
    hh = [r["human_correct"] for r in rs]
    print("  grade %-4s n_items=%4d  AUC %.3f" % (g, len(rs), auc([r["gop_min"] for r in rs], hh)), end="")
    report("", hh, [r["gop_correct"] for r in rs])

print("\nBY ITEM (first 10; most children stop in line 1)")
for i in range(1, 11):
    rs = [r for r in rows if r["item_index"] == i]
    hh = [r["human_correct"] for r in rs]
    print("  %2d %-4s n=%3d  human correct %.0f%%  AUC %.3f"
          % (i, GOLD[i - 1], len(rs), 100 * np.mean(hh), auc([r["gop_min"] for r in rs], hh)))

tag = MODEL.split("/")[-1]
with open(OUT / ("per_item_%s.csv" % tag), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(OUT / ("side_by_side_%s.txt" % tag), "w", encoding="utf-8") as f:
    for k in kids:
        v = by[k["uuid"]]
        f.write("=" * 70 + "\n%s  attempted=%d  human_correct=%d  gop_correct=%d  blank=%.2f\n"
                % (k["uuid"], k["attempted"], sum(x["human_correct"] for x in v),
                   sum(x["gop_correct"] for x in v), k["blank_frac"]))
        for x in v:
            f.write("  %2d %-4s expected /%s/  heard /%s/  (%d/%d sounds wrong)  enumerator=%s  gop=%d\n" % (
                x["item_index"], x["gold_word"], x["expected_ipa"], x["heard_ipa"] or "-",
                x["sound_errors"], x["n_sounds"], "correct" if x["human_correct"] else "WRONG", x["gop_correct"]))
        f.write("  greedy IPA: %s\n\n" % k["greedy_ipa"])
print("\n-> eval_pwr_eng/per_item_%s.csv, side_by_side_%s.txt" % (tag, tag))
