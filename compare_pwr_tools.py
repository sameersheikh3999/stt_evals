# -*- coding: utf-8 -*-
"""
Comparison of SpeechAce and OpenPronounce on PWR_ENG, original vs cleaned audio.

Reads the per-item results written by eval_pwr_speechace_openpronounce.py
(eval_pwr_eng/sa_op/ = original, eval_pwr_eng/sa_op_clean/ = PREPROC=clean) for the
children given on the command line, and reports agreement with the enumerator
overall, by grade and by child.

Usage: .venv/Scripts/python compare_pwr_tools.py uuid8 [uuid8 ...]
  -> eval_pwr_eng/comparison_per_item.csv
"""
import csv, sys, pathlib, statistics as st

ROOT = pathlib.Path(__file__).parent
KIDS = sys.argv[1:]
VARIANTS = (("original", "sa_op"), ("cleaned", "sa_op_clean"))
TOOLS = (("SpeechAce", "speechace_ok"), ("OpenPronounce", "openpronounce_ok"))

man = {r["KEY"][5:13]: r for r in csv.DictReader(open(ROOT / "eval_manifest.csv", encoding="utf-8-sig"))
       if r["task_id"] == "PWR_ENG"}

rows = {}                         # (uuid, item) -> merged row
for vname, d in VARIANTS:
    for r in csv.DictReader(open(ROOT / "eval_pwr_eng" / d / "per_item.csv", encoding="utf-8-sig")):
        if r["uuid"] not in KIDS:
            continue
        k = (r["uuid"], int(r["item"]))
        m = man[r["uuid"]]
        row = rows.setdefault(k, {"uuid": r["uuid"], "grade": m["class_grade_a"][:1], "enumerator_name": m["enumerator"],
                                  "item": int(r["item"]), "word": r["word"], "enumerator": int(r["enumerator"])})
        row["sa_score_" + vname] = float(r["speechace_score"])
        row["sa_ok_" + vname] = int(r["speechace_ok"])
        row["op_ok_" + vname] = int(r["openpronounce_ok"])
R = sorted(rows.values(), key=lambda r: (r["uuid"], r["item"]))
missing = [k for k in KIDS if not any(r["uuid"] == k for r in R)]
if missing:
    print("not in results yet:", missing)
R = [r for r in R if all(("op_ok_" + v) in r for v, _ in VARIANTS)]


def stats(h, p):
    tp = sum(a == 0 and b == 0 for a, b in zip(h, p)); tn = sum(a == 1 and b == 1 for a, b in zip(h, p))
    fp = sum(a == 1 and b == 0 for a, b in zip(h, p)); fn = sum(a == 0 and b == 1 for a, b in zip(h, p))
    N = len(h); po = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / N / N
    pr = tn / (tn + fp) if tn + fp else float("nan")        # correct words passed
    cw = tp / (tp + fn) if tp + fn else float("nan")        # wrong words caught
    return po, (po - pe) / (1 - pe) if pe < 1 else 0.0, (pr + cw) / 2, pr, cw


def auc(score, h):
    pos = [s for s, y in zip(score, h) if y == 1]; neg = [s for s, y in zip(score, h) if y == 0]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / len(pos) / len(neg)


def col(tool, v):
    return ("sa_ok_" if tool == "SpeechAce" else "op_ok_") + v


h = [r["enumerator"] for r in R]
print("%d children, %d words; enumerator: %d correct / %d wrong\n"
      % (len({r["uuid"] for r in R}), len(R), sum(h), len(h) - sum(h)))

print("OVERALL")
print("  %-15s %-9s %6s %6s %8s %13s %13s" % ("tool", "audio", "agree", "kappa", "bal.acc", "passes-right", "catches-wrong"))
for t, _ in TOOLS:
    for v, _ in VARIANTS:
        po, k, ba, pr, cw = stats(h, [r[col(t, v)] for r in R])
        print("  %-15s %-9s %6.2f %6.2f %8.2f %13.2f %13.2f" % (t, v, po, k, ba, pr, cw))
for v, _ in VARIANTS:
    print("  SpeechAce score AUC (%s): %.2f   mean score correct/wrong words: %.1f / %.1f"
          % (v, auc([r["sa_score_" + v] for r in R], h),
             st.mean(r["sa_score_" + v] for r in R if r["enumerator"]),
             st.mean(r["sa_score_" + v] for r in R if not r["enumerator"])))
print("  (kappa / bal.acc: 0 / 0.50 = chance)")

print("\nBY GRADE (kappa; SpeechAce AUC)")
print("  %-6s %5s  %-26s %-26s %s" % ("grade", "words", "SpeechAce orig -> clean", "OpenPronounce orig -> clean", "SA AUC orig -> clean"))
for g in sorted({r["grade"] for r in R}):
    rs = [r for r in R if r["grade"] == g]; hh = [r["enumerator"] for r in rs]
    ks = {(t, v): stats(hh, [r[col(t, v)] for r in rs])[1] for t, _ in TOOLS for v, _ in VARIANTS}
    print("  %-6s %5d  %+.2f -> %+.2f %14s %+.2f -> %+.2f %14s %.2f -> %.2f" % (
        g, len(rs), ks["SpeechAce", "original"], ks["SpeechAce", "cleaned"], "",
        ks["OpenPronounce", "original"], ks["OpenPronounce", "cleaned"], "",
        auc([r["sa_score_original"] for r in rs], hh), auc([r["sa_score_cleaned"] for r in rs], hh)))

print("\nBY CHILD (number of words judged correct)")
print("  %-9s %-5s %-16s %9s %13s %13s" % ("child", "grade", "enumerator", "enum", "SA orig/clean", "OP orig/clean"))
for u in dict.fromkeys(r["uuid"] for r in R):
    rs = [r for r in R if r["uuid"] == u]
    print("  %-9s %-5s %-16s %4d/%-4d %6d / %-6d %6d / %-6d" % (
        u, rs[0]["grade"], rs[0]["enumerator_name"][:16], sum(r["enumerator"] for r in rs), len(rs),
        sum(r["sa_ok_original"] for r in rs), sum(r["sa_ok_cleaned"] for r in rs),
        sum(r["op_ok_original"] for r in rs), sum(r["op_ok_cleaned"] for r in rs)))

print("\nTOOLS vs EACH OTHER")
for v, _ in VARIANTS:
    a = [r[col("SpeechAce", v)] for r in R]; b = [r[col("OpenPronounce", v)] for r in R]
    both = sum(x and y for x, y in zip(a, b))
    print("  %-9s agree with each other %.2f; words both passed: %d" % (v, sum(x == y for x, y in zip(a, b)) / len(R), both))

out = ROOT / "eval_pwr_eng" / "comparison_per_item.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(R[0]))
    w.writeheader()
    w.writerows(R)
print("\n-> %s" % out.relative_to(ROOT))
