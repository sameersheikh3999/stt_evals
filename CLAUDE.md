# stt_evals: project context

Can speech-to-text / pronunciation tools replace the human enumerator who scores
children's EGRA reading tasks? Each eval cuts one subtask out of a session
recording, scores it with a tool, and compares the tool's per-word
correct/incorrect verdict with the enumerator's tap.

Last updated 2026-10-02. Branch `eval-pwr-eng-wav2vec` (pushed to
`origin`, github.com/sameersheikh3999/stt_evals). It has not been merged into `main` yet.

## Data (all local, all gitignored: PII and size)

| File | What it is |
|---|---|
| `EGMA audio files - path-20260918T004653Z-1-001/` | Session recordings `AA_<uuid36>_enumerator.m4a`, one per child, whole session (all subtasks). 8 kHz AAC, phone-grade, far-field, classroom noise. Duplicate `Copy of ...(1).m4a` files are identical. |
| `audios_selected/` | Second audio folder, same naming (searched by `eval_pwr_eng.py`). |
| `eval_manifest.csv` | One row per child × subtask. Key columns: `KEY` (`uuid:<uuid36>`), `task_id`, `offset_start_sec`/`offset_end_sec` (subtask window inside the m4a), `offset_usable`, `attempted` (items the child reached), `class_grade_a`, `enumerator`. |
| `gold_items_long.csv` | Enumerator labels, one row per child × item: `KEY`, `task_id`, `spoken_position`, `item_index`, `item_text`, `is_correct`. |
| `EGRA_STT_eval_key.xlsx` | Sheets `Item_Key` (col 0 `task_id`, col 5 `item_text` = gold word), `Passages`, `Comprehension_Key`, `Scoring_Rules`, `Subtasks`. |
| `EGRA_EGMA client N=2233.xlsx`, `EGRA_EGMA_Combine codebook v109.html` | Raw client data and codebook. |
| `transcripts/json/` | Full-session Deepgram nova-3 (`language=ur`) transcripts used by ORF_URD, plus `sample_audio*` test outputs. |
| `.env` | `deepgram_api_key=...`, `speechace_api_key=...` (read by `env_key()` in the scripts). |

Join on the **full 36-char uuid** from the filename (`AA_([0-9a-f-]{36})_enumerator`).
The newer scripts take 8-char uuid prefixes on the command line.

## Subtasks evaluated

- **ORF_URD**: Urdu oral reading fluency, a 60-word passage.
- **PWR_ENG**: English pseudowords (phonics), 50 items: `maz ver lut paf nom yod fut et zib mib ...`.
  Most children stop after line 1 (5 items, firstline autostop).
- **FWR_ENG**: English familiar words (same Deepgram script with `TASK=FWR_ENG`).

## Scripts

Run everything with `.venv/Scripts/python <script>` (Python 3.12, Windows).

| Script | Does |
|---|---|
| `eval_orf_urd.py` | ORF_URD: slices the full-session Deepgram Urdu transcript to the window (`PAD` default 5 s), normalises Urdu, aligns the gold passage as an *infix* (so enumerator speech either side is not counted), and reports WER/CER plus per-item P/R/F1 with "incorrect" as the positive class. Output goes to `eval_orf_urd/`. |
| `eval_pwr_eng.py` | PWR_ENG with the wav2vec2 phoneme model `facebook/wav2vec2-xlsr-53-espeak-cv-ft`. It uses hand-written reference IPA (short vowels a/e/i/o/u = æ/ɛ/ɪ/ɒ/ʌ) collapsed into broad phone classes. There are two scorers: **GOP**, a constrained Viterbi forced alignment with filler states for enumerator speech and a threshold cross-fitted across children, and **greedy** CTC decoding. Log-probs are cached in `eval_pwr_eng/<model>/`. Env vars: `MODEL`, `PAD`, `FILLER_PEN`, `LIMIT`, `DEVICE`. |
| `eval_pwr_speechace_openpronounce.py` | PWR_ENG with **SpeechAce** (API, `api5` region, 30 s maximum per request, so it uses 29 s chunks with a 20 s hop and keeps each word's best score) and **OpenPronounce** (local). Env vars: `SA_THR` (default 80), `PAD` (2), `PREPROC` (`none`/`gain`/`clean`). Saved responses are reused so quota isn't spent twice. Output goes to `eval_pwr_eng/sa_op[_<preproc>]/`. |
| `eval_pwr_deepgram.py` | PWR_ENG or FWR_ENG with **Deepgram nova-3 English**, on original and cleaned audio. It judges each word two ways: an *exact* spelling match, and a *sound* match (g2p_en phonemes, cosine ≥ 0.8). Transcripts are cached in `eval_<task>/deepgram[_clean]/`. |
| `compare_pwr_tools.py` | Merges the SpeechAce/OpenPronounce `per_item.csv` results for original and cleaned audio, and breaks them down overall, by grade, and by child. |
| `g2p_cosine.py` | Scores a Deepgram JSON against the 50 pseudowords by phoneme cosine (unigrams + bigrams). Rough guide: identical 1.0, one extra phone ≈0.76, one phone swapped ≈0.57. |
| `preproc.py` | NumPy-only audio cleanup: `gain` (to −20 dBFS on speech frames), `highpass` (80 Hz), `denoise` (spectral subtraction). `clean` = highpass → denoise → gain. |
| `run_openpronounce.py`, `transcribe_w2v_960h.py` | Single-file test runners on `sample_audio.ogg`. The second one runs wav2vec2-large-960h and writes Deepgram-shaped JSON. |

The 8 children used for the API tool comparisons:
`3b6b6972 52be51a8 4f99136d 7e464fe2 9f532c2d 27e407f8 aa8eabb8 788dfa6a`
(grades 1–5, 379 PWR_ENG words). The earlier pilot used `52be51a8 7e464fe2 d99c7b13`.

## Results so far

Metrics treat "incorrect" as the positive class. kappa 0 / balanced accuracy 0.50 / AUC 0.50 = chance.

**ORF_URD, Deepgram Urdu (6 children, 223 items).** Median WER 0.80, CER 0.57.
Item agreement 0.60, P(incorrect) 0.50, R 0.92, F1 0.65, compared with an
all-"incorrect" baseline at a 39.5% base rate (F1 ≈ 0.57).

**PWR_ENG, wav2vec2 xlsr-53 phonemes (193 children, 3606 items, 40.7% incorrect).**
95% of CTC frames are blank. GOP AUC is 0.52 and kappa 0.03, and greedy decoding has kappa 0.00: both are at chance at the
item level. At the child level the GOP correct-count tracks the enumerator
(r 0.87, MAE 5.8 items), but that comes mostly from how many items each child attempted, which is taken
from the human.

**PWR_ENG, 8 children / 379 words (213 correct / 166 wrong):**

| Tool | Audio | agree | kappa | passes right | catches wrong | AUC |
|---|---|---|---|---|---|---|
| SpeechAce (≥80) | original | 0.49 | 0.06 | 0.15 | 0.91 | 0.53 |
| SpeechAce (≥80) | cleaned | 0.49 | 0.06 | 0.19 | 0.87 | 0.58 |
| OpenPronounce | original | 0.45 | −0.02 | 0.19 | 0.80 | – |
| OpenPronounce | cleaned | 0.46 | 0.00 | 0.21 | 0.80 | – |
| Deepgram exact/sound | original | 0.44 | 0.00 | 0.00 | 0.99 | 0.56 (cosine) |
| Deepgram exact/sound | cleaned | 0.44 | 0.00 | 0.00 | 1.00 | 0.54 (cosine) |

**FWR_ENG, Deepgram (8 children, 388 words, 292 correct).** kappa 0.06–0.07,
passes only 14–16% of correct words, cosine AUC 0.47.

**Takeaway so far:** none of the tools agree with the enumerator meaningfully
better than chance on this audio. All of them reject almost everything. Cleaning the
audio barely helps (SpeechAce AUC goes from 0.53 to 0.58). The bottleneck looks like the audio
itself: 8 kHz, far-field, children and classroom noise. Better recordings are
probably needed before tool choice matters.

## Gotchas on this machine

- No ffmpeg: decode audio with **PyAV** (`av`).
- The GPU driver is too old for the installed torch, so run on **CPU** (`OPENPRONOUNCE_DEVICE=cpu`, `DEVICE` defaults to cpu).
- Windows Smart App Control sometimes blocks the torch/scipy DLLs. That is why `preproc.py` uses NumPy only, and why the SpeechAce script falls back to SpeechAce-only when OpenPronounce fails to import.
- espeak-ng comes from the `espeakng-loader` wheel, so no system install is needed.
- OpenPronounce uses gTTS, which sends the expected **text** (not audio) to Google.
- Print with UTF-8 in mind: the default cp1252 console fails on Urdu text.
- **Known break:** `eval_orf_urd.py` exits with `cannot parse uuid from transcript filename: sample_audio.json` because test files sit in `transcripts/json/`. Either move the `sample_audio*` files out or make the loader skip non-matching names.
- `eval_*/` output folders and `*.out` are gitignored. Results live only locally.
