"""Turn the segment JSONL from transcribe.py into the two transcript files.

Writes the Hebrew transcript (model output, verbatim) and, if a translation TSV
(segment index <TAB> English) is given, the English file with the same timestamps.

A segment gets a [?] mark when any of these holds:
  avg_logprob < LOGPROB      the 30 s window it came from scored low overall
  mean word probability < MEAN_WORD
  lowest word probability < MIN_WORD   (the model guessed at least one word)
  sparse: 10 s or longer with under one word per second (speech probably missed)

no_speech_prob is not used: this model reports 0.0 for every segment.

Usage: python build_transcript.py <segments.jsonl> <out_he.md> [<en.tsv> <out_en.md>]
"""
import json
import statistics
import sys

LOGPROB = -0.32
MEAN_WORD = 0.80
MIN_WORD = 0.30

jsonl, out_he = sys.argv[1], sys.argv[2]
en_tsv, out_en = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else (None, None)


def ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def low_confidence(r: dict) -> bool:
    w = r["word_probs"]
    dur = r["end"] - r["start"]
    sparse = dur >= 10 and len(r["text"].split()) / dur < 1.0
    return (r["avg_logprob"] < LOGPROB or statistics.mean(w) < MEAN_WORD
            or min(w) < MIN_WORD or sparse)


rows = [json.loads(line) for line in open(jsonl, encoding="utf-8") if line.strip()]
flags = [low_confidence(r) for r in rows]
rule = (f"[?] = low confidence, listen yourself. Rule: window avg_logprob < {LOGPROB}, "
        f"or mean word probability < {MEAN_WORD}, or any word below {MIN_WORD}, "
        f"or a stretch of 10 s+ with under one word per second.")
stats = f"Segments: {len(rows)}. Marked [?]: {sum(flags)}."


def write(path: str, title: str, intro: str, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{intro}\n\n{stats}\n\n{rule}\n\n---\n\n")
        for r, low, text in zip(rows, flags, texts):
            f.write(f"[{ts(r['start'])}–{ts(r['end'])}]{' [?]' if low else ''} {text}\n\n")
    print(f"wrote {path}")


write(out_he, "Meeting 2026-11-23, Hebrew transcript",
      "Source: `2026-11-23 11-49-30.mp3` (40:23), cleaned with ffmpeg, transcribed with "
      "faster-whisper and `ivrit-ai/whisper-large-v3-ct2`, language Hebrew, VAD on. "
      "The text is the model's output, uncorrected. Speakers are not labelled.",
      [r["text"] for r in rows])

if en_tsv:
    en = {}
    for line in open(en_tsv, encoding="utf-8"):
        if line.strip():
            i, text = line.rstrip("\n").split("\t", 1)
            en[int(i)] = text
    missing = [i for i in range(len(rows)) if i not in en]
    assert not missing and len(en) == len(rows), f"translation rows missing: {missing[:10]}"
    write(out_en, "Meeting 2026-11-23, English translation",
          "Translated from the Hebrew transcript, one line per segment with the same "
          "timestamps and [?] marks. Square brackets are translator notes: [heard \"…\"] "
          "gives the Hebrew the model produced where the intended word was inferred from "
          "context, and [unclear] marks text that could not be made sense of. "
          "A slash ( / ) marks a change of speaker inside one segment. Speakers are not labelled.",
          [en[i] for i in range(len(rows))])
print(stats)
