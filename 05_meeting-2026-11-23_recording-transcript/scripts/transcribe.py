"""Transcribe a WAV with faster-whisper (Hebrew, VAD on) and write one JSON line per segment.

Usage: python transcribe.py <wav> <model> <out.jsonl> [--threads N]
"""
import argparse
import json
import sys
import time
import wave

import numpy as np
from faster_whisper import WhisperModel

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()

# Read the 16 kHz mono 16-bit WAV directly (the installed PyAV and faster-whisper
# versions disagree on a decoder argument, and the file needs no decoding anyway).
with wave.open(a.wav, "rb") as w:
    assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), "need 16 kHz mono s16 WAV"
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

t0 = time.time()
model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
t_load = time.time() - t0
segments, info = model.transcribe(
    audio,
    language="he",
    task="transcribe",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
    beam_size=5,
    # avg_logprob is shared by all segments cut from one 30 s window, so also keep
    # per-word probabilities to judge each segment on its own.
    word_timestamps=True,
    # Poor audio: don't let one badly heard segment steer the next ones.
    condition_on_previous_text=False,
)
t1 = time.time()
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for s in segments:
        row = dict(
            start=round(s.start, 2),
            end=round(s.end, 2),
            text=s.text.strip(),
            avg_logprob=round(s.avg_logprob, 4),
            no_speech_prob=round(s.no_speech_prob, 4),
            compression_ratio=round(s.compression_ratio, 3),
            word_probs=[round(w.probability, 3) for w in (s.words or [])],
        )
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        n += 1
        print(f"[{s.end:8.1f}s / {info.duration:.0f}s] elapsed {time.time() - t1:6.0f}s",
              file=sys.stderr, flush=True)
el = time.time() - t1
print(json.dumps(dict(
    model=a.model, threads=a.threads, audio_s=round(info.duration, 1),
    speech_s_after_vad=round(info.duration_after_vad, 1), load_s=round(t_load, 1),
    transcribe_s=round(el, 1), segments=n)))
