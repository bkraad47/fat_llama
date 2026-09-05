# Audio quality rules — audio-quality-checker

These are the rules `audio-quality-checker` follows. Edit this file to change its behavior without touching the agent definition.

## Model

This agent runs on the **latest available Opus model** — set in its frontmatter — rather than the project's Sonnet default, since judging audio coherence/quality is the most subjective task in this project's pipeline and benefits from the strongest available model.

## Output contract

Your final message must be *only* this JSON — no prose before or after it:

```json
{
  "tests": [
    { "name": "test_read_audio_coherence", "type": "audio_quality", "status": "pass", "failure_reason": null }
  ],
  "log": ".claude/log/audio-quality-checker-20260905-141500-bkraad47.log"
}
```

- One entry per quality/coherence check you ran.
- `type` is always `"audio_quality"` for this agent.
- `status` is `"pass"` or `"fail"`.
- `failure_reason` is `null` when `status` is `"pass"`; otherwise a short, specific description (e.g. "bitrate out of range", "output shorter than input", "test only checks file existence, not audio content").
- `log`: the path to this run's log file, per `.claude/rules/logging.md`.

## What to check

1. **Output audio quality** — exercise `read_audio`/`write_audio` from [feed.py](../../../fat_llama/audio_fattener/feed.py) on a synthesized sample (reuse the `pydub.generators.Sine` pattern from [test_feed.py](../../../fat_llama/tests/test_feed.py)), then inspect the produced file with `soundfile`/`pydub`/`mutagen` for:
   - bitrate within an acceptable range (TBD — fill in target ranges per format)
   - sample rate / channel count preserved or upsampled as intended
   - duration matches the input (within a small tolerance)
2. **Test coherence** — review [test_feed.py](../../../fat_llama/tests/test_feed.py) and flag any test that doesn't meaningfully assert on audio content (e.g. only checks that a file exists or is non-empty, without checking duration/format/sample properties).

## Acceptable ranges

TBD — fill in bitrate/sample-rate/duration-tolerance thresholds once baseline outputs are measured.

## Fixing philosophy

- You may fix or strengthen test *assertions* in [test_feed.py](../../../fat_llama/tests/test_feed.py) to make them coherent.
- Leave deep fixes to production code in `feed.py` to `generate-code` — report the issue instead of fixing it yourself, unless it's a trivial, obviously-correct one-line fix.
- Per `.claude/rules/project-mission.md`: fat_llama enhances audio strictly via iterative soft thresholding over FFT data, never AI/ML-based upscaling. Judge "coherent and high-quality" against that DSP method — never suggest or apply an assertion fix that would only make sense if an AI-upscaling step were added.

## Scope restrictions

See `.claude/rules/scope-and-safety.md` for the full project-wide policy. For this agent specifically:

- Writes are limited to test *assertions* under `fat_llama/tests/**` — never `fat_llama/audio_fattener/**` or any other source.
- Never write under `.claude/` (except your own log entry) or `.github/workflows/`.
- Run only the audio/test exercises this task calls for — no unrelated commands, no network access beyond what running the local test/audio pipeline requires.

## Open items

Fill in over time: target bitrate/sample-rate ranges per supported format, tolerance thresholds, any perceptual-quality checks to add later.
