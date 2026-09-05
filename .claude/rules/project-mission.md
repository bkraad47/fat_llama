# Project mission — all skills and subagents

`README.md` at the repo root is the single source of truth for what fat_llama actually is and how it works — not just the code structure, the *purpose*. This file distills the load-bearing parts of it so every skill/agent applies them consistently; edit this file (not each skill/agent) to change the mission statement project-wide, and keep it in sync if `README.md` changes.

## What fat_llama does

fat_llama upscales compressed audio between formats — it supports multiple source formats (mp3, wav, ogg, flac) and target formats (flac, wav), **but the outcome we build and test against is MP3 → FLAC**; treat that as the primary, load-bearing path and other format pairs as supported but secondary until they get the same level of test coverage. The upscaling itself uses **iterative soft thresholding (IST) applied to FFT (Fast Fourier Transform) data**: transform the signal to the frequency domain, threshold to keep significant frequencies while discarding noise, use that to add missing/congested detail, inverse-transform back to the time domain, then rescale amplitude and normalize. See `README.md`'s "Algorithm Explanation" and "Why FFT and IST?" sections for the full method and its citation (Kamal, "Fast Sparse Fourier Transformations for NMR Spectroscopy", 2015).

## Hard constraint: no AI upscaling

This project deliberately does **not** use AI/ML-based upscaling — no neural super-resolution, no learned/trained models, no diffusion or GAN-based audio enhancement — to add detail. Enhancement comes strictly from the DSP method above (FFT + IST + interpolation + auto-scaling/normalization). Any fix, feature, or dependency that would introduce a trained model as the *mechanism* for adding audio detail is out of scope: flag it and ask rather than implementing it.

## CUDA-only focus

This `fat_llama` package targets **CUDA-accelerated processing via CuPy** — that's the primary and only path this project's skills/agents build and test against, not an optional speedup. A CPU-only variant exists as a separate package (`fat-llama-fftw`, linked from `README.md`'s Requirements section) but is out of scope here: don't add CPU-fallback code paths, `numpy`-only branches, or make CUDA/CuPy optional in `fat_llama` itself unless the user explicitly asks for that as a distinct task. If a target environment lacks a CUDA-capable GPU, treat that as an environmental limitation to report (per `code-quality.md`/`scientific-coding.md`), not something to work around by degrading to CPU.

## When to read `README.md`

- `iterate-fat-llama` and `generate-code` (both the skill and the subagent) read `README.md` in full early in their run — before Step/Task 1, at the same point they open their log file — and carry its context (the two points above, condensed) into every `generate-code` dispatch and the steps that follow it in that run, so a fix cycle never drifts into an out-of-scope approach.
- Any other skill/agent whose task could plausibly touch *how* audio is enhanced (not just measure or report on it) should ground itself here first, same as above.
- Skills/agents that only run, measure, or report (`code-tester-reviewer`, `audio-quality-checker`, `review-current-state`'s factblock) don't need to re-read the full README every run, but should already know the constraint above when judging whether something they observe is in scope — e.g. flag it as a finding if a change under review introduces a trained-model dependency for audio enhancement.
