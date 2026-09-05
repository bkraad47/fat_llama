---
name: review-current-state
description: Walks the project's files, builds a file tree, reads through each source file, and produces a reusable "factblock" reference — per function/method/class: signature, description, parameters, returns, and a usage example. Use when asked to review, document, snapshot, or get an overview of the current state of the codebase.
---

Produce a single up-to-date reference document for the project: a file tree plus a "factblock" for every function, method, and class, in a fixed, reusable template. This is a snapshot skill — it reads the current code, it does not modify it.

## Logging

Before Step 1, open this run's log file per `.claude/rules/logging.md` (name: `review-current-state-<time>-<user>.log`). Append one entry per step below as it happens.

## Scope

- Default scope is the whole repository.
- If the user names a path/module in `args`, scope the review to that path only and note the narrowed scope in the output.
- Exclude: `.git`, `.venv`, `venv`, `build`, `dist`, `*.egg-info`, `__pycache__`, `.pytest_cache`, `node_modules`, and any other virtualenv/artifact directories you find. Only include tracked source, not generated/vendored files.

## Steps

1. **Enumerate files.** Use Glob/Bash (`git ls-files` is the cleanest source of truth if in a git repo — it naturally skips ignored build/venv directories) to list every tracked file in scope.

2. **Build the file tree.** Render it as a fenced text block showing the directory hierarchy, e.g.:

   ```
   fat_llama/
   ├── __init__.py
   ├── audio_fattener/
   │   └── feed.py
   └── tests/
       └── test_feed.py
   ```

   Keep non-code files (README, setup.py, requirements.txt, etc.) in the tree for completeness, but only factblock actual source files (see next step).

3. **Read every source file** (`.py`, or the project's primary language) in scope with the Read tool — do not skip files to save time; the point of this skill is completeness. For each file, identify every module-level function, class, and method (including `__init__` when it does non-trivial setup).

4. **Write one factblock per function/method/class**, using this exact template so the output stays reusable and parseable by later tooling or skills:

   ```markdown
   ### `qualified.name(args) -> ReturnType`
   **File:** path/to/file.py:LINE
   **Kind:** function | method | class
   **Description:** One or two sentences on what it does and why it exists (pull from the docstring if present; otherwise infer from the body — say so if inferred).
   **Parameters:**
   - `arg` (`type`): meaning
   **Returns:** type and meaning (omit for classes/void functions).
   **Usage:**
   \`\`\`python
   # minimal, runnable-looking example
   \`\`\`
   ```

   For the usage example: prefer lifting real call sites from the codebase's own tests or `example.py`-style scripts when one exists for that function; only synthesize a minimal example when no real call site is found, and mark it `# illustrative` in that case.

5. **Assemble the document** in this order: title, scope note, file tree, then factblocks grouped under a `## path/to/file.py` heading per file, in file-tree order.

6. **Write the output** to `docs/CURRENT_STATE.md` at the repo root (create the `docs/` directory if it doesn't exist), overwriting any previous version — this file is meant to be regenerated on demand, not hand-edited. If the user specifies a different output path in `args`, use that instead.

7. **Report back** with a short summary: number of files reviewed, number of factblocks written, and the output path — not the full document contents.
