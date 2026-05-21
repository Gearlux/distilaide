# Distilaide Mandates

- **Adapter, Not Denoiser:** Distilaide is a thin (~250 LOC) glue layer over upstream denoising libraries — `bm3d`, `noisereduce`, `deepinv`, `asteroid`. Never reimplement a denoising algorithm; wrap the upstream call. If a method doesn't exist upstream, file an issue rather than rolling your own.
- **One Backend Module Per Library:** Each upstream package gets exactly one `<name>_backend.py` with lazy imports so missing optional extras never break top-level import. Don't merge backends; don't split a single backend across files.
- **Single Dispatcher Surface:** `denoise(signal, method=...)` is the single entry point; the `Denoiser` Confluid-`@configurable` wraps it. New methods register into the per-backend method table; never add parallel dispatchers.
- **Optional Extras Discipline:** Heavy deps (`deepinv`, `asteroid`, classical `bm3d`/`noisereduce`) live behind `[classical]` / `[deepinv]` / `[asteroid]` / `[all]` extras. Core install stays minimal — top-level import MUST succeed without any extra.
- **CLI via Liquifai:** Verbs `list-methods` and `denoise <yaml>` follow workspace Liquifai conventions. `.npy` / `.wav` round-trip is the canonical IO; don't add ad-hoc format flags.
- **Logflow / Confluid First:** Diagnostic output goes through `logflow.get_logger`, not `print`. All knobs (method, params, IO paths) MUST be reachable from a single Confluid YAML.
