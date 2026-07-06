"""Distilaide CLI.

Two commands:

    distilaide list-methods             # list registered methods + availability
    distilaide denoise <yaml>           # apply a configured Denoiser to a file

The ``denoise`` command's YAML provides a ``denoiser:`` block (a
``!class:distilaide.Denoiser`` instantiation) plus ``input:`` / ``output:``
paths. Inputs and outputs are resolved by extension:

- ``.npy``  — :func:`numpy.load` / :func:`numpy.save` round-trip.
- ``.wav``  — :mod:`scipy.io.wavfile` (when ``[classical]`` extra is
  installed; ``scipy`` is part of that extra).
- anything else — error.

NOTE: do not add ``from __future__ import annotations`` here — liquifai's DI
inspects the actual class objects from parameter annotations and PEP 563
string annotations break that lookup.
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from liquifai import LiquifyApp, get_context
from loggair import get_logger

from distilaide.api import Denoiser, list_methods

logger = get_logger(__name__)

app = LiquifyApp(name="distilaide", description="adapter over deepinv/asteroid/bm3d/noisereduce.")


def _load_signal(path: Path) -> Tuple[np.ndarray, dict[str, Any]]:
    """Load a numpy array (or wav) from ``path``.

    Returns the array and a dict of side-channel metadata (e.g. sample rate)
    that the writer needs to round-trip the file.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path), {}
    if suffix == ".wav":
        from scipy.io import wavfile

        sr, data = wavfile.read(path)
        return data, {"sample_rate": int(sr)}
    raise ValueError(f"Unsupported input extension: {suffix!r}. Use .npy or .wav.")


def _save_signal(path: Path, signal: np.ndarray, meta: dict[str, Any]) -> None:
    """Write ``signal`` back to ``path`` using the format implied by extension."""
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".npy":
        np.save(path, signal)
        return
    if suffix == ".wav":
        from scipy.io import wavfile

        sample_rate = int(meta.get("sample_rate", 16000))
        wavfile.write(path, sample_rate, signal)
        return
    raise ValueError(f"Unsupported output extension: {suffix!r}. Use .npy or .wav.")


def _config() -> dict[str, Any]:
    ctx = get_context()
    if ctx is None or not ctx.config_data:
        raise RuntimeError("No distilaide config loaded. Provide a YAML file as the first argument.")
    return dict(ctx.config_data)


@app.script_command(name="list-methods")
def list_methods_cmd() -> None:
    """List every registered denoising method and whether its backend is installed."""
    rows = list_methods()
    width = max(len(r["method"]) for r in rows)
    print(f"{'method'.ljust(width)}  backend       installed")
    print(f"{'-' * width}  ------------  ---------")
    for row in rows:
        flag = "yes" if row["available"] else "no"
        print(f"{row['method'].ljust(width)}  {row['backend']:<12}  {flag}")


@app.script_command(name="denoise")
def denoise_cmd(denoiser: Denoiser) -> None:
    """Apply the configured ``denoiser:`` to ``input:`` and write to ``output:``."""
    config = _config()
    if "input" not in config or "output" not in config:
        raise ValueError("Config must contain 'input:' and 'output:' top-level keys")
    in_path = Path(str(config["input"])).expanduser()
    out_path = Path(str(config["output"])).expanduser()

    signal, meta = _load_signal(in_path)
    logger.info(f"Loaded {in_path} shape={signal.shape} dtype={signal.dtype} backend={denoiser.backend}")
    cleaned = denoiser.apply(signal, **{k: v for k, v in meta.items() if k == "sample_rate"})
    _save_signal(out_path, cleaned, meta)
    logger.info(f"Wrote denoised output to {out_path}")


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
