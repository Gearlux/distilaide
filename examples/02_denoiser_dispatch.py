"""Construct a Denoiser without invoking any backend.

This example runs in CI without optional extras — it exercises the
configurable wrapper, dispatcher routing, and backend introspection
without actually loading bm3d/deepinv/asteroid. Useful as a smoke test
for the dispatcher itself.
"""

from __future__ import annotations

from distilaide import Denoiser, available_methods


def main() -> None:
    for method in ("bm3d", "noisereduce", "dncnn", "dccrn"):
        d = Denoiser(method=method, sigma=0.05)
        installed = "installed" if available_methods(method) else "extra missing"
        print(f"  {method:<14} → backend={d.backend:<12} ({installed})")


if __name__ == "__main__":
    main()
