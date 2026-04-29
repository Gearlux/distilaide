"""List every registered denoising method and which backends are installed.

This example runs in CI without any optional extras — it inspects the
registry and prints availability flags. Use it as a sanity check after
installing extras (``[classical]`` / ``[deepinv]`` / ``[asteroid]``).
"""

from __future__ import annotations

from distilaide import list_methods


def main() -> None:
    rows = list_methods()
    width = max(len(r["method"]) for r in rows)
    print(f"{'method'.ljust(width)}  backend       installed")
    print(f"{'-' * width}  ------------  ---------")
    for row in rows:
        flag = "yes" if row["available"] else "no"
        print(f"{row['method'].ljust(width)}  {row['backend']:<12}  {flag}")
    print()
    print(f"{sum(1 for r in rows if r['available'])} of {len(rows)} methods available.")


if __name__ == "__main__":
    main()
