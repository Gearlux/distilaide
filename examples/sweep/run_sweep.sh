#!/usr/bin/env bash
# Train the SAME denoising config on every backend this machine has — the torch ones twice, once on
# the deepinv zoo wrapper and once on the shared DnCNN spec — and report the lot from one tracker.
#
#   examples/sweep/run_sweep.sh                     # every installed backend, into MLflow
#   examples/sweep/run_sweep.sh --tracker csv       # ...into the always-available CSV backend
#   examples/sweep/run_sweep.sh --only jax,mlx      # just those two
#   examples/sweep/run_sweep.sh --epochs 5          # the example's default is what the table used
#   examples/sweep/run_sweep.sh --dry-run           # print the matrix and the commands, run nothing
#
# ---------------------------------------------------------------------------
# What it is for
# ---------------------------------------------------------------------------
# The project's claim is that the config is the program: the same file trains under eight engines and
# only a `!scope:framework=` block changes. A sweep is that claim, executed — every run below is the
# SAME `examples/sweep/all_backends.yaml` with two flags changed, and the tracker at the end is where
# they become comparable.
#
# The second claim it executes is the SPEC one. `distilaide/dncnn/spec.py` says the five
# implementations ARE the same 57,635-parameter network; the unit tests check that against the
# derived count, and `--model dncnn` is what makes them TRAIN beside each other on one dataset.
#
# ---------------------------------------------------------------------------
# Read the table against the BASELINE, not against zero
# ---------------------------------------------------------------------------
# A restoration metric is never near zero: the NOISY INPUT ITSELF scores PSNR ~20.5 dB on this
# config, so that is the do-nothing floor every row has to beat. And the `*DnCNN` rows train from
# SCRATCH, which is why `--epochs 1` is a smoke test rather than a comparison — measured, one epoch
# of `--framework tensorflow` lands at 8.0 dB, i.e. still on its way UP to the identity, since
# predicting zero noise is a residual denoiser's cheapest local optimum and it takes a pass or two to
# find it. Use `--epochs 5` (the example's own default) for numbers worth comparing.
#
# ---------------------------------------------------------------------------
# Four things it does that are not obvious
# ---------------------------------------------------------------------------
#  1. INSTALLED, not declared. The CLI advertises all eight backends whether or not their framework
#     is importable, so the matrix is built by PROBING (`find_spec`) rather than by reading
#     `distilaide frameworks`. A machine with three extras runs three backends and says so, instead
#     of failing five times.
#  2. ONE PROCESS PER RUN, which is not a stylistic choice: Keras 3 binds its compute engine at
#     import time, so `keras` and `tensorflow` cannot both run in one interpreter. A subprocess per
#     run is what makes the sweep possible at all — and it is also what contains `import fastai`'s
#     process-wide switch of the multiprocessing start method to `fork`.
#  3. NO ROW PINS AN ENGINE, and that IS the notable part. The per-pixel sibling has to export
#     `KERAS_BACKEND=tensorflow` on its keras row, because `keras.metrics.MeanIoU` scatters per PIXEL
#     through a Python loop on the torch engine and the run looks hung. Restoration's metrics are
#     `keras.ops.psnr` and a mean squared error — no scatter, no per-pixel accumulation — so the
#     keras row runs on whichever engine is bound, which is the whole point of that backend. The
#     `tensorflow` row needs the pin and the CLI now sets it ITSELF (`cli._pin_keras_backend`), which
#     is a bug this sweep found: before that, every `distilaide train --framework tensorflow` on a
#     default checkout died at `_require_tensorflow_backend()`.
#  4. FAILURE IS ISOLATED. One backend dying must not abort the other ten — the point of a sweep is
#     the comparison, and a missing row is itself a result. Each failure is recorded and printed in
#     the summary with its log file.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # …/distilaide
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
CONFIG="examples/sweep/all_backends.yaml"

# The report is matrainer's, not this project's: naming an experiment and reading a sweep back are
# queries against a TRACKER, and three task projects carried byte-identical copies of them before the
# lift. What stays here is the matrix and the per-engine quirks — the halves that are genuinely this
# project's. matrainer ships no binary, so it is a module main.
REPORT=(python -m matrainer.sweep)

TRACKER="mlflow"
ONLY=""
DRY_RUN=0
EPOCHS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracker)  TRACKER="$2"; shift 2 ;;
    --only)     ONLY="$2"; shift 2 ;;
    --epochs)   EPOCHS="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)          echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# `project.bashrc` activates .venv AND exports the workspace .env — the mandated incantation, because
# the configs read `$DATA_ROOT`-style variables through confluid substitution.
#
# `set +u` around it on purpose: an activation script is a user profile, written against an
# interactive shell where referencing an unset variable is normal. Under `nounset` the first such
# reference aborts this script INSIDE the source, before a single line of its own output.
set +u
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/project.bashrc" >/dev/null 2>&1
set -u
cd "${PROJECT_DIR}" || exit 1

# ---------------------------------------------------------------------------
# The matrix: which backends are here, and what each can be given as a model
# ---------------------------------------------------------------------------
# ELEVEN cells rather than sixteen, and both gaps are structural (see `all_backends.yaml` for the
# full argument): there is no non-torch restoration ZOO to wrap — deepinv is torch-only, which is the
# whole reason the shared spec exists; and huggingface takes no dncnn, because a `Trainer`'s
# EVALUATION path calls `model(**inputs)` and `TorchDnCNN.forward(images)` takes no `pixel_values=`
# keyword (measured: it trains, then dies on the first eval).
#
# The env column is empty on every row, and that is a RESULT rather than an omission — see note 3 in
# the header. It stays in the format because a future engine-specific quirk belongs there and not in
# the loop below.
#
#   backend : import probes (comma-separated) : variants : env assignments (space-separated)
BACKENDS=(
  "lightning:pytorch_lightning:zoo dncnn:"
  "torch:torch:zoo dncnn:"
  "fastai:fastai:zoo dncnn:"
  "keras:keras:dncnn:"
  "tensorflow:tensorflow:dncnn:"
  "huggingface:transformers:zoo:"
  "jax:flax:dncnn:"
  "mlx:mlx:dncnn:"
)

installed() {
  # Every probe in the comma-separated list, or the backend is skipped.
  local probes="$1"
  python - "${probes}" <<'PY' 2>/dev/null
import importlib.util, sys

sys.exit(0 if all(importlib.util.find_spec(name) for name in sys.argv[1].split(",")) else 1)
PY
}

MATRIX=()
SKIPPED=()
for entry in "${BACKENDS[@]}"; do
  IFS=":" read -r backend probes variants env <<< "${entry}"
  if [[ -n "${ONLY}" && ",${ONLY}," != *",${backend},"* ]]; then continue; fi
  if ! installed "${probes}"; then
    SKIPPED+=("${backend} (no ${probes})")
    continue
  fi
  for variant in ${variants}; do MATRIX+=("${backend}|${variant}|${env}"); done
done

if [[ ${#MATRIX[@]} -eq 0 ]]; then
  echo "nothing to run: no selected backend has its framework installed." >&2
  [[ ${#SKIPPED[@]} -gt 0 ]] && printf '  skipped: %s\n' "${SKIPPED[@]}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# The tracker has to be up before the first run, not after the first failure
# ---------------------------------------------------------------------------
if [[ "${TRACKER}" == "mlflow" && ${DRY_RUN} -eq 0 ]]; then
  if ! aisland mlflow status 2>/dev/null | grep -qi "running"; then
    echo "==> starting MLflow (aisland mlflow start)"
    aisland mlflow start || { echo "could not start MLflow" >&2; exit 1; }
  fi
  echo "==> MLflow: $(aisland mlflow info 2>/dev/null | grep -i 'url' | head -1 | tr -s ' ')"
fi

# `Thu_Aug_6_2026_3` — the date, then which run of the sweep this is. The counter comes from the
# TRACKER (see `matrainer.sweep`), so a wiped tracker restarts at 1 rather than counting a history
# that is no longer there.
PREFIX="$(date '+%a_%b_%-d_%Y')"
if ! EXPERIMENT="$("${REPORT[@]}" next-experiment "${TRACKER}" "${PREFIX}")" || [[ -z "${EXPERIMENT}" ]]; then
  # Checked rather than assumed: an MLflow whose SQLite schema has never been created 500s on every
  # API call, and unguarded the empty name would flow into `--experiment_name ''` — eleven runs
  # training into an unnamed experiment.
  echo "could not name the experiment: the ${TRACKER} tracker did not answer." >&2
  [[ "${TRACKER}" == "mlflow" ]] && echo "  check: aisland mlflow status && aisland mlflow restart" >&2
  exit 1
fi

echo
echo "experiment : ${EXPERIMENT}"
echo "tracker    : ${TRACKER}"
echo "config     : ${CONFIG}"
echo "runs       : ${#MATRIX[@]}"
[[ ${#SKIPPED[@]} -gt 0 ]] && printf 'skipped    : %s\n' "${SKIPPED[*]}"
echo

LOG_DIR="sweep_logs/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"
PASSED=(); FAILED=()

for cell in "${MATRIX[@]}"; do
  IFS="|" read -r backend variant env <<< "${cell}"
  run_name="${backend}_${variant}"
  log="${LOG_DIR}/${run_name}.log"

  cmd=(distilaide train "${CONFIG}"
       --framework "${backend}"
       --tracker "${TRACKER}"
       --experiment_name "${EXPERIMENT}"
       --run_name "${run_name}")
  [[ "${variant}" == "dncnn" ]] && cmd+=(--model dncnn)
  [[ -n "${EPOCHS}" ]] && cmd+=(--max_epochs "${EPOCHS}")

  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '%s%s\n' "${env:+${env} }" "${cmd[*]}"
    continue
  fi

  printf '==> %-24s ' "${run_name}"
  start=${SECONDS}
  # `env` with no assignments is a no-op passthrough, so a row that ever needs a pin takes the same
  # code path as the ten that do not — and the variable stays scoped to the ONE subprocess.
  if env ${env} "${cmd[@]}" >"${log}" 2>&1; then
    printf 'ok   (%ds)\n' "$((SECONDS - start))"
    PASSED+=("${run_name}")
  else
    printf 'FAIL (%ds) — %s\n' "$((SECONDS - start))" "${log}"
    # The last non-empty line is almost always the exception; enough to triage without opening it.
    tail -n 20 "${log}" | grep -vE '^\s*$' | tail -n 1 | sed 's/^/      /'
    FAILED+=("${run_name}")
  fi
done

[[ ${DRY_RUN} -eq 1 ]] && exit 0

echo
echo "============================================================"
echo " ${EXPERIMENT} — ${#PASSED[@]} ok, ${#FAILED[@]} failed"
echo "============================================================"
[[ ${#FAILED[@]} -gt 0 ]] && printf '  failed: %s\n  logs:   %s/\n' "${FAILED[*]}" "${LOG_DIR}"
echo
"${REPORT[@]}" report "${TRACKER}" "${EXPERIMENT}"

# A failed run is a failed sweep for CI purposes, but only AFTER the report — the comparison is the
# deliverable and it is still worth reading when one backend is broken.
[[ ${#FAILED[@]} -eq 0 ]]
