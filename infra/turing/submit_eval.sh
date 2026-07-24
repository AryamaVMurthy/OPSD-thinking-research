#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 6 )); then
  echo "usage: $0 CONFIG EVAL_MODULE RUN_NAME [ADAPTER] [METHOD] [CHECKPOINT]" >&2
  exit 2
fi

CONFIG="$1"
EVAL_MODULE="$2"
RUN_NAME="$3"
ADAPTER="${4:-}"
METHOD="${5:-qwen3-instruct}"
CHECKPOINT="${6:-none}"

sbatch \
  --export=ALL,CONFIG="${CONFIG}",EVAL_MODULE="${EVAL_MODULE}",RUN_NAME="${RUN_NAME}",ADAPTER="${ADAPTER}",METHOD="${METHOD}",CHECKPOINT="${CHECKPOINT}" \
  infra/turing/run_eval.sbatch
