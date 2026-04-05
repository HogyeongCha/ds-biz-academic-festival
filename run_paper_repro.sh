#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.profile"

unset FAST_MODE
unset MAX_PRODUCTS
unset EVAL_USERS

export PAPER_REPRO=1
export USE_GEMINI=1
# The paper cites Gemini 2.0 Flash, but that API model ID is retired for new users.
# Use the closest currently-available Flash family model for reproducible reruns.
export GEMINI_MODEL="${GEMINI_MODEL_OVERRIDE:-gemini-2.5-flash}"
export EMBEDDING_BACKEND=sbert
export SBERT_MODEL="${SBERT_MODEL:-paraphrase-mpnet-base-v2}"
export AUG_COPIES=2
export PERSONA_BATCH_SIZE="${PERSONA_BATCH_SIZE:-50}"
export RUN_CLUSTER_GRID=1
export LGBM_MIN_POSITIVES="${LGBM_MIN_POSITIVES:-1}"
export LGBM_MIN_POSITIVE_RATE="${LGBM_MIN_POSITIVE_RATE:-0.0}"
export LLM_CACHE_PATH="${LLM_CACHE_PATH:-$PWD/pipeline/outputs/llm_cache.json}"

python3 run_all.py
