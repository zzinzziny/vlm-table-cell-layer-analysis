# Shared settings for the experiment scripts. Sourced, not executed.
#   PY        python for Qwen / Ministral / LLaVA runs and all scoring   (default: python)
#   PY_GEMMA  python with transformers 5.17 for Gemma-4 runs             (default: $PY)
#   LIMIT     if set, pass --limit $LIMIT to runners that support it (quick smoke test)
#   DRY_RUN=1 print the commands without running them

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CODE=$REPO/code
PY=${PY:-python}
PY_GEMMA=${PY_GEMMA:-$PY}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_TAGS="qwen35_9b qwen3vl_8b qwen25_7b ministral3_8b llava_ov2_8b gemma4_12b qwen3vl_32b gemma4_31b"

# model_info <tag> sets MODEL, GEO (token geometry of the item files), BACKEND, NT, PYBIN, BS
# NT: Qwen3.5-9B is the only thinking model; for it the chat template is called with enable_thinking=False.
#     For every other model --no-thinking-flag leaves the prompt unchanged (checked), so it is passed uniformly.
model_info() {
  PYBIN=$PY; NT="--no-thinking-flag"; BS=8
  case $1 in
    qwen35_9b)     MODEL=Qwen/Qwen3.5-9B;                                GEO=qwen35;     BACKEND=qwen; NT="" ;;
    qwen3vl_8b)    MODEL=Qwen/Qwen3-VL-8B-Instruct;                      GEO=qwen35;     BACKEND=qwen ;;
    qwen25_7b)     MODEL=Qwen/Qwen2.5-VL-7B-Instruct;                    GEO=qwen25;     BACKEND=qwen ;;
    ministral3_8b) MODEL=mistralai/Ministral-3-8B-Instruct-2512-BF16;    GEO=ministral3; BACKEND=mistral3 ;;
    llava_ov2_8b)  MODEL=lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct; GEO=llava_ov2;  BACKEND=llava_ov2 ;;
    gemma4_12b)    MODEL=google/gemma-4-12B-it;                          GEO=gemma4;     BACKEND=gemma; NT=""; PYBIN=$PY_GEMMA; BS=6 ;;
    qwen3vl_32b)   MODEL=Qwen/Qwen3-VL-32B-Instruct;                     GEO=qwen35;     BACKEND=qwen; BS=4 ;;
    gemma4_31b)    MODEL=google/gemma-4-31B-it;                          GEO=gemma4;     BACKEND=gemma; NT=""; PYBIN=$PY_GEMMA; BS=3 ;;
    *) echo "unknown model tag: $1 (one of: $MODEL_TAGS)" >&2; return 1 ;;
  esac
}

lim() { [[ -n "${LIMIT:-}" ]] && echo "--limit $LIMIT" || true; }

run() {  # run <dir> <command...>
  local dir=$1; shift
  echo "+ (cd ${dir#$REPO/} && $*)"
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0
  (cd "$dir" && eval "$@")
}
