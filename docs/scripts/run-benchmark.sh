#!/usr/bin/env bash
#
# Sweep all OVMS pipelines × all 3 client kinds × {cpu, gpu} and append rows to
# embedder-test-harness/build/bench/sweep.csv. Pipelines are discovered from
# $OVMS_MODELS_DIR/config-cpu.json so this script stays in sync with the models
# you actually deployed.
#
# Override the corpus-limit / batch / output via env vars; defaults match the
# numbers in the README's benchmark table.
#
set -euo pipefail

OVMS_HOST="${OVMS_HOST:-localhost}"
OVMS_CPU_PORT="${OVMS_CPU_PORT:-9001}"
OVMS_GPU_PORT="${OVMS_GPU_PORT:-9002}"
OVMS_MODELS_DIR="${OVMS_MODELS_DIR:-$HOME/ovms-models}"
CORPUS_LIMIT="${CORPUS_LIMIT:-100}"
BATCH="${BATCH:-32}"
WARMUP="${WARMUP:-3}"
OUTPUT="${OUTPUT:-build/bench/sweep.csv}"
KINDS="${KINDS:-streaming unary pipelined}"
PIPELINES_OVERRIDE="${PIPELINES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GRADLEW="$REPO_ROOT/gradlew"

if [[ ! -f "$OVMS_MODELS_DIR/config-cpu.json" ]]; then
  echo "ERROR: $OVMS_MODELS_DIR/config-cpu.json not found." >&2
  echo "Run scripts/setup-models.sh first." >&2
  exit 1
fi

if [[ -n "$PIPELINES_OVERRIDE" ]]; then
  PIPELINES="$PIPELINES_OVERRIDE"
else
  PIPELINES=$(python3 -c "
import json
with open('$OVMS_MODELS_DIR/config-cpu.json') as f:
    cfg = json.load(f)
print(' '.join(p['name'] for p in cfg['pipeline_config_list']))
")
fi

echo "==> Sweep config:"
echo "      pipelines:    $PIPELINES"
echo "      client kinds: $KINDS"
echo "      batch:        $BATCH  (corpus limit $CORPUS_LIMIT, warmup $WARMUP)"
echo "      output:       $REPO_ROOT/embedder-test-harness/$OUTPUT"
echo

run_sweep() {
  local label_suffix="$1"
  local notes="$2"
  local port="$3"

  if ! (echo > "/dev/tcp/$OVMS_HOST/$port") 2>/dev/null; then
    echo "==> $label_suffix endpoint $OVMS_HOST:$port not reachable, skipping"
    return
  fi

  for kind in $KINDS; do
    for pipeline in $PIPELINES; do
      echo "==> $label_suffix : $pipeline / $kind"
      ( cd "$REPO_ROOT" && "$GRADLEW" :embedder-test-harness:run --no-daemon --quiet --console=plain \
          --args="--host $OVMS_HOST --port $port --pipeline $pipeline --kind $kind \
                  --batch $BATCH --max-pipeline 8 --warmup $WARMUP \
                  --corpus-limit $CORPUS_LIMIT \
                  --label ${pipeline}-${kind}-batch${BATCH}-${label_suffix} \
                  --notes $notes \
                  --output $OUTPUT" ) | grep -E "throughput|latency|provider=|FAIL" || true
    done
  done
}

run_sweep cpu cpu-fp16 "$OVMS_CPU_PORT"
run_sweep gpu b70-gpu  "$OVMS_GPU_PORT"

echo
echo "==> Done. Results in $REPO_ROOT/embedder-test-harness/$OUTPUT"
