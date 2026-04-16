#!/usr/bin/env bash
#
# Smoke-test every pipeline declared in $OVMS_MODELS_DIR/config-cpu.json.
# For each one, calls ModelMetadata to confirm tensor names + dimensions, then
# runs a single-sentence ModelInfer and validates the response shape and
# L2-norm of the returned vector.
#
# Usage:
#   ./verify.sh                          # tests both CPU (9001) and GPU (9002) if reachable
#   OVMS_CPU_PORT=9011 ./verify.sh
#
set -euo pipefail

OVMS_HOST="${OVMS_HOST:-localhost}"
OVMS_CPU_PORT="${OVMS_CPU_PORT:-9001}"
OVMS_GPU_PORT="${OVMS_GPU_PORT:-9002}"
OVMS_MODELS_DIR="${OVMS_MODELS_DIR:-$HOME/ovms-models}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(cd "$SCRIPT_DIR/../../../../quarkus-openvino-embeddings/runtime/build/protos/export/kserve-v2" 2>/dev/null && pwd || true)"

if [[ -z "$PROTO_DIR" || ! -f "$PROTO_DIR/grpc_predict_v2.proto" ]]; then
  echo "ERROR: KServe v2 proto file not found." >&2
  echo "Run './gradlew :quarkus-openvino-embeddings:fetchProtos' first to download it." >&2
  exit 1
fi

if ! command -v grpcurl >/dev/null 2>&1; then
  echo "ERROR: grpcurl not on PATH. Install: https://github.com/fullstorydev/grpcurl" >&2
  exit 1
fi

if [[ ! -f "$OVMS_MODELS_DIR/config-cpu.json" ]]; then
  echo "ERROR: $OVMS_MODELS_DIR/config-cpu.json not found." >&2
  echo "Run scripts/setup-models.sh first." >&2
  exit 1
fi

# Discover the set of pipelines from the generated config — single source of truth.
PIPELINES=$(python3 -c "
import json, sys
with open('$OVMS_MODELS_DIR/config-cpu.json') as f:
    cfg = json.load(f)
for p in cfg['pipeline_config_list']:
    print(p['name'])
")

probe_endpoint() {
  local label="$1"
  local port="$2"

  echo
  echo "============================================================"
  echo "  $label  ($OVMS_HOST:$port)"
  echo "============================================================"

  if ! grpcurl -plaintext -import-path "$PROTO_DIR" -proto grpc_predict_v2.proto \
       -d '{}' "$OVMS_HOST:$port" inference.GRPCInferenceService/ServerLive >/dev/null 2>&1; then
    echo "  SKIP — server not reachable"
    return
  fi

  for pipeline in $PIPELINES; do
    echo
    echo "  -- $pipeline metadata --"
    grpcurl -plaintext -import-path "$PROTO_DIR" -proto grpc_predict_v2.proto \
      -d "{\"name\":\"$pipeline\"}" \
      "$OVMS_HOST:$port" inference.GRPCInferenceService/ModelMetadata 2>&1 \
      | python3 -c "
import sys, json
try:
    m = json.loads(sys.stdin.read())
    print(f\"    input:  {m['inputs'][0]['name']:14s} {m['inputs'][0]['datatype']:6s} {m['inputs'][0]['shape']}\")
    print(f\"    output: {m['outputs'][0]['name']:14s} {m['outputs'][0]['datatype']:6s} {m['outputs'][0]['shape']}\")
except Exception as e:
    print(f'    METADATA FAILED: {e}')
    sys.exit(1)
"

    echo "  -- $pipeline single-sentence infer --"
    local b64
    b64="$(printf 'hello world' | base64)"
    local out
    if ! out=$(grpcurl -plaintext -import-path "$PROTO_DIR" -proto grpc_predict_v2.proto \
        -d "{\"model_name\":\"$pipeline\",\"inputs\":[{\"name\":\"strings\",\"datatype\":\"BYTES\",\"shape\":[\"1\"],\"contents\":{\"bytes_contents\":[\"$b64\"]}}]}" \
        "$OVMS_HOST:$port" inference.GRPCInferenceService/ModelInfer 2>&1); then
      echo "    INFER FAILED: $out"
      continue
    fi

    echo "$out" | python3 -c "
import sys, json, base64, struct
r = json.loads(sys.stdin.read())
out = r['outputs'][0]
shape = list(out['shape'])
raw = base64.b64decode(r['rawOutputContents'][0])
floats = struct.unpack(f'<{len(raw)//4}f', raw)
norm = sum(x*x for x in floats) ** 0.5
status = 'OK' if abs(norm - 1.0) < 1e-3 else f'FAIL: norm={norm:.6f}'
print(f'    shape={shape} L2_norm={norm:.6f}  first 4 dims={floats[:4]}  [{status}]')
"
  done
}

probe_endpoint "CPU"  "$OVMS_CPU_PORT"
probe_endpoint "GPU"  "$OVMS_GPU_PORT"
