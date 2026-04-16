#!/usr/bin/env bash
#
# Export sentence-transformers / encoder-style HF models to OpenVINO IR with
# tokenizer + pooling + L2 normalization fused, then write OVMS config files
# (CPU and GPU variants) into the deployment directory.
#
# This script is the single source of truth for which models are deployed.
# Add a model: append one line to MODELS below, re-run.
#
# Required tools (install once, the rest is offline):
#   pip install --user openvino openvino-tokenizers optimum[openvino] \
#                      sentence-transformers transformers torch pillow
#
# Usage:
#   ./setup-models.sh                           # default models, default location
#   OVMS_MODELS_DIR=/srv/ovms ./setup-models.sh
#   WEIGHT_FORMAT=int8 ./setup-models.sh        # smaller, slightly less accurate
#   FORCE=1 ./setup-models.sh                   # re-export even if files exist
#   MODEL_NAMES=minilm,mpnet ./setup-models.sh  # export selected models only
#
set -euo pipefail

OVMS_MODELS_DIR="${OVMS_MODELS_DIR:-$HOME/ovms-models}"
WEIGHT_FORMAT="${WEIGHT_FORMAT:-fp16}"
FORCE="${FORCE:-0}"
MODEL_NAMES="${MODEL_NAMES:-}"

# ---- Model catalog ---------------------------------------------------------
# Format: "local_name|hf_model_id|expected_dim"
# The local_name becomes the OVMS pipeline name suffix:
#   <local_name>_pipeline   — what the Java client connects to
# The expected_dim is informational only — actual dim is read from ModelMetadata.
MODELS=(
  "minilm|sentence-transformers/all-MiniLM-L6-v2|384"
  "mpnet|sentence-transformers/all-mpnet-base-v2|768"
  "e5_small|intfloat/e5-small-v2|384"
  "e5_large|intfloat/e5-large-v2|1024"
  "bge_m3|BAAI/bge-m3|1024"
)
# ----------------------------------------------------------------------------

if [[ -n "$MODEL_NAMES" ]]; then
  IFS=',' read -r -a requested_models <<< "$MODEL_NAMES"
  declare -A requested_by_name=()
  declare -A matched_by_name=()
  for requested in "${requested_models[@]}"; do
    trimmed="${requested//[[:space:]]/}"
    if [[ -n "$trimmed" ]]; then
      requested_by_name["$trimmed"]=1
    fi
  done
  filtered_models=()
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name _ <<< "$entry"
    for requested in "${!requested_by_name[@]}"; do
      if [[ "$name" == "$requested" ]]; then
        filtered_models+=("$entry")
        matched_by_name["$name"]=1
        break
      fi
    done
  done
  unknown_models=()
  for requested in "${!requested_by_name[@]}"; do
    if [[ -z "${matched_by_name[$requested]:-}" ]]; then
      unknown_models+=("$requested")
    fi
  done
  if [[ ${#unknown_models[@]} -gt 0 ]]; then
    echo "ERROR: unknown model name(s) in MODEL_NAMES: ${unknown_models[*]}" >&2
    exit 1
  fi
  MODELS=("${filtered_models[@]}")
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: no models selected. Check MODEL_NAMES='$MODEL_NAMES'." >&2
  exit 1
fi

echo "==> Models directory: $OVMS_MODELS_DIR"
echo "==> Weight format:    $WEIGHT_FORMAT"
echo "==> Models:"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name id dim <<< "$entry"
  printf "      %-12s %-50s (%s dim)\n" "$name" "$id" "$dim"
done
mkdir -p "$OVMS_MODELS_DIR"

export_pair() {
  local local_name="$1"
  local model_id="$2"

  local tok_dir="$OVMS_MODELS_DIR/tokenizer_${local_name}/1"
  local emb_dir="$OVMS_MODELS_DIR/embedding_${local_name}/1"

  if [[ "$FORCE" != "1" && -f "$tok_dir/openvino_tokenizer.xml" && -f "$emb_dir/openvino_model.xml" ]]; then
    echo "==> $local_name already exported, skipping (FORCE=1 to re-export)"
    return
  fi

  echo "==> [$local_name] exporting tokenizer for $model_id"
  rm -rf "$tok_dir" && mkdir -p "$tok_dir"
  convert_tokenizer -o "$tok_dir" "$model_id"

  echo "==> [$local_name] exporting embedding model for $model_id"
  rm -rf "$emb_dir" && mkdir -p "$emb_dir"
  optimum-cli export openvino \
    --model "$model_id" \
    --library sentence_transformers \
    --disable-convert-tokenizer \
    --task feature-extraction \
    --weight-format "$WEIGHT_FORMAT" \
    --trust-remote-code \
    "$emb_dir"

  # optimum-cli writes a bunch of HF tokenizer files alongside the IR — harmless
  # for OVMS but we trim them so the deployment dir only contains what OVMS uses.
  find "$emb_dir" -maxdepth 1 -type f \
    ! -name 'openvino_model.xml' \
    ! -name 'openvino_model.bin' \
    -delete
}

# ---- Export all models -----------------------------------------------------
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name id _ <<< "$entry"
  export_pair "$name" "$id"
done

# ---- Generate OVMS configs -------------------------------------------------
write_configs() {
  local cpu_path="$OVMS_MODELS_DIR/config-cpu.json"
  local gpu_path="$OVMS_MODELS_DIR/config-gpu.json"

  python3 - "$cpu_path" "$gpu_path" <<PY
import json, sys

models = [
$(for entry in "${MODELS[@]}"; do
    IFS='|' read -r name id dim <<< "$entry"
    printf '    ("%s", "%s", %s),\n' "$name" "$id" "$dim"
  done)
]

def make_config(target_device_for_embedding):
    model_config_list = []
    pipeline_config_list = []
    for local_name, hf_id, _ in models:
        model_config_list.append({
            "config": {
                "name": f"tokenizer_{local_name}",
                "base_path": f"/models/tokenizer_{local_name}",
                "target_device": "CPU",
            }
        })
        model_config_list.append({
            "config": {
                "name": f"embedding_{local_name}",
                "base_path": f"/models/embedding_{local_name}",
                "target_device": target_device_for_embedding,
            }
        })
        pipeline_config_list.append({
            "name": f"{local_name}_pipeline",
            "inputs": ["strings"],
            "nodes": [
                {
                    "name": "tokenizer_node",
                    "model_name": f"tokenizer_{local_name}",
                    "type": "DL model",
                    "inputs": [
                        {"Parameter_1": {"node_name": "request", "data_item": "strings"}}
                    ],
                    "outputs": [
                        {"data_item": "input_ids",      "alias": "input_ids"},
                        {"data_item": "attention_mask", "alias": "attention_mask"},
                    ],
                },
                {
                    "name": "embedding_node",
                    "model_name": f"embedding_{local_name}",
                    "type": "DL model",
                    "inputs": [
                        {"input_ids":      {"node_name": "tokenizer_node", "data_item": "input_ids"}},
                        {"attention_mask": {"node_name": "tokenizer_node", "data_item": "attention_mask"}},
                    ],
                    "outputs": [
                        {"data_item": "sentence_embedding", "alias": "sentence_embedding"}
                    ],
                },
            ],
            "outputs": [
                {"sentence_embedding": {"node_name": "embedding_node", "data_item": "sentence_embedding"}}
            ],
        })
    return {
        "model_config_list": model_config_list,
        "pipeline_config_list": pipeline_config_list,
    }

cpu_path, gpu_path = sys.argv[1], sys.argv[2]
with open(cpu_path, "w") as f:
    json.dump(make_config("CPU"), f, indent=2); f.write("\n")
with open(gpu_path, "w") as f:
    json.dump(make_config("GPU"), f, indent=2); f.write("\n")

print(f"  wrote {cpu_path}")
print(f"  wrote {gpu_path}")
print(f"  {len(models)} models → {len(models)} pipelines per config")
PY
}

echo "==> Generating OVMS configs"
write_configs

echo
echo "==> Done. Layout:"
find "$OVMS_MODELS_DIR" -maxdepth 3 -type f \( -name '*.xml' -o -name '*.json' \) | sort
echo
echo "Pipelines now available (use these as model_name in ModelInfer):"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name id dim <<< "$entry"
  printf "      %-20s -> %-50s (%s dim)\n" "${name}_pipeline" "$id" "$dim"
done
echo
echo "Next:"
echo "  cd $(dirname "$(dirname "$(readlink -f "$0")")")"
echo "  OVMS_MODELS_DIR=$OVMS_MODELS_DIR docker compose --profile cpu up -d"
echo "  OVMS_MODELS_DIR=$OVMS_MODELS_DIR docker compose --profile gpu up -d   # if Intel GPU"
echo "  ./scripts/verify.sh"
