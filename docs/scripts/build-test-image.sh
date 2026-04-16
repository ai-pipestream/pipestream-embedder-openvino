#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_REPO="${IMAGE_REPO:-ghcr.io/ai-pipestream/module-embedder-ovms-test}"
IMAGE_TAG="${IMAGE_TAG:-minilm-mpnet}"
MODELS_WORK_DIR="${MODELS_WORK_DIR:-/tmp/module-embedder-ovms-test-models}"
DOCKER_BUILD_CONTEXT="${DOCKER_BUILD_CONTEXT:-/tmp/module-embedder-ovms-test-build-context}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-test-image.txt"

echo "==> Preparing model exports for test image"
rm -rf "$MODELS_WORK_DIR" "$DOCKER_BUILD_CONTEXT"
mkdir -p "$MODELS_WORK_DIR" "$DOCKER_BUILD_CONTEXT/models"

echo "==> Running setup-models.sh"
OVMS_MODELS_DIR="$MODELS_WORK_DIR" MODEL_NAMES="minilm,mpnet" FORCE="${FORCE:-0}" "$SCRIPT_DIR/setup-models.sh"

echo "==> Keeping only minilm + mpnet artifacts"
cp -R \
  "$MODELS_WORK_DIR/tokenizer_minilm" \
  "$MODELS_WORK_DIR/embedding_minilm" \
  "$MODELS_WORK_DIR/tokenizer_mpnet" \
  "$MODELS_WORK_DIR/embedding_mpnet" \
  "$MODELS_WORK_DIR/config-cpu.json" \
  "$DOCKER_BUILD_CONTEXT/models/"

cat > "$DOCKER_BUILD_CONTEXT/Dockerfile" <<'EOF'
FROM openvino/model_server:latest
COPY models/ /models/
EOF

MODEL_SET_HASH="$(cat "$SCRIPT_DIR/setup-models.sh" "$DOCKER_BUILD_CONTEXT/models/config-cpu.json" "$SCRIPT_DIR/build-test-image.sh" "$REQUIREMENTS_FILE" | sha256sum | cut -c1-12)"
HASHED_TAG="${IMAGE_TAG}-${MODEL_SET_HASH}"
echo "$IMAGE_TAG" > "$DOCKER_BUILD_CONTEXT/image-tag.txt"
echo "$MODEL_SET_HASH" > "$DOCKER_BUILD_CONTEXT/model-set-hash.txt"
echo "$HASHED_TAG" > "$DOCKER_BUILD_CONTEXT/hashed-tag.txt"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  echo "==> Building image ${IMAGE_REPO}:${IMAGE_TAG}"
  docker build -t "${IMAGE_REPO}:${IMAGE_TAG}" -t "${IMAGE_REPO}:${HASHED_TAG}" "$DOCKER_BUILD_CONTEXT"

  echo "==> Built:"
else
  echo "==> Build context prepared (BUILD_IMAGE=0)"
fi
echo "    ${IMAGE_REPO}:${IMAGE_TAG}"
echo "    ${IMAGE_REPO}:${HASHED_TAG}"
