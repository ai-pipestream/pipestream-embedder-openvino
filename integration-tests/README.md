# OpenVINO Integration Tests

Plain JUnit5 tests that exercise the three OVMS gRPC client implementations
(`OpenVinoMutinyStreamingBatchedClient`, `OpenVinoMutinyUnaryBatchedClient`,
`OpenVinoMutinyPipelinedBatchedClient`) against a live OpenVINO Model Server.

These are **not** `@QuarkusIntegrationTest` tests — they don't need the embedder
Quarkus app to be built or running. They construct a `ManagedChannel` directly
and talk to OVMS, so they catch wire-level regressions that mocked tests would
miss.

## Prerequisites

OVMS must be up on `localhost:9001` with both DAG pipelines deployed:

- `minilm_pipeline` → `sentence-transformers/all-MiniLM-L6-v2`, **384 dim**
- `mpnet_pipeline`  → `sentence-transformers/all-mpnet-base-v2`, **768 dim**

One-shot setup from a fresh checkout:

```bash
cd module-embedder/docs/openvino
./scripts/setup-models.sh                  # ~5 min, downloads HF models, exports IRs
docker compose --profile cpu up -d         # CPU container on :9001
./scripts/verify.sh                        # grpcurl smoke test
```

Full setup + benchmark documentation: [`module-embedder/docs/openvino/README.md`](../../../module-embedder/docs/openvino/README.md).

## Running the tests

```bash
# All OpenVINO integration tests
./gradlew :quarkus-openvino-embeddings-integration-tests:test

# A single test
./gradlew :quarkus-openvino-embeddings-integration-tests:test \
    --tests OpenVinoBatchingClientsIT.testSemanticCosineSimilarity
```

## What's covered

The single test class `OpenVinoBatchingClientsIT` runs 11 tests:

| Test | What it asserts |
|---|---|
| `testModelDescriptorDiscovery` | `ModelMetadata` returns the discovered input/output names and 384-dim sentence embedding output for `minilm_pipeline`. |
| `testStreamingSingleBatch` | Streaming client returns one 384-dim unit vector per input string. |
| `testStreamingMultipleBatches` | 7 texts at batch size 3 split correctly into 3 gRPC calls; final embeddings preserve order. |
| `testUnaryMatchesStreaming` | Streaming and unary clients return bit-identical embeddings for the same input (cos sim > 0.9999). |
| `testPipelinedConcurrentBatches` | Pipelined client (depth 4) returns embeddings in correct order across 9 texts / 3 batches. |
| `testSemanticCosineSimilarity` | A dog sentence vs another dog sentence has higher cos sim than the dog sentence vs a quantum-physics sentence. |
| `testAllClientsAgree` | All three client kinds return identical embeddings (pairwise cos sim > 0.9999). |
| `testEmptyInput` | Empty list short-circuits without a gRPC call. |
| `testMetricsCollection` | `getTotalRequests()`, `getTotalTextsProcessed()`, latency, throughput counters update across calls. |
| `testMpnetPipeline` | Same suite reduced for the 768-dim MPNet pipeline — confirms `ModelMetadata` discovery is dimension-agnostic. |
| `testMinilmAndMpnetSideBySide` | Both pipelines work simultaneously through one channel; embeddings have the right dimensions and are non-degenerate. |

Each test creates its own `ManagedChannel` in `@BeforeEach` and tears it down in
`@AfterEach`, so they don't interfere. The `OpenVinoModelDescriptor` discovery
happens once per client construction (a `ModelMetadata` RPC), so total RPC
volume is small and the suite runs in ~1 second after OVMS is warm.

## What the tests don't do

- **No throughput benchmarking.** That lives in `embedder-test-harness` so it
  can run on demand against the court-opinions corpus (~32k sentences) without
  slowing CI. See `module-embedder/docs/openvino/README.md` for the benchmark
  setup and results table.
- **No Quarkus app boot.** These tests are intentionally decoupled from the
  embedder Quarkus application. They only depend on `quarkus-openvino-embeddings`
  runtime classes (the three client implementations + the descriptor), so they
  survive any future refactor of `Vectorizer` / `EmbeddingModelRecord`.

## Troubleshooting

**`StatusRuntimeException: UNAVAILABLE: io exception`** — OVMS isn't up on
:9001. Run `docker ps | grep ovms` and `./scripts/verify.sh` from
`module-embedder/docs/openvino/`.

**`StatusRuntimeException: NOT_FOUND: Model with requested name is not found`** —
The compose file is up but the pipelines didn't load. Check
`docker logs ovms-cpu` for `Pipeline ... state changed to: AVAILABLE`. If you
see "FAILED" instead, the model files probably aren't where the config expects
them — re-run `./scripts/setup-models.sh`.

**`Model X output Y has datatype STRING, expected FP32`** — You pointed at the
tokenizer model directly instead of the pipeline. The Java client's first arg is
the **pipeline name** (`minilm_pipeline`), not the underlying model name
(`embedding_minilm`).
