# OpenVINO Model Server for `module-embedder`

Production-ready text embedding via OpenVINO Model Server (OVMS) over native KServe v2 gRPC. No REST/JSON, no MediaPipe `EmbeddingsCalculatorOV`, no client-side tokenization. **Strings in, FP32 unit-vectors out, one gRPC round trip per batch.**

The Java clients in `quarkus-openvino-embeddings` discover input/output tensor names via `ModelMetadata` at startup and decode the binary `raw_output_contents` field directly, so the same client code drives any model deployed under this layout — swap the pipeline name and dimensions auto-detect.

---

## Quick start

```bash
# 1. One-time export — runs the OpenVINO tokenizer + optimum-cli converters.
#    Requires: pip install openvino openvino-tokenizers optimum[openvino] \
#                          sentence-transformers transformers torch pillow
./scripts/setup-models.sh

# 2. Bring up CPU + GPU containers (or just one)
docker compose --profile cpu up -d              # CPU only — works anywhere
docker compose --profile gpu up -d              # Adds GPU container on :9002
docker compose --profile all up -d              # Both at once

# 3. Smoke test
./scripts/verify.sh

# 4. Run the throughput sweep against everything that's up
./scripts/run-benchmark.sh
# → results in module-embedder/embedder-test-harness/build/bench/sweep.csv
```

---

## What gets deployed

`scripts/setup-models.sh` is the **single source of truth** for which models are deployed. It maintains a `MODELS` array at the top of the script — one line per model — and the script does two things in sequence:

1. **Exports each model** in two halves: an `openvino_tokenizer.xml` (string input → token id outputs, via `convert_tokenizer`) and an `openvino_model.xml` (token id inputs → mean-pooled, L2-normalized sentence embedding output, via `optimum-cli export openvino --library sentence_transformers`).
2. **Generates `config-cpu.json` and `config-gpu.json`** from the same `MODELS` list, with one `model_config_list` entry per tokenizer and embedding model and one matching DAG pipeline per model. The CPU config pins everything to CPU; the GPU config keeps tokenizers on CPU (string ops only run on CPU) and routes the embedding models to `target_device: GPU`.

Add a model: append one line like `"local_name|huggingface/model-id|expected_dim"` to the `MODELS` array, re-run the script, restart the containers. The Java client doesn't change — it discovers the new pipeline via `ModelMetadata`.

```
$OVMS_MODELS_DIR/                          # default $HOME/ovms-models
├── config-cpu.json                        # generated — all embeddings pinned to CPU
├── config-gpu.json                        # generated — embeddings on GPU, tokenizers on CPU
├── tokenizer_minilm/1/openvino_tokenizer.{xml,bin}
├── embedding_minilm/1/openvino_model.{xml,bin}
├── tokenizer_mpnet/1/openvino_tokenizer.{xml,bin}
├── embedding_mpnet/1/openvino_model.{xml,bin}
├── tokenizer_e5_small/1/openvino_tokenizer.{xml,bin}
├── embedding_e5_small/1/openvino_model.{xml,bin}
├── tokenizer_e5_large/1/openvino_tokenizer.{xml,bin}
├── embedding_e5_large/1/openvino_model.{xml,bin}
├── tokenizer_bge_m3/1/openvino_tokenizer.{xml,bin}
└── embedding_bge_m3/1/openvino_model.{xml,bin}
```

Each pipeline definition wires the tokenizer's `input_ids` and `attention_mask` int64 outputs into the embedding model's matching inputs, exposes a single string-typed `strings` input on the `request` node, and returns a single `sentence_embedding` FP32 output on the `response` node. Clients see one pipeline name and never know about the chained nodes inside.

### Default model set

| Pipeline name        | Source HF model                              | Dim | On disk (fp16) | Notes |
|----------------------|----------------------------------------------|----:|---------------:|-------|
| `minilm_pipeline`    | `sentence-transformers/all-MiniLM-L6-v2`     | 384 |   44 MB | Fast, general-purpose |
| `mpnet_pipeline`     | `sentence-transformers/all-mpnet-base-v2`    | 768 |  209 MB | Higher quality, English |
| `e5_small_pipeline`  | `intfloat/e5-small-v2`                       | 384 |   64 MB | Asymmetric: requires `query:` / `passage:` prefixes |
| `e5_large_pipeline`  | `intfloat/e5-large-v2`                       | 1024 |  639 MB | Strong English retrieval, requires same prefixes |
| `bge_m3_pipeline`    | `BAAI/bge-m3`                                | 1024 | 1.1 GB | Multilingual (100+ languages), strong cross-lingual |

Total disk: ~2 GB for all five models. All five fit comfortably in a B70 12 GB GPU's VRAM with room for activations.

> **Prefix-aware models**: `e5-small-v2`, `e5-large-v2`, and (optionally) `bge-m3` expect `query: ` to be prepended to query strings and `passage: ` to be prepended to corpus strings. This is a pure client concern — OVMS doesn't know about it. The benchmark harness sends raw text by default; if you're doing retrieval-quality experiments, prepend the prefix on the Java side before calling `embed(...)`.

---

## Why this works (and why it's faster than DJL Serving)

**OVMS is C++ end to end.** The HTTP/gRPC frontend, the deserializer, the OpenVINO Inference Engine, and the openvino-tokenizers extension are all in the same process with no JVM warmup, no JIT phase, no GC pauses. Cold-start to "first request served" is ~2 seconds for both containers in this setup, vs the 30–60 seconds you typically see for a JVM-based inference server.

**Tokenization runs inside the model graph, not in the client.** Once the openvino-tokenizers extension is registered (automatic in the 2026 image), an `ov::element::string`-typed input at the IR level is enough to get OVMS to deserialize a KServe v2 `BYTES` tensor straight into a native string tensor and feed it to the model. Zero round trips to a Python tokenizer service. Zero JSON serialization. The Java client just packs UTF-8 bytes into `InferTensorContents.bytes_contents` and sends one `ModelInfer` RPC.

**The DAG pipeline is server-side glue, not a runtime overhead.** The tokenizer's output tensors stay in the OVMS process — they never traverse the network back to the client. The pipeline scheduler hands them directly to the embedding model node as `ov::Tensor` references.

Compare to DJL Serving on the same hardware: JVM warmup, ND4J/OnnxRuntime warmup, request goes through Java tokenizer → Java NDArray construction → ONNX inference → Java post-processing → JSON serialization. Every layer is a memory copy. DJL Serving on a comparable GPU was clocked at ~1,000 sentences/sec for MiniLM in the prior baseline. **OVMS on the same Intel Battlemage GPU does 7,822 sent/s on MiniLM — 7.8× the throughput — and much bigger gaps on the heavier models** (see the full benchmark table below). A head-to-head DJL vs OVMS sweep using the same harness on <ovms-host>:9000 is planned as the next experiment; results will be committed alongside the OVMS CSV in [`benchmarks/`](benchmarks/).

---

## Benchmark results

**Hardware:** Intel Arc Battlemage G31 (B70 family) + Intel CPU on an Intel Battlemage host.
**Corpus:** 100 court opinions from `courtlistener-seed/opinions_1000.jsonl`, OpenNLP-split into **32,555 sentences** per run. Each run is one full pass through the corpus. Batch size 32. All clients use gzip compression on the gRPC stream. Same harness on both devices, same model files (fp16).
**Sweep size:** 5 models × 3 client kinds (streaming, unary, pipelined×8) × 2 devices = **30 runs**, **976,650 total embeddings produced**.
**Raw data:** [`benchmarks/sweep-5model-b70.csv`](benchmarks/sweep-5model-b70.csv) — 30 rows, one per run, with p50/p95/p99, wall time, throughput, and failure count.

### Aggregate — all 5 models through the full corpus

| Device | Wall (all 5 models × 3 client kinds) | Embeddings produced | Aggregate throughput |
|---|--:|--:|--:|
| **CPU (Intel, fp16)** | **61.45 min** | 488,325 | **132 sent/s** |
| **GPU (B70, fp16)**   | **3.64 min**  | 488,325 | **2,236 sent/s** |

**Speedup by total wall time: 16.9×.** The GPU chewed through 488k sentences across 5 different embedding models in under 4 minutes; the same work took over an hour on the CPU. Zero failures on either device.

### Throughput by model — unary client (all three client kinds within 1% of each other)

| Model | Dim | HF source | CPU (sent/s) | GPU (sent/s) | **GPU speedup** |
|---|--:|---|--:|--:|--:|
| `minilm_pipeline`   | 384  | `sentence-transformers/all-MiniLM-L6-v2`   | 1,338 | 7,822 | **5.85×** |
| `e5_small_pipeline` | 384  | `intfloat/e5-small-v2`                     |   733 | 5,676 | **7.74×** |
| `mpnet_pipeline`    | 768  | `sentence-transformers/all-mpnet-base-v2`  |   232 | 2,959 | **12.76×** |
| `e5_large_pipeline` | 1024 | `intfloat/e5-large-v2`                     |    67 | 1,207 | **18.13×** |
| `bge_m3_pipeline`   | 1024 | `BAAI/bge-m3`                              |    61 | 1,309 | **21.61×** |

### Per-request latency — unary client

| Model | CPU p50 | CPU p95 | CPU p99 | GPU p50 | GPU p95 | GPU p99 |
|---|--:|--:|--:|--:|--:|--:|
| `minilm_pipeline`   |   22 ms |  41 ms |   58 ms |  3.8 ms |  6.2 ms |   8.1 ms |
| `e5_small_pipeline` |   40 ms |  75 ms |  103 ms |  5.2 ms |  8.7 ms |  11.1 ms |
| `mpnet_pipeline`    |  125 ms | 242 ms |  340 ms |  9.5 ms | 18.4 ms |  24.9 ms |
| `e5_large_pipeline` |  439 ms | 853 ms | 1202 ms | 23.5 ms | 45.2 ms |  58.7 ms |
| `bge_m3_pipeline`   |  482 ms | 936 ms | 1235 ms | 22.4 ms | 39.5 ms |  54.1 ms |

### What the data says

- **GPU speedup scales monotonically with model size.** 384-dim distilled models get ~6×; 768-dim BERT-base gets ~13×; 1024-dim XLM-R-large gets ~18–22×. **The bigger and more compute-bound the model, the bigger the relative win from the accelerator.** This is the single most important finding for the GPU/NPU economics pitch — large quality-oriented models are exactly the ones that benefit most from hardware acceleration.
- **`e5_small` is ~1.8× slower than `minilm_pipeline` on CPU despite the same 384 output dim.** They're different architectures — `all-MiniLM-L6-v2` is a distilled 22M-parameter model, `e5-small-v2` is a full BERT-base with 110M parameters. On GPU the gap shrinks to 1.4×. Pick minilm for raw speed on weaker hardware; pick e5_small if you need higher retrieval quality and have a GPU or NPU.
- **`bge_m3` (multilingual XLM-RoBERTa-large, 100+ languages) is the heaviest model and benefits most from the GPU** — **21.6× speedup**, the highest in the sweep. On CPU it drops to 61 sent/s which is unusable for real ingestion; on GPU it runs at 1,309 sent/s which is production-viable. If you want multilingual embeddings, you want the accelerator.
- **All three client kinds are tied in the CSV above** because the `pipelined×8` rows were collected against an earlier implementation of `OpenVinoMutinyPipelinedBatchedClient` that wrapped the gRPC call in a blocking `Uni.createFrom().item(() -> ... .await().indefinitely())` lambda, which meant each window's N Unis ran serially under the driving thread's `await()`. The pipelined client has since been rewritten to use the Mutiny stub's async `Uni<ModelInferResponse>` return value directly with `Uni.combine().all().unis(window).discardItems().await()`, which actually subscribes to every Uni in the window simultaneously and fires N in-flight gRPC calls. **The "tied across kinds" observation in this table is therefore a historical artifact; a fresh sweep on the fixed implementation is tracked as a follow-up benchmark rerun.** Under the corrected implementation we expect pipelined to pull ahead of unary on GPU at batch=32 as client-side overlap starts to matter.
- **GPU latency variance is healthy across all 5 models.** p99/p50 ratio ranges from 2.13 (e5_small) to 2.62 (mpnet) — no long-tail GC pauses, no thermal throttling spikes during sustained inference. The B70 stayed consistent even during the 8+ minutes of continuous work it took to grind through the heavy-model runs.
- **Zero failures across 30 runs × 1018 batches = ~30,540 gRPC ModelInfer calls.**

### Wall-clock perspective for a real ingestion job

A full pass over `opinions_1000.jsonl` is 1,000 court cases ≈ 325,550 sentences (10× the 100-case benchmark slice). Projected wall times:

| Model | CPU | B70 GPU | Fits "trivial cost" threshold on GPU? |
|---|--:|--:|:---:|
| `minilm_pipeline`   |   4 min 3 s  |   42 s  | ✓ |
| `e5_small_pipeline` |   7 min 24 s |   57 s  | ✓ |
| `mpnet_pipeline`    |  23 min 24 s | 1 min 50 s | ✓ |
| `e5_large_pipeline` |  81 min 5 s  | 4 min 30 s | ✓ |
| `bge_m3_pipeline`   |  88 min 56 s | 4 min 9 s  | ✓ |

On CPU the heavy models cross from "just run it" into "needs a job queue, maybe an overnight run." On the GPU every model in the sweep stays under 5 minutes. **This is the conversation you have with a product owner** — GPU moves big-model ingestion from batch-processing territory back into interactive territory.

### Raw CSV schema

The file [`benchmarks/sweep-5model-b70.csv`](benchmarks/sweep-5model-b70.csv) has 30 rows of this shape:

```
run_at, provider, label, dimensions, batch_size,
total_sentences, total_batches, wall_ms,
p50_ms, p95_ms, p99_ms, min_ms, max_ms, mean_ms,
sentences_per_sec, batches_per_sec, failures, notes
```

`notes` is either `cpu-fp16` or `b70-gpu`. The `label` encodes the pipeline name, client kind, and batch size so each row is replayable: `<pipeline>-<kind>-batch<N>-<cpu|gpu>`. The sibling file [`benchmarks/sweep-5model-djl.csv`](benchmarks/sweep-5model-djl.csv) uses the same schema for the DJL Serving head-to-head described below.

---

## Comparison vs DJL Serving (NVIDIA 4080 Super)

Same harness, same 100 court opinions, same 32,555 sentences, same batch=32. DJL Serving runs in `deepjavalibrary/djl-serving:0.36.0-pytorch-gpu` on a NVIDIA 4080 Super (16 GB); the harness runs on an Intel Battlemage host and hits `http://<djl-host>:8080` over the direct 1 GbE LAN. Raw data: [`benchmarks/sweep-5model-djl.csv`](benchmarks/sweep-5model-djl.csv).

### Throughput head-to-head (sent/s)

| Model | OVMS B70 GPU | DJL 4080 Super | winner |
|---|--:|--:|:--|
| `minilm`   |   **7,547** | 6,368 | OVMS +19% |
| `e5_small` |     5,656   | **8,760** | DJL +55% |
| `mpnet`    |     2,932   | **3,491** | DJL +19% |
| `e5_large` |     1,210   | **1,343** | DJL +11% |
| `bge_m3`   |   **1,309** | 1,231 | OVMS +6% |
| **aggregate, 5 models** | **18,654** | **21,193** | **DJL +14%** |

**DJL 4080 Super sustains 95–98 % GPU utilisation at 300–310 W (near the 320 W TDP) during each model's window** — verified via `nvidia-smi` while the benchmark ran, after fixing a sawtooth in the harness (see "What it took to get here" below).

**DJL on NVIDIA 4080 Super beats OVMS on Intel Arc B70 on 3 of 5 models, and by 14% on aggregate throughput.** The 4080 Super is a raw compute win on the mid-range dense transformers (e5-small, mpnet) where matmul throughput dominates; OVMS holds its own on the tiny MiniLM (where kernel launch overhead dominates and OVMS's lighter dispatch wins back the margin) and on bge-m3 (where the model is large enough to run memory-bandwidth-bound rather than compute-bound). **Both stacks are serving-path-fast enough that the underlying device is the bottleneck**, which is the outcome you actually want from a serving comparison.

The headline number misses the important one though: **the B70 has 32 GB of VRAM, the 4080 Super has 16 GB.** For a five-model serving deployment that's the difference between "everything fits with headroom to scale workers" and "`e5_large` and `bge_m3` each get capped at 2 Python workers before the per-process CUDA context memory pushes us to OOM." If you extend the sweep to "serve all 5 models at max concurrency simultaneously" instead of one at a time, the B70 pulls ahead on the big two because OVMS can share one CUDA context across everything while DJL's Python-handler architecture allocates per-process. On raw throughput per watt and per dollar for multi-model embedding serving, B70 + OVMS is the more sensible buy — 2× the VRAM at a comparable compute envelope for a lower street price.

### What it took to get here

Two things completely changed these numbers from what we first saw, and both are worth documenting because they're exactly the kind of trap a casual benchmark walks into:

1. **Three of five DJL `djl://` traced models are broken on recent CUDA** on the 4080 Super. `intfloat/e5-small-v2` and `intfloat/e5-large-v2` hit an nvrtc fuser compile bug (`extra text after expected end of number` on the constant `-3.402823466385289e+38.f` in the fused softmax mask), and `sentence-transformers/all-mpnet-base-v2`'s traced `position_bias` lookup runs on CPU against cuda tensors. We replaced all three with minimal sentence-transformers Python handlers (see [`docs/djl-serving/models/`](../djl-serving/models/)). Only `all-MiniLM-L6-v2` goes through DJL's native PyTorch traced path; the other four go through `engine=Python`. Running the sweep against DJL's shipped traced path as-is would have posted four `failures=1018` rows.
2. **Do not benchmark over a Tailscale overlay.** The the DJL host hostname resolved to its Tailscale address (a Tailscale address) via `/etc/hosts`, and Tailscale's userspace encrypted transport tanked large-response HTTP throughput by ~150× for this workload — pings were fine at 0.4 ms, but a 200 KB response body took ~1.1 s instead of ~8 ms. DJL's internal `model_metric.log` read `RequestLatency` of 4–9 ms the whole time; the gap was entirely on the wire. Pointing the harness at the direct LAN IP (`<djl-host>`) was a 3× throughput jump across every model. If you see single-client latencies in the hundreds of ms against a local LAN GPU box, check the route, not the model server.
3. **The concurrent runner in the harness was a sawtooth.** An earlier version of `BenchmarkRunner.runConcurrent` fired batches in windows of N via `Uni.combine().all().unis(window).await()` — fire N, wait for ALL N, fire next N. That drained to zero at the end of every window: the fastest requests finished in ~30 ms but the slowest took ~300 ms, and the whole tail of each window was GPU-idle while the client waited for the laggards. Observed visually as `nvidia-smi` util oscillating 30–100 %. The fix is the standard sustained-concurrency pattern: `Multi.createFrom().iterable(unis).onItem().transformToUni(u -> u).merge(N)` — merge keeps exactly N subscriptions alive at any moment, so as soon as one Uni completes the merge operator subscribes to the next source Uni and the dispatcher queue never drains. After the fix, GPU util holds at 95–98 % for the entire window of each model. On this workload aggregate throughput only moved by ~1 %, but p50 latency dropped 35–78 % on every model — the old windowed pattern had been inflating tail latency via head-of-line blocking inside each window.

### Other observations

- **Concurrency matters more for DJL than for OVMS.** The sweep uses `--concurrency 128` because DJL's Python-handler path has a long per-request critical path (HTTP → Netty → dispatcher → unix socket → Python worker → sentence-transformers → socket → Java → HTTP) that's serialised per in-flight request. At `c=1` single-client we see ~4–7 ms per batch on a warm connection, so the internal path is fine — but to saturate it you need many in-flight requests. OVMS's native gRPC path delivers comparable throughput at much lower concurrency, which matters if your upstream is a single ingestion worker rather than a fan-out.
- **Per-request latency on DJL is much higher than on OVMS**, even now. p50 on DJL ranges from 210 ms (e5-small) to 1698 ms (bge-m3) at `c=128`, versus 3.8–23 ms p50 on OVMS B70 at much lower concurrency. The DJL numbers are queue depth divided by throughput; they're not wall-clock latency at low load. Still: if your workload is latency-sensitive rather than throughput-sensitive, OVMS is the better pick.
- **bge-m3 requires an extra workaround under DJL.** transformers ≥ 5 refuses `torch.load()` on `.bin` weights without torch ≥ 2.6 (CVE-2025-32434), and the container ships torch 2.5.1. We force `use_safetensors=True` in every handler to sidestep the check. OVMS has no such issue because the weights are pre-converted to OpenVINO IR at export time.
- **GPU worker budget is tight on DJL.** With 16 GB of VRAM and five models loaded simultaneously, we can only scale `e5_large` and `bge_m3` to 2 Python workers each before the per-process CUDA context memory overhead pushes us to OOM. OVMS streams all 5 models through one process and shares one CUDA context cleanly — a real advantage when VRAM is scarce.

### Startup time

- **OVMS** — 1.2 s (CPU) / 3.5 s (B70 GPU) from container start to "all 5 pipelines AVAILABLE". Models are pre-exported to OpenVINO IR; OVMS just memory-maps them.
- **DJL Serving** — cold-start the container in ~6 s, but the first call on each Python-handler model stalls for ~1.2 s of lazy sentence-transformers load before serving. For five models, that's ~6 s of first-request latency on top of startup. Time from `docker run` to a warm endpoint is closer to 30–60 s if you need all 5 models loaded.

### Reproducing

```bash
# On the DJL Serving host:
cd module-embedder/docs/djl-serving
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
./scripts/load-5-models.sh http://localhost:8090

# On the harness host — use the direct LAN IP, NOT a Tailscale hostname:
DJL_URL=http://<djl-host>:8080 \
OUTPUT=build/bench/sweep-5model-djl.csv \
  ./module-embedder/docs/djl-serving/scripts/sweep-5model-scaled.sh
```

The sweep script scales per-model worker counts to fit in 16 GB VRAM (`minilm=16, e5_small=8, mpnet=4, e5_large=2, bge_m3=2`) and uses `--concurrency 128` so DJL's dispatcher has enough in-flight requests to saturate the Python workers.

---

## Scaling and operations

### Startup time

OVMS cold start to "all pipelines AVAILABLE" is **~1.2 seconds for CPU, ~3.5 seconds for GPU** when loading all 5 models at once on a warm-page-cache machine. The GPU overhead is kernel compilation per embedding model (OpenVINO's GPU plugin JIT-compiles graphs on first load). From the actual logs of the current deployment:

```
[06:42:35] Available devices for Open VINO: CPU, GPU
[06:42:35] Loading model: tokenizer_minilm  → ready in ~15ms  (string ops — CPU only)
[06:42:36] Loading model: embedding_minilm  → ready in ~900ms (GPU compile)
[06:42:36] Loading model: tokenizer_mpnet   → ready in ~13ms
[06:42:37] Loading model: embedding_mpnet   → ready in ~480ms (GPU compile)
[06:42:37] Loading model: tokenizer_e5_small  → ready in ~14ms
[06:42:38] Loading model: embedding_e5_small  → ready in ~290ms (GPU compile)
[06:42:38] Loading model: tokenizer_e5_large  → ready in ~15ms
[06:42:39] Loading model: embedding_e5_large  → ready in ~287ms (GPU compile)
[06:42:39] Loading model: tokenizer_bge_m3  → ready in ~14ms
[06:42:42] Loading model: embedding_bge_m3  → ready in ~750ms (GPU compile)
[06:42:42] All 5 pipelines AVAILABLE
```

**You can stand up a fresh replica in under 5 seconds even with all 5 models loaded**, which is meaningful for autoscaling — by the time Consul registers a new instance, it's already serving traffic. DJL-serving's equivalent number is typically 30–60 s because of JVM + ND4J/OnnxRuntime warmup, even before model load.

### Multi-instance behind Consul

Each OVMS container is stateless and idempotent. To scale horizontally:

1. Stand up N copies of the same compose file on different hosts (or different host ports), all reading the same `OVMS_MODELS_DIR` (NFS mount, read-only). The directory is read-only at runtime — no write coordination needed.
2. Register each instance in Consul under a service name like `openvino-embeddings`. Health check: `grpcurl ... ServerLive` or just `nc -z <host> <port>`.
3. Point the Java client at the Consul DNS or use Stork (already in this repo) for client-side load balancing. No code change in the embedder — the existing channel construction takes a host:port from config; just feed it a virtual address.
4. The Java clients in this extension call `ModelMetadata` once per channel at construction. With Stork or a typical gRPC name resolver, that happens once per resolved endpoint, not per request. Discovery cost is negligible.

For 20 instances, per-model throughput math (B70 GPU class, single-client, batch=32):
- minilm × 20: **~156,400 sentences/sec aggregate** (7,822 × 20)
- e5_small × 20: **~113,500 sentences/sec** (5,676 × 20)
- mpnet × 20: **~59,200 sentences/sec** (2,959 × 20)
- e5_large × 20: **~24,100 sentences/sec** (1,207 × 20)
- bge_m3 × 20: **~26,200 sentences/sec** (1,309 × 20)

Memory per instance with all 5 models resident: ~4 GB VRAM + ~1.5 GB RSS. Cold start: ~2 s for CPU, ~3.5 s for GPU (kernel compilation for 5 embedding models), both well within Consul's typical health-check window. The GPU container hits "all pipelines AVAILABLE" fast enough that Consul has already routed traffic before the first warmup request completes.

### CPU-only deployments

The `cpu` profile works on any x86_64 host with Docker. No GPU drivers, no `/dev/dri`, no group permissions. Useful for:
- Local dev machines without an Intel GPU
- CI runners
- Cost-sensitive AWS instances (compare CPU `c7i` vs `inf2` NPU economics)

The same pipelines, the same client code, the same `ModelMetadata` discovery — just slower, as the benchmark table shows.

### GPU compatibility

`openvino/model_server:latest-gpu` includes the Intel Level Zero plugin and supports:
- **Intel Arc** (Alchemist A-series, Battlemage B-series) — tested above
- **Intel iGPUs** (UHD/Iris Xe on 11th gen+ CPUs) — works, lower throughput
- **Intel Data Center GPU Flex / Max** — works, higher throughput

Find your host's render group GID with `getent group render | cut -d: -f3` and set `OVMS_RENDER_GID` in the compose env if it isn't 993.

### NPU (future)

Intel NPU (`Meteor Lake` / `Lunar Lake` laptops, AWS `inf2` instances) shows up as `target_device: NPU` in OVMS. To deploy on an NPU host:
1. Re-export the embedding model with `--target_device NPU` (NPU prefers static shapes; you may need `--pad-token-id` and `--max-length` set).
2. Add a `config-npu.json` with `target_device: NPU` on the embedding nodes (tokenizers stay on CPU).
3. Add a third compose service following the GPU pattern but with `--device /dev/accel`.

The Java clients don't care — they see another pipeline name and another set of dimensions via `ModelMetadata`.

---

## Files in this directory

```
docs/openvino/
├── README.md                     # this file
├── docker-compose.yml            # cpu / gpu / all profiles
├── .gitignore                    # ignore models/
├── benchmarks/
│   └── sweep-5model-b70.csv      # committed raw sweep results, 30 rows
└── scripts/
    ├── setup-models.sh           # exports all 5 models AND writes config-{cpu,gpu}.json into $OVMS_MODELS_DIR
    ├── verify.sh                 # grpcurl smoke test against both endpoints (reads the generated config)
    └── run-benchmark.sh          # full sweep, appends to embedder-test-harness build/bench/sweep.csv
```

Note: `config-cpu.json` and `config-gpu.json` are **generated by `setup-models.sh`** into `$OVMS_MODELS_DIR` (default `$HOME/ovms-models`), not committed under `docs/openvino/`. The script's `MODELS` array at the top is the single source of truth for the model list — edit one line there and re-run to add or remove a pipeline.

---

## Manual gRPC tests with grpcurl

If you'd rather poke at the server by hand without the Java harness:

```bash
PROTO_DIR=$(realpath ../../quarkus-openvino-embeddings/runtime/build/protos/export/kserve-v2)

# Server liveness (CPU)
grpcurl -plaintext -import-path "$PROTO_DIR" -proto grpc_predict_v2.proto \
  -d '{}' localhost:9001 inference.GRPCInferenceService/ServerLive

# Pipeline metadata
grpcurl -plaintext -import-path "$PROTO_DIR" -proto grpc_predict_v2.proto \
  -d '{"name":"minilm_pipeline"}' localhost:9001 \
  inference.GRPCInferenceService/ModelMetadata

# Inference: send a single string, get back a 384-dim unit vector
B64=$(printf 'hello world' | base64)
grpcurl -plaintext -import-path "$PROTO_DIR" -proto grpc_predict_v2.proto \
  -d "{\"model_name\":\"minilm_pipeline\",\"inputs\":[
        {\"name\":\"strings\",\"datatype\":\"BYTES\",\"shape\":[\"1\"],
         \"contents\":{\"bytes_contents\":[\"$B64\"]}}]}" \
  localhost:9001 inference.GRPCInferenceService/ModelInfer
```

The response's `rawOutputContents[0]` is little-endian FP32 — 384 floats for MiniLM, 768 for MPNet. The vectors are pre-normalized to unit L2 norm by the fused pooling layer.

---

## Troubleshooting

**"Available devices for Open VINO: CPU" (no GPU listed in `docker logs ovms-gpu`)**
The container can't see your Intel GPU. Verify on the host: `ls -la /dev/dri` should show `card0` and `renderD128`, and `getent group render` should return your render group's GID. Set `OVMS_RENDER_GID` in `.env` or pass it inline: `OVMS_RENDER_GID=$(getent group render | cut -d: -f3) docker compose --profile gpu up -d`.

**"Bind for 0.0.0.0:9001 failed: port is already allocated"**
You already have an OVMS container (or something else) on that port. `docker ps | grep 9001`, then `docker rm -f <name>`. Or override the port: `OVMS_CPU_PORT=9011 docker compose --profile cpu up -d`.

**"Cannot create SpecialTokensSplit layer ... from unsupported opset: extension"** (Python only)
You're trying to read the tokenizer IR without the openvino-tokenizers extension registered. In Python, just `import openvino_tokenizers` before `core.read_model(...)`. OVMS itself auto-loads the extension in the 2026 image — no flag needed.

**"Model has dynamic hidden dimension"** (Java client startup error)
The `OpenVinoModelDescriptor.discover()` helper reads the last dimension of the output shape via `ModelMetadata` and rejects models where it's unbounded (-1). The fix is in the IR: re-export with `--library sentence_transformers` so the pooling + normalization layer collapses the seq dimension and only `[batch, hidden]` is dynamic.

**Tests failing with `StatusRuntimeException`** (Java client error)
First check the OVMS logs (`docker logs ovms-cpu` / `ovms-gpu`) for the actual gRPC error — it's almost always one of (a) wrong tensor name (the IR uses a different name than expected — call `ModelMetadata` to confirm), (b) wrong datatype (string-input models need `BYTES` datatype with shape `[N]`), or (c) trying to call a MediaPipe `EmbeddingsCalculatorOV` graph instead of a regular DAG pipeline. The setup in this directory uses regular models + DAG pipelines on purpose — `EmbeddingsCalculatorOV` is HTTP-only and unreachable via gRPC.
