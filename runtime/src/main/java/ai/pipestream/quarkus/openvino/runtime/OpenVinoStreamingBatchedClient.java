package ai.pipestream.quarkus.openvino.runtime;

import com.google.protobuf.ByteString;
import inference.GRPCInferenceServiceGrpc;
import inference.GrpcPredictV2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Batched OVMS client over the KServe v2 blocking gRPC stub.
 *
 * <p><b>Concurrency model.</b> Sub-batches of {@code batchSize} are dispatched
 * as concurrent unary {@code ModelInfer} calls on virtual threads. The
 * gRPC channel's {@code max-concurrent-streams} setting bounds the
 * in-flight count; the JVM allocates one virtual thread per sub-batch and
 * parks each on its blocking gRPC call until the response arrives. The
 * single-batch fast path skips the executor entirely.
 *
 * <p>Compared to the prior Mutiny-based implementation
 * ({@code Multi.range(...).transformToUniAndMerge(...)}), this version is
 * straight-line blocking Java; the parallelism comes from virtual threads
 * + the gRPC channel's stream multiplexing, not from a reactive scheduler.
 * Per-call deadlines come from {@code stub.withDeadlineAfter(...)}.
 */
public class OpenVinoStreamingBatchedClient {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoStreamingBatchedClient.class);

    private final GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub;
    private final OpenVinoModelDescriptor descriptor;
    private final int batchSize;
    private final int timeoutMs;

    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);
    private final AtomicLong totalTextsProcessed = new AtomicLong(0);

    /**
     * Build a client — synchronously runs a {@code ModelMetadata} RPC to
     * discover tensor names + dims, then constructs the client. The caller
     * is responsible for being on a virtual thread (or worker thread); this
     * method blocks for the duration of the metadata RPC.
     */
    public static OpenVinoStreamingBatchedClient create(
            GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub,
            String modelName, int batchSize, int timeoutMs) {
        OpenVinoModelDescriptor descriptor = OpenVinoModelDescriptor.discover(stub, modelName, timeoutMs);
        return new OpenVinoStreamingBatchedClient(stub, descriptor, batchSize, timeoutMs);
    }

    public OpenVinoStreamingBatchedClient(GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub,
                                          OpenVinoModelDescriptor descriptor,
                                          int batchSize, int timeoutMs) {
        this.stub = stub.withCompression("gzip");
        this.descriptor = descriptor;
        this.batchSize = batchSize;
        this.timeoutMs = timeoutMs;

        log.info("Created Batched client for model: {} (batch_size={}, dims={}, timeout={}ms)",
                descriptor.modelName(), batchSize, descriptor.dimensions(), timeoutMs);
    }

    /**
     * Embed a list of texts via OVMS KServe v2. Blocks until every
     * sub-batch completes; returns the embeddings in input order.
     *
     * <p>Internally:
     * <ol>
     *   <li>Single-batch input: one {@code ModelInfer} RPC on the calling
     *       thread, no executor overhead.</li>
     *   <li>Multi-batch input: a per-call virtual-thread executor dispatches
     *       one sub-batch per VT. Each VT issues its own
     *       {@code stub.withDeadlineAfter(...).modelInfer(req)} call and
     *       writes the resulting embeddings into a shared {@code float[][]}
     *       at disjoint indices. Failures from any sub-batch are unwrapped
     *       and rethrown.</li>
     * </ol>
     */
    public List<float[]> embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }
        final long startNanos = System.nanoTime();
        final int total = texts.size();
        final int numBatches = (total + batchSize - 1) / batchSize;
        final float[][] out = new float[total][];

        if (numBatches == 1) {
            runOneBatch(texts, out, 0);
        } else {
            try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
                List<Future<?>> futures = new ArrayList<>(numBatches);
                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                    final int from = batchIdx * batchSize;
                    final int to = Math.min(from + batchSize, total);
                    final List<String> slice = new ArrayList<>(texts.subList(from, to));
                    futures.add(executor.submit(() -> {
                        runOneBatch(slice, out, from);
                        return null;
                    }));
                }
                awaitAll(futures);
            }
        }

        long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
        totalRequests.addAndGet(numBatches);
        totalLatencyMs.addAndGet(elapsedMs);
        totalTextsProcessed.addAndGet(total);
        if (log.isDebugEnabled()) {
            log.debug("OV batched embed: {} texts / {} batches / {} ms",
                    total, numBatches, elapsedMs);
        }
        return Arrays.asList(out);
    }

    private void runOneBatch(List<String> batchTexts, float[][] out, int writeFrom) {
        GrpcPredictV2.ModelInferRequest request = buildRequest(batchTexts);
        GrpcPredictV2.ModelInferResponse response = stub
                .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
                .modelInfer(request);
        extractEmbeddings(response, batchTexts.size(), out, writeFrom);
    }

    private static void awaitAll(List<Future<?>> futures) {
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (ExecutionException e) {
                Throwable cause = e.getCause() != null ? e.getCause() : e;
                if (cause instanceof RuntimeException re) {
                    throw re;
                }
                if (cause instanceof Error err) {
                    throw err;
                }
                throw new RuntimeException(cause);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while embedding", e);
            }
        }
    }

    private GrpcPredictV2.ModelInferRequest buildRequest(List<String> batchTexts) {
        GrpcPredictV2.ModelInferRequest.Builder requestBuilder = GrpcPredictV2.ModelInferRequest.newBuilder()
                .setModelName(descriptor.modelName())
                .setModelVersion("");

        GrpcPredictV2.InferTensorContents.Builder contentsBuilder = GrpcPredictV2.InferTensorContents.newBuilder();
        for (String text : batchTexts) {
            contentsBuilder.addBytesContents(ByteString.copyFromUtf8(text));
        }

        GrpcPredictV2.ModelInferRequest.InferInputTensor inputTensor =
                GrpcPredictV2.ModelInferRequest.InferInputTensor.newBuilder()
                        .setName(descriptor.inputTensorName())
                        .setDatatype("BYTES")
                        .addShape(batchTexts.size())
                        .setContents(contentsBuilder)
                        .build();

        GrpcPredictV2.ModelInferRequest.InferRequestedOutputTensor outputTensor =
                GrpcPredictV2.ModelInferRequest.InferRequestedOutputTensor.newBuilder()
                        .setName(descriptor.outputTensorName())
                        .build();

        return requestBuilder.addInputs(inputTensor).addOutputs(outputTensor).build();
    }

    private void extractEmbeddings(GrpcPredictV2.ModelInferResponse response,
                                   int batchSize, float[][] out, int writeFrom) {
        int dims = descriptor.dimensions();

        if (response.getRawOutputContentsCount() > 0) {
            ByteString raw = response.getRawOutputContents(0);
            FloatBuffer fb = raw.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            int expected = batchSize * dims;
            if (fb.remaining() < expected) {
                throw new IllegalStateException("Response raw_output_contents too small: got "
                        + fb.remaining() + " floats, expected " + expected);
            }
            for (int i = 0; i < batchSize; i++) {
                float[] embedding = new float[dims];
                fb.get(embedding);
                out[writeFrom + i] = embedding;
            }
            return;
        }

        if (response.getOutputsCount() > 0) {
            GrpcPredictV2.ModelInferResponse.InferOutputTensor tensor = response.getOutputs(0);
            if (tensor.hasContents()) {
                List<Float> fp = tensor.getContents().getFp32ContentsList();
                for (int i = 0; i < batchSize; i++) {
                    float[] embedding = new float[dims];
                    for (int j = 0; j < dims; j++) {
                        embedding[j] = fp.get(i * dims + j);
                    }
                    out[writeFrom + i] = embedding;
                }
                return;
            }
        }

        throw new IllegalStateException("ModelInfer response had neither raw_output_contents nor contents.fp32_contents");
    }

    public double getAverageLatencyMs() {
        long requests = totalRequests.get();
        if (requests == 0) return 0;
        return (double) totalLatencyMs.get() / requests;
    }

    public double getThroughputPerSec() {
        long latencyMs = totalLatencyMs.get();
        if (latencyMs == 0) return 0;
        return (totalTextsProcessed.get() * 1000.0) / latencyMs;
    }

    public long getTotalRequests() {
        return totalRequests.get();
    }

    public long getTotalTextsProcessed() {
        return totalTextsProcessed.get();
    }

    public OpenVinoModelDescriptor getDescriptor() {
        return descriptor;
    }
}
