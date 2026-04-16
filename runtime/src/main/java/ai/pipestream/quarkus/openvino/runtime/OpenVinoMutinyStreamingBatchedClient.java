package ai.pipestream.quarkus.openvino.runtime;

import com.google.protobuf.ByteString;
import inference.GrpcPredictV2;
import inference.MutinyGRPCInferenceServiceGrpc;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public class OpenVinoMutinyStreamingBatchedClient {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoMutinyStreamingBatchedClient.class);

    private final MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub;
    private final OpenVinoModelDescriptor descriptor;
    private final int batchSize;
    private final Duration requestTimeout;

    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);
    private final AtomicLong totalTextsProcessed = new AtomicLong(0);

    /**
     * Asynchronously build a client — kicks off a {@code ModelMetadata} RPC
     * to discover tensor names + dims, then constructs the client. Never
     * blocks the caller.
     */
    public static Uni<OpenVinoMutinyStreamingBatchedClient> create(
            MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
            String modelName, int batchSize, int timeoutMs) {
        return OpenVinoModelDescriptor.discover(stub, modelName, timeoutMs)
                .map(descriptor -> new OpenVinoMutinyStreamingBatchedClient(stub, descriptor, batchSize, timeoutMs));
    }

    public OpenVinoMutinyStreamingBatchedClient(MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
                                                OpenVinoModelDescriptor descriptor,
                                                int batchSize, int timeoutMs) {
        this.stub = stub.withCompression("gzip");
        this.descriptor = descriptor;
        this.batchSize = batchSize;
        this.requestTimeout = Duration.ofMillis(timeoutMs);

        log.info("Created Mutiny Streaming Batched client for model: {} (batch_size={}, dims={}, timeout={}ms)",
                descriptor.modelName(), batchSize, descriptor.dimensions(), timeoutMs);
    }

    /**
     * Embed a list of texts via OVMS KServe v2. Returns a {@link Uni} that
     * never blocks on any thread — previously this wrapped a blocking
     * {@code .await()} inside a {@code Uni.createFrom().item(Supplier)},
     * which ran on whatever thread the subscriber was on, including the
     * Vert.x event loop ("cannot block event loop" fatal when called from
     * inside a reactive pipeline).
     *
     * <p>The current impl:
     * <ol>
     *   <li>Partitions {@code texts} into sub-batches of {@code batchSize}
     *       (typically just one because the embedder pipeline already
     *       sub-batches at the {@code batchSize=32} layer — but safe for
     *       direct callers with larger inputs).</li>
     *   <li>Fires each sub-batch via the reactive Mutiny stub
     *       ({@code stub.modelInfer(request)}), collecting responses as
     *       a {@code Multi}. Sub-batches run concurrently bounded by the
     *       Netty channel's max-concurrent-streams setting.</li>
     *   <li>Each sub-batch writes its slice of vectors into a shared
     *       {@code float[][]} at disjoint indices under a happens-before
     *       boundary from {@code collect().asList()}. Ordering is
     *       preserved by index, not arrival order.</li>
     * </ol>
     *
     * <p>Zero blocking calls, zero worker-pool pinning. Returns immediately
     * with a Uni that resolves when all sub-batches complete.
     */
    public Uni<List<float[]>> embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        final long startNanos = System.nanoTime();
        final int total = texts.size();
        final int numBatches = (total + batchSize - 1) / batchSize;
        final float[][] out = new float[total][];

        return Multi.createFrom().range(0, numBatches)
                .onItem().transformToUniAndMerge(batchIdx -> {
                    final int from = batchIdx * batchSize;
                    final int to = Math.min(from + batchSize, total);
                    final List<String> batchTexts = new ArrayList<>(texts.subList(from, to));

                    GrpcPredictV2.ModelInferRequest request = buildRequest(batchTexts);
                    return stub.modelInfer(request)
                            .ifNoItem().after(requestTimeout).fail()
                            .map(response -> {
                                List<float[]> slice = new ArrayList<>(batchTexts.size());
                                extractEmbeddings(response, batchTexts.size(), slice);
                                for (int i = 0; i < slice.size(); i++) {
                                    out[from + i] = slice.get(i);
                                }
                                return batchIdx;
                            });
                })
                .collect().asList()
                .map(ignored -> {
                    long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
                    totalRequests.addAndGet(numBatches);
                    totalLatencyMs.addAndGet(elapsedMs);
                    totalTextsProcessed.addAndGet(total);
                    if (log.isDebugEnabled()) {
                        log.debug("OV batched embed: {} texts / {} batches / {} ms",
                                total, numBatches, elapsedMs);
                    }
                    return Arrays.asList(out);
                });
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

    private void extractEmbeddings(GrpcPredictV2.ModelInferResponse response, int batchSize, List<float[]> out) {
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
                out.add(embedding);
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
                    out.add(embedding);
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
