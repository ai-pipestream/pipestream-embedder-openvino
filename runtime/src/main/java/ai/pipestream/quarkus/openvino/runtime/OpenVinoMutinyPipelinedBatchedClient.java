package ai.pipestream.quarkus.openvino.runtime;

import com.google.protobuf.ByteString;
import inference.GrpcPredictV2;
import inference.MutinyGRPCInferenceServiceGrpc;
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

/**
 * Batching client that submits up to {@code maxPipeline} {@code ModelInfer}
 * RPCs concurrently against an OVMS endpoint, then waits for that window to
 * complete before submitting the next window. Useful for keeping GPU / NPU
 * execution slots full when the per-batch latency leaves client-submission
 * headroom; has no benefit on fully-saturated CPU inference (see benchmark
 * table in {@code module-embedder/docs/openvino/README.md}).
 * <p>
 * Built on top of the Mutiny gRPC stub's async {@code Uni<ModelInferResponse>}
 * return type — each batch's Uni is constructed up-front without firing the
 * RPC; {@code Uni.combine().all().unis(window)} subscribes to all Unis in a
 * window concurrently, which is what actually fires N in-flight gRPC calls.
 */
public class OpenVinoMutinyPipelinedBatchedClient {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoMutinyPipelinedBatchedClient.class);

    private final MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub;
    private final OpenVinoModelDescriptor descriptor;
    private final int batchSize;
    private final int maxPipeline;
    private final Duration requestTimeout;

    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);
    private final AtomicLong totalTextsProcessed = new AtomicLong(0);

    public static Uni<OpenVinoMutinyPipelinedBatchedClient> create(
            MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
            String modelName, int batchSize, int maxPipeline, int timeoutMs) {
        return OpenVinoModelDescriptor.discover(stub, modelName, timeoutMs)
                .map(descriptor -> new OpenVinoMutinyPipelinedBatchedClient(stub, descriptor,
                        batchSize, maxPipeline, timeoutMs));
    }

    public OpenVinoMutinyPipelinedBatchedClient(MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
                                                OpenVinoModelDescriptor descriptor,
                                                int batchSize, int maxPipeline, int timeoutMs) {
        this.stub = stub.withCompression("gzip");
        this.descriptor = descriptor;
        this.batchSize = batchSize;
        this.maxPipeline = Math.max(1, maxPipeline);
        this.requestTimeout = Duration.ofMillis(timeoutMs);

        log.info("Created Mutiny Pipelined Batched client for model: {} (batch_size={}, max_pipeline={}, dims={})",
                descriptor.modelName(), batchSize, this.maxPipeline, descriptor.dimensions());
    }

    public Uni<List<float[]>> embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        final long startTime = System.currentTimeMillis();
        final int numBatches = (texts.size() + batchSize - 1) / batchSize;

        return Uni.createFrom().item(() -> {
            final float[][] result = new float[texts.size()][];

            // Build each batch's Uni from the Mutiny stub's async modelInfer
            // directly. Nothing fires until the Uni is subscribed, so this
            // loop is cheap (just proto construction and Uni composition).
            List<Uni<Void>> batchUnis = new ArrayList<>(numBatches);
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                final int startIdx = batchIdx * batchSize;
                final int endIdx = Math.min(startIdx + batchSize, texts.size());
                final List<String> batchTexts = texts.subList(startIdx, endIdx);
                final GrpcPredictV2.ModelInferRequest request = buildRequest(batchTexts);

                Uni<Void> batchUni = stub.modelInfer(request)
                        .ifNoItem().after(requestTimeout).failWith(() ->
                                new IllegalStateException("OVMS ModelInfer timed out after "
                                        + requestTimeout.toMillis() + "ms for " + descriptor.modelName()))
                        .invoke(response -> extractEmbeddingsInto(response, batchTexts.size(), result, startIdx))
                        .replaceWithVoid();
                batchUnis.add(batchUni);
            }

            // Pipeline window: subscribe to up to `maxPipeline` Unis at once,
            // wait for all of them, then advance. Subscription is what fires
            // the gRPC call, so this is genuinely concurrent.
            int submitted = 0;
            while (submitted < batchUnis.size()) {
                int windowEnd = Math.min(submitted + maxPipeline, batchUnis.size());
                List<Uni<Void>> window = batchUnis.subList(submitted, windowEnd);
                if (window.size() == 1) {
                    window.get(0).await().indefinitely();
                } else {
                    Uni.combine().all().unis(window).discardItems().await().indefinitely();
                }
                submitted = windowEnd;
            }

            List<float[]> allEmbeddings = Arrays.asList(result);

            long elapsedMs = System.currentTimeMillis() - startTime;
            totalRequests.addAndGet(numBatches);
            totalLatencyMs.addAndGet(elapsedMs);
            totalTextsProcessed.addAndGet(texts.size());

            log.debug("Pipelined batched embedding completed: {} texts in {} batches (pipeline={}), {}ms",
                    texts.size(), numBatches, maxPipeline, elapsedMs);

            return allEmbeddings;
        });
    }

    private GrpcPredictV2.ModelInferRequest buildRequest(List<String> batchTexts) {
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

        return GrpcPredictV2.ModelInferRequest.newBuilder()
                .setModelName(descriptor.modelName())
                .setModelVersion("")
                .addInputs(inputTensor)
                .addOutputs(outputTensor)
                .build();
    }

    private void extractEmbeddingsInto(GrpcPredictV2.ModelInferResponse response, int batchSize,
                                       float[][] dest, int destOffset) {
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
                dest[destOffset + i] = embedding;
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
                    dest[destOffset + i] = embedding;
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
