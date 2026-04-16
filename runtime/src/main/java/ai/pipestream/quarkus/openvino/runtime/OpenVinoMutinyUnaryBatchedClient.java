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
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public class OpenVinoMutinyUnaryBatchedClient {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoMutinyUnaryBatchedClient.class);

    private final MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub;
    private final OpenVinoModelDescriptor descriptor;
    private final int batchSize;
    private final Duration requestTimeout;

    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);
    private final AtomicLong totalTextsProcessed = new AtomicLong(0);

    public static Uni<OpenVinoMutinyUnaryBatchedClient> create(
            MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
            String modelName, int batchSize, int timeoutMs) {
        return OpenVinoModelDescriptor.discover(stub, modelName, timeoutMs)
                .map(descriptor -> new OpenVinoMutinyUnaryBatchedClient(stub, descriptor, batchSize, timeoutMs));
    }

    public OpenVinoMutinyUnaryBatchedClient(MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
                                            OpenVinoModelDescriptor descriptor,
                                            int batchSize, int timeoutMs) {
        this.stub = stub.withCompression("gzip");
        this.descriptor = descriptor;
        this.batchSize = batchSize;
        this.requestTimeout = Duration.ofMillis(timeoutMs);

        log.info("Created Mutiny Unary Batched client for model: {} (batch_size={}, dims={}, timeout={}ms)",
                descriptor.modelName(), batchSize, descriptor.dimensions(), timeoutMs);
    }

    public Uni<List<float[]>> embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        long startTime = System.currentTimeMillis();
        int numBatches = (texts.size() + batchSize - 1) / batchSize;

        return Uni.createFrom().item(() -> {
            List<float[]> allEmbeddings = new ArrayList<>(texts.size());

            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                int startIdx = batchIdx * batchSize;
                int endIdx = Math.min(startIdx + batchSize, texts.size());
                List<String> batchTexts = texts.subList(startIdx, endIdx);

                GrpcPredictV2.ModelInferRequest request = buildRequest(batchTexts);
                GrpcPredictV2.ModelInferResponse response = stub.modelInfer(request)
                        .await().atMost(requestTimeout);

                extractEmbeddings(response, batchTexts.size(), allEmbeddings);
            }

            long elapsedMs = System.currentTimeMillis() - startTime;
            totalRequests.addAndGet(numBatches);
            totalLatencyMs.addAndGet(elapsedMs);
            totalTextsProcessed.addAndGet(texts.size());

            log.debug("Unary batched embedding completed: {} texts in {} batches, {}ms",
                    texts.size(), numBatches, elapsedMs);

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
