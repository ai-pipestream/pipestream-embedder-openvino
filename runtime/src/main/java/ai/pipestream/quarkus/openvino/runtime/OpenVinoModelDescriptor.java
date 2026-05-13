package ai.pipestream.quarkus.openvino.runtime;

import inference.GRPCInferenceServiceGrpc;
import inference.GrpcPredictV2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * Immutable snapshot of an OVMS model's tensor metadata, discovered via the
 * KServe v2 {@code ModelMetadata} RPC. Holds the model's declared input
 * tensor name, output tensor name, and embedding dimension so the batching
 * client doesn't have to hardcode any of them.
 *
 * <p><b>Blocking by design.</b> {@link #discover} runs a single short
 * {@code ModelMetadata} RPC under a per-call deadline and returns the
 * descriptor synchronously. Callers must already be on a virtual thread
 * (or an explicit worker thread); the SPI boundary in
 * {@link OpenVinoBackend} is responsible for routing the call onto a VT.
 *
 * <p>Models whose output has a dynamic ({@code -1}) hidden dimension are
 * rejected at discovery time because the client needs a static dim to
 * pre-size its float arrays.
 */
public final class OpenVinoModelDescriptor {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoModelDescriptor.class);

    private final String modelName;
    private final String inputTensorName;
    private final String outputTensorName;
    private final int dimensions;

    private OpenVinoModelDescriptor(String modelName, String inputTensorName, String outputTensorName, int dimensions) {
        this.modelName = modelName;
        this.inputTensorName = inputTensorName;
        this.outputTensorName = outputTensorName;
        this.dimensions = dimensions;
    }

    /**
     * Calls {@code ModelMetadata} synchronously and builds a descriptor for
     * the named model. Throws if the model is missing, has an unsupported
     * data type, or has a dynamic hidden dimension. The single RPC is
     * bounded by {@code timeoutMs} via {@code stub.withDeadlineAfter}.
     *
     * @param stub      blocking gRPC stub — typically obtained from
     *                  {@code GRPCInferenceServiceGrpc.newBlockingStub(channel)}
     *                  or injected via {@code @GrpcClient("ovms")} at the
     *                  blocking-stub type
     * @param modelName the pipeline or model name as registered in OVMS config
     * @param timeoutMs deadline for the one-shot metadata call, in ms
     */
    public static OpenVinoModelDescriptor discover(
            GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub,
            String modelName,
            int timeoutMs) {
        GrpcPredictV2.ModelMetadataRequest req = GrpcPredictV2.ModelMetadataRequest.newBuilder()
                .setName(modelName)
                .build();

        GrpcPredictV2.ModelMetadataResponse meta = stub
                .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
                .modelMetadata(req);

        return buildDescriptor(modelName, meta);
    }

    private static OpenVinoModelDescriptor buildDescriptor(String modelName, GrpcPredictV2.ModelMetadataResponse meta) {
        if (meta.getInputsCount() == 0) {
            throw new IllegalStateException("Model " + modelName + " has no inputs");
        }
        if (meta.getOutputsCount() == 0) {
            throw new IllegalStateException("Model " + modelName + " has no outputs");
        }

        GrpcPredictV2.ModelMetadataResponse.TensorMetadata input = meta.getInputs(0);
        GrpcPredictV2.ModelMetadataResponse.TensorMetadata output = meta.getOutputs(0);

        if (!"BYTES".equals(input.getDatatype())) {
            throw new IllegalStateException("Model " + modelName + " input " + input.getName()
                    + " has datatype " + input.getDatatype() + ", expected BYTES for string input");
        }
        if (!"FP32".equals(output.getDatatype())) {
            throw new IllegalStateException("Model " + modelName + " output " + output.getName()
                    + " has datatype " + output.getDatatype() + ", expected FP32 for embeddings");
        }

        int dims = extractHiddenDim(output);

        log.info("Discovered OpenVINO model {}: input='{}' output='{}' dimensions={}",
                modelName, input.getName(), output.getName(), dims);

        return new OpenVinoModelDescriptor(modelName, input.getName(), output.getName(), dims);
    }

    private static int extractHiddenDim(GrpcPredictV2.ModelMetadataResponse.TensorMetadata output) {
        int count = output.getShapeCount();
        if (count < 2) {
            throw new IllegalStateException("Output " + output.getName()
                    + " has shape rank " + count + ", expected at least 2 (batch, hidden_dim)");
        }
        long lastDim = output.getShape(count - 1);
        if (lastDim <= 0) {
            throw new IllegalStateException("Output " + output.getName()
                    + " has dynamic hidden dimension, cannot determine embedding size");
        }
        return (int) lastDim;
    }

    public String modelName() {
        return modelName;
    }

    public String inputTensorName() {
        return inputTensorName;
    }

    public String outputTensorName() {
        return outputTensorName;
    }

    public int dimensions() {
        return dimensions;
    }
}
