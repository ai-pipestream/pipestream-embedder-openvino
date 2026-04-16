package ai.pipestream.quarkus.openvino.runtime;

import inference.GRPCInferenceServiceGrpc;
import inference.GrpcPredictV2;
import io.grpc.Channel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * Immutable snapshot of an OVMS model's tensor metadata, discovered via the
 * KServe v2 {@code ModelMetadata} RPC at client construction time. Holds the
 * model's declared input tensor name, output tensor name, and embedding
 * dimension so the Mutiny batching clients don't have to hardcode any of
 * them — swap the pipeline name at the construction site and everything
 * else auto-adjusts.
 * <p>
 * The descriptor is deliberately read-only and allocation-free on the hot
 * path: {@link #discover(Channel, String, int)} makes exactly one gRPC call,
 * caches the results in a final record, and every subsequent {@code embed}
 * call reuses the cached strings and dimension. Models whose output has a
 * dynamic ({@code -1}) hidden dimension are rejected at discovery time
 * because the client needs a static dim to pre-size its float arrays.
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
     * Calls {@code ModelMetadata} on the given channel and builds a descriptor
     * for the named model. The call has a {@code timeoutMs} deadline.
     *
     * @param channel   an already-constructed {@link Channel} to OVMS
     * @param modelName the pipeline or model name as registered in OVMS config
     * @param timeoutMs deadline for the one-shot metadata call, in ms
     * @return a descriptor holding the first input tensor's name, the first
     *         output tensor's name, and the last dimension of the output shape
     * @throws IllegalStateException if the model has no inputs/outputs, if the
     *         input datatype is not {@code BYTES} (we only drive string-input
     *         pipelines), if the output datatype is not {@code FP32}, or if
     *         the output's last shape dimension is dynamic ({@code ≤0})
     */
    public static OpenVinoModelDescriptor discover(Channel channel, String modelName, int timeoutMs) {
        GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub =
                GRPCInferenceServiceGrpc.newBlockingStub(channel)
                        .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS);

        GrpcPredictV2.ModelMetadataResponse meta = stub.modelMetadata(
                GrpcPredictV2.ModelMetadataRequest.newBuilder()
                        .setName(modelName)
                        .build());

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

    /** The pipeline / model name as registered in OVMS, echoed back from the metadata response. */
    public String modelName() {
        return modelName;
    }

    /** Name of the first input tensor — used as the {@code name} field on every {@code ModelInferRequest.InferInputTensor}. */
    public String inputTensorName() {
        return inputTensorName;
    }

    /** Name of the first output tensor — used as the {@code name} field on every {@code ModelInferRequest.InferRequestedOutputTensor}. */
    public String outputTensorName() {
        return outputTensorName;
    }

    /** Embedding dimension (length of each {@code float[]} returned from an {@code embed} call). */
    public int dimensions() {
        return dimensions;
    }
}
