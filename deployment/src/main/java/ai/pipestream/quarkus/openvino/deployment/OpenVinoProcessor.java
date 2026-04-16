package ai.pipestream.quarkus.openvino.deployment;

import ai.pipestream.quarkus.openvino.runtime.OpenVinoBackend;
import io.quarkus.arc.deployment.AdditionalBeanBuildItem;
import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.builditem.FeatureBuildItem;
import io.quarkus.deployment.builditem.IndexDependencyBuildItem;

/**
 * Build-time configuration for the OpenVINO Embeddings extension.
 *
 * <p>Registers {@link OpenVinoBackend} as an ARC-managed {@code @Singleton}
 * and indexes the {@code module-embedder-api} jar so ARC's bean-graph
 * builder can see the {@code EmbeddingBackend} interface at build time.
 *
 * <p>The gRPC transport is handled entirely by Quarkus's {@code @GrpcClient}
 * infrastructure — Stork service discovery, TLS, interceptors, deadlines
 * all configured via {@code quarkus.grpc.clients.ovms.*} config keys. No
 * hand-rolled {@code ManagedChannel}, no startup/shutdown lifecycle code.
 */
public class OpenVinoProcessor {

    private static final String FEATURE = "openvino-embeddings";

    @BuildStep
    FeatureBuildItem feature() {
        return new FeatureBuildItem(FEATURE);
    }

    @BuildStep
    IndexDependencyBuildItem indexEmbedderApi() {
        return new IndexDependencyBuildItem("ai.pipestream.module", "module-embedder-api");
    }

    @BuildStep
    AdditionalBeanBuildItem beans() {
        return AdditionalBeanBuildItem.builder()
                .addBeanClass(OpenVinoBackend.class)
                .setUnremovable()
                .build();
    }
}
