package ai.pipestream.quarkus.openvino.deployment;

import ai.pipestream.quarkus.openvino.runtime.OpenVinoBackend;
import ai.pipestream.quarkus.openvino.runtime.OpenVinoGrpcClient;
import io.quarkus.arc.deployment.AdditionalBeanBuildItem;
import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.builditem.FeatureBuildItem;
import io.quarkus.deployment.builditem.IndexDependencyBuildItem;

/**
 * Build-time configuration for the OpenVINO Embeddings extension.
 *
 * <p>Registers every runtime class the extension ships and tells ARC
 * to index the published {@code module-embedder-api} jar so the
 * {@code EmbeddingBackend} interface is visible to the bean-graph
 * builder:
 *
 * <ul>
 *   <li>{@link OpenVinoGrpcClient} — {@code @ApplicationScoped} gRPC
 *       channel holder with {@code @Observes StartupEvent /
 *       ShutdownEvent} wiring. Also declares the three
 *       {@code embedder.openvino.*} keys via {@code @ConfigProperty},
 *       which claims them for SmallRye validation.</li>
 *   <li>{@link OpenVinoBackend} — {@code @Singleton} implementation
 *       of {@code EmbeddingBackend}, discovered by module-embedder's
 *       {@code @Inject Instance<EmbeddingBackend>} injection point.
 *       Uses {@code @Singleton} (not {@code @ApplicationScoped})
 *       specifically to avoid the client-proxy class-cast trap when
 *       the SPI interface lives in a separate jar.</li>
 * </ul>
 *
 * <p>Both are marked unremovable because ARC cannot statically trace
 * their usages through the cross-jar CDI dependency chain — without
 * {@code setUnremovable()} the bean-graph pruning deletes them as
 * "unused" at build time and the extension silently does nothing.
 *
 * <p>{@link IndexDependencyBuildItem} on {@code module-embedder-api}
 * makes ARC scan the api jar's classes at build time so the
 * {@code implements EmbeddingBackend} relationship on
 * {@link OpenVinoBackend} is visible to the bean-graph builder. Without
 * this step, the extension would compile but ARC would not know the
 * backend is assignable to the SPI interface and wouldn't register it
 * against {@code @Inject Instance<EmbeddingBackend>} injection points.
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
                .addBeanClass(OpenVinoGrpcClient.class)
                .addBeanClass(OpenVinoBackend.class)
                .setUnremovable()
                .build();
    }
}
