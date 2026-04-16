package ai.pipestream.quarkus.openvino.runtime;

import ai.pipestream.module.embedder.spi.EmbeddingBackend;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * OpenVINO Model Server backend. Implements the single
 * {@link EmbeddingBackend} SPI over a KServe v2 gRPC transport.
 *
 * <p>Marked {@link Singleton} (not {@code @ApplicationScoped}) because
 * ARC does not generate a client proxy for {@code @Singleton} beans.
 * The proxy-less type is directly assignable to {@link EmbeddingBackend}
 * across classloader boundaries, which is the whole reason the SPI
 * lives in a separate published jar rather than inside module-embedder
 * core.
 *
 * <p>Registered at build time via the deployment processor's
 * {@code AdditionalBeanBuildItem} + {@code IndexDependencyBuildItem}
 * pair — the former marks the class unremovable so ARC bean-graph
 * pruning does not delete it, the latter tells ARC to scan the
 * {@code module-embedder-api} jar for the {@link EmbeddingBackend}
 * interface so the bean's implements relationship is visible.
 *
 * <p>One {@link OpenVinoMutinyStreamingBatchedClient} is lazily
 * constructed per distinct {@code servingName} and cached for the
 * lifetime of the bean. Client construction fetches model metadata
 * via a blocking gRPC stub, so the first {@link #supports(String)}
 * or {@link #embed(String, List)} call for a serving name must not
 * run on the Vert.x event loop. module-embedder's startup-phase
 * warm-up loop drives this on the main thread.
 */
@Singleton
public class OpenVinoBackend implements EmbeddingBackend {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoBackend.class);

    private static final int DEFAULT_BATCH_SIZE = 32;
    private static final int DEFAULT_TIMEOUT_MS = 30_000;

    private final OpenVinoGrpcClient grpcClient;
    private final ConcurrentHashMap<String, OpenVinoMutinyStreamingBatchedClient> clients = new ConcurrentHashMap<>();

    @Inject
    public OpenVinoBackend(OpenVinoGrpcClient grpcClient) {
        this.grpcClient = grpcClient;
    }

    @Override
    public String name() {
        return "openvino";
    }

    @Override
    public boolean supports(String servingName) {
        if (servingName == null || servingName.isBlank()) {
            return false;
        }
        try {
            getOrCreateClient(servingName);
            return true;
        } catch (Exception e) {
            log.debug("OpenVINO backend does not support serving name '{}': {}", servingName, e.getMessage());
            return false;
        }
    }

    @Override
    public Uni<List<float[]>> embed(String servingName, List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        return getOrCreateClient(servingName).embed(texts);
    }

    private OpenVinoMutinyStreamingBatchedClient getOrCreateClient(String servingName) {
        return clients.computeIfAbsent(servingName, sn -> {
            log.info("Creating OpenVINO client for serving name '{}' (batch={}, timeout={}ms)",
                    sn, DEFAULT_BATCH_SIZE, DEFAULT_TIMEOUT_MS);
            return new OpenVinoMutinyStreamingBatchedClient(
                    grpcClient.getChannel(), sn, DEFAULT_BATCH_SIZE, DEFAULT_TIMEOUT_MS);
        });
    }
}
