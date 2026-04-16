package ai.pipestream.quarkus.openvino.runtime;

import ai.pipestream.module.embedder.spi.EmbeddingBackend;
import inference.MutinyGRPCInferenceServiceGrpc;
import io.quarkus.grpc.GrpcClient;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * OpenVINO Model Server backend. Implements the single
 * {@link EmbeddingBackend} SPI over a KServe v2 gRPC transport using the
 * Quarkus {@link GrpcClient} stack — Stork service discovery, TLS,
 * deadlines, and interceptors are all configurable via
 * {@code quarkus.grpc.clients.ovms.*} keys, not hand-rolled.
 *
 * <pre>
 * quarkus.grpc.clients.ovms.host=localhost
 * quarkus.grpc.clients.ovms.port=9001
 * # TLS:
 * quarkus.grpc.clients.ovms.tls-configuration-name=ovms-tls
 * quarkus.tls.ovms-tls.trust-store.p12.path=/etc/certs/ovms-trust.p12
 * # Stork service discovery:
 * quarkus.grpc.clients.ovms.name-resolver=stork
 * quarkus.stork.ovms.service-discovery.type=consul
 * </pre>
 *
 * <p><b>Fully reactive.</b> Every gRPC call on the hot path returns a
 * {@link Uni} — no {@code .await()}, no blocking stubs. Model metadata
 * discovery ({@code ModelMetadata} RPC) runs asynchronously and its
 * result is memoised per serving name so subsequent {@code embed()}
 * calls skip the RPC. {@link #supports(String)} is the one synchronous
 * method on this interface; it kicks off metadata discovery in the
 * background and returns optimistically — actual readiness is asserted
 * at {@code embed()} time, where a failure surfaces as a failed
 * {@code Uni} that the router can failover on.
 *
 * <p>Marked {@link Singleton} (not {@code @ApplicationScoped}) so ARC
 * does not generate a client proxy — required because
 * {@link EmbeddingBackend} is discovered via
 * {@code Instance<EmbeddingBackend>} across a Quarkus extension
 * classloader boundary, and client proxies can't cross that cleanly.
 */
@Singleton
public class OpenVinoBackend implements EmbeddingBackend {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoBackend.class);

    @Inject
    @GrpcClient("ovms")
    MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub;

    @ConfigProperty(name = "embedder.openvino.batch-size", defaultValue = "32")
    int batchSize;

    @ConfigProperty(name = "embedder.openvino.timeout-ms", defaultValue = "30000")
    int timeoutMs;

    /**
     * Per-servingName memoised {@code Uni<client>}. The {@code Uni} resolves
     * when {@code ModelMetadata} discovery completes; subsequent
     * subscriptions skip the RPC thanks to {@code memoize().indefinitely()}.
     */
    private final ConcurrentHashMap<String, Uni<OpenVinoMutinyStreamingBatchedClient>> clients = new ConcurrentHashMap<>();

    @Override
    public String name() {
        return "openvino";
    }

    @Override
    public boolean supports(String servingName) {
        if (servingName == null || servingName.isBlank()) {
            return false;
        }
        // Kick off (or reuse) the memoised discovery Uni. Don't await — fire
        // a fire-and-forget subscription so the cache warms up in background.
        // Actual success / failure is asserted at embed() time via the
        // failed-Uni path, which the router can failover on.
        clientUni(servingName).subscribe().with(
                c -> {},
                err -> log.debug("OpenVINO probe failed for '{}': {}", servingName, err.getMessage()));
        return true;
    }

    @Override
    public Uni<List<float[]>> embed(String servingName, List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        return clientUni(servingName).chain(client -> client.embed(texts));
    }

    private Uni<OpenVinoMutinyStreamingBatchedClient> clientUni(String servingName) {
        return clients.computeIfAbsent(servingName, sn -> {
            log.info("Registering OpenVINO client Uni for serving name '{}' (batch={}, timeout={}ms)",
                    sn, batchSize, timeoutMs);
            return OpenVinoMutinyStreamingBatchedClient.create(stub, sn, batchSize, timeoutMs)
                    .memoize().indefinitely();
        });
    }
}
