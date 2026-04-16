package ai.pipestream.quarkus.openvino.runtime;

import ai.pipestream.module.embedder.spi.EmbeddingBackend;
import inference.MutinyGRPCInferenceServiceGrpc;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
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
 * <p><b>Reactive contract (honest version).</b> Every method on
 * {@link EmbeddingBackend} that does I/O returns a {@link Uni}. There
 * is no {@code .await()}, no blocking stub, no
 * {@code Uni.createFrom().item(() -> syncBlock)} wrapper anywhere on
 * the hot path. {@link #supports(String)} runs the real probe
 * reactively by chaining {@link OpenVinoModelDescriptor#discover} →
 * {@code map(c -> true)} with errors surfaced via {@code .invoke(log)}
 * and recovered to {@code false} — callers get a truthful probe
 * result, not an optimistic {@code true} with the actual probe fired
 * into the void.
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
    public Uni<Boolean> supports(String servingName) {
        if (servingName == null || servingName.isBlank()) {
            return Uni.createFrom().item(Boolean.FALSE);
        }
        // Honest probe: only "this specific model isn't served" signals
        // (gRPC NOT_FOUND / UNIMPLEMENTED) resolve to false. Every other
        // error (UNAVAILABLE, DEADLINE_EXCEEDED, INTERNAL, UNAUTHENTICATED,
        // PERMISSION_DENIED, plain RuntimeException, etc.) propagates so
        // the router, ops, and metrics see the real failure instead of a
        // silent "backend doesn't support this model" that leaves a sick
        // backend silently disabled forever.
        return clientUni(servingName)
                .map(c -> Boolean.TRUE)
                .onFailure(StatusRuntimeException.class).recoverWithUni(err -> {
                    StatusRuntimeException sre = (StatusRuntimeException) err;
                    Status.Code code = sre.getStatus().getCode();
                    if (code == Status.Code.NOT_FOUND || code == Status.Code.UNIMPLEMENTED) {
                        log.info("OVMS does not serve '{}': {}", servingName, sre.getStatus());
                        return Uni.createFrom().item(Boolean.FALSE);
                    }
                    return Uni.createFrom().failure(err);
                });
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
