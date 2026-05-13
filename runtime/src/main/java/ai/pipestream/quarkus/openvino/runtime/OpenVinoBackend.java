package ai.pipestream.quarkus.openvino.runtime;

import ai.pipestream.module.embedder.spi.EmbeddingBackend;
import inference.GRPCInferenceServiceGrpc;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
 * <p><b>Concurrency model.</b> The {@link EmbeddingBackend} SPI still
 * returns {@link Uni} for now — the cross-repo flip to synchronous return
 * types is deferred. To stay off the event loop while keeping the SPI
 * shape, each Uni is created via {@code Uni.createFrom().item(supplier)}
 * and pinned to a per-bean virtual-thread executor with
 * {@code .runSubscriptionOn(...)}. The supplier itself is straight-line
 * blocking Java: blocking gRPC stub calls, {@code try/catch} on
 * {@link StatusRuntimeException}, plain {@code List<float[]>} returns
 * inside {@link OpenVinoStreamingBatchedClient}.
 *
 * <p><b>Honest probe semantics.</b> {@link #supports(String)} treats only
 * "this specific model isn't served" signals (gRPC {@code NOT_FOUND} /
 * {@code UNIMPLEMENTED}) as {@code false}. Every other gRPC error
 * ({@code UNAVAILABLE}, {@code DEADLINE_EXCEEDED}, {@code INTERNAL},
 * {@code UNAUTHENTICATED}, {@code PERMISSION_DENIED}, plain
 * {@code RuntimeException}) propagates so the router, ops, and metrics
 * see the real failure instead of a silent "backend doesn't support this
 * model" that leaves a sick backend disabled forever.
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

    /**
     * Per-bean virtual-thread executor used to drive the SPI's blocking
     * supplier off whatever thread the caller's reactive pipeline is on.
     * Keeps the event loop unblocked even when the caller subscribes to
     * the returned {@code Uni} from {@code @GrpcClient} or REST handlers
     * that haven't been migrated to {@code @RunOnVirtualThread} yet.
     */
    private static final ExecutorService VT_EXECUTOR = Executors.newVirtualThreadPerTaskExecutor();

    @Inject
    @GrpcClient("ovms")
    GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub;

    @ConfigProperty(name = "embedder.openvino.batch-size", defaultValue = "32")
    int batchSize;

    @ConfigProperty(name = "embedder.openvino.timeout-ms", defaultValue = "30000")
    int timeoutMs;

    /**
     * Per-servingName cache of resolved clients. Populated lazily by the
     * first {@link #supports(String)} or {@link #embed} call for a given
     * model. Race-tolerant init: two concurrent first-calls may both run
     * the {@code ModelMetadata} discovery RPC — the loser's client is
     * discarded (it's a snapshot, not a long-lived resource), the winner
     * is published via {@link ConcurrentHashMap#putIfAbsent}. Failed
     * discoveries are NOT cached; the caller may retry later.
     */
    private final ConcurrentHashMap<String, OpenVinoStreamingBatchedClient> clients = new ConcurrentHashMap<>();

    @Override
    public String name() {
        return "openvino";
    }

    @Override
    public Uni<Boolean> supports(String servingName) {
        if (servingName == null || servingName.isBlank()) {
            return Uni.createFrom().item(Boolean.FALSE);
        }
        return Uni.createFrom().item(() -> {
            try {
                clientFor(servingName);
                return Boolean.TRUE;
            } catch (StatusRuntimeException sre) {
                Status.Code code = sre.getStatus().getCode();
                if (code == Status.Code.NOT_FOUND || code == Status.Code.UNIMPLEMENTED) {
                    log.info("OVMS does not serve '{}': {}", servingName, sre.getStatus());
                    return Boolean.FALSE;
                }
                throw sre;
            }
        }).runSubscriptionOn(VT_EXECUTOR);
    }

    @Override
    public Uni<List<float[]>> embed(String servingName, List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        return Uni.createFrom().item(() -> {
            OpenVinoStreamingBatchedClient client = clientFor(servingName);
            return client.embed(texts);
        }).runSubscriptionOn(VT_EXECUTOR);
    }

    private OpenVinoStreamingBatchedClient clientFor(String servingName) {
        OpenVinoStreamingBatchedClient cached = clients.get(servingName);
        if (cached != null) {
            return cached;
        }
        log.info("Registering OpenVINO client for serving name '{}' (batch={}, timeout={}ms)",
                servingName, batchSize, timeoutMs);
        OpenVinoStreamingBatchedClient created = OpenVinoStreamingBatchedClient.create(
                stub, servingName, batchSize, timeoutMs);
        OpenVinoStreamingBatchedClient existing = clients.putIfAbsent(servingName, created);
        return existing != null ? existing : created;
    }
}
