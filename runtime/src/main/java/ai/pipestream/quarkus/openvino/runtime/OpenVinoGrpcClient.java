package ai.pipestream.quarkus.openvino.runtime;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * gRPC client for communicating with OpenVINO embedding service.
 * <p>
 * Manages the gRPC channel lifecycle and provides stub instances for
 * embedding requests.
 */
@ApplicationScoped
public class OpenVinoGrpcClient {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoGrpcClient.class);

    @ConfigProperty(name = "embedder.openvino.host", defaultValue = "localhost")
    String host;

    @ConfigProperty(name = "embedder.openvino.port", defaultValue = "9001")
    int port;

    @ConfigProperty(name = "embedder.openvino.deadline-seconds", defaultValue = "30")
    int deadlineSeconds;

    private ManagedChannel channel;

    void onStart(@Observes StartupEvent ev) {
        channel = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext()
                .build();
        log.info("OpenVINO gRPC client connected to {}:{}", host, port);
    }

    void onStop(@Observes ShutdownEvent ev) {
        if (channel != null) {
            try {
                channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
                log.info("OpenVINO gRPC channel shut down");
            } catch (InterruptedException e) {
                log.warn("gRPC channel shutdown interrupted", e);
            }
        }
    }

    public ManagedChannel getChannel() {
        return channel;
    }

    public int getDeadlineSeconds() {
        return deadlineSeconds;
    }
}
