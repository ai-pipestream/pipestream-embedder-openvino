package ai.pipestream.quarkus.openvino.it;

import ai.pipestream.quarkus.openvino.runtime.OpenVinoModelDescriptor;
import ai.pipestream.quarkus.openvino.runtime.OpenVinoMutinyPipelinedBatchedClient;
import ai.pipestream.quarkus.openvino.runtime.OpenVinoMutinyStreamingBatchedClient;
import ai.pipestream.quarkus.openvino.runtime.OpenVinoMutinyUnaryBatchedClient;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests for OpenVINO batching clients against a real OVMS instance
 * serving the five-model DAG pipelines exported by
 * {@code module-embedder/docs/openvino/scripts/setup-models.sh}.
 * <p>
 * By default this class starts OVMS via testcontainers using the pre-baked
 * GHCR image and waits for both pipelines to become AVAILABLE.
 * <p>
 * For local development against an already-running OVMS, set either
 * {@code -Dovms.host/-Dovms.port} or {@code OVMS_HOST/OVMS_PORT}:
 * <pre>
 *   ./gradlew :quarkus-openvino-embeddings-integration-tests:test \
 *     -Dovms.host=localhost -Dovms.port=9001
 * </pre>
 */
@DisplayName("OpenVINO Batching Clients Integration Tests")
@Testcontainers
public class OpenVinoBatchingClientsIT {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoBatchingClientsIT.class);

    private static final String OVMS_HOST_PROPERTY = System.getProperty("ovms.host");
    private static final String OVMS_PORT_PROPERTY = System.getProperty("ovms.port");
    private static final String OVMS_HOST_ENV = System.getenv("OVMS_HOST");
    private static final String OVMS_PORT_ENV = System.getenv("OVMS_PORT");

    static {
        boolean hasHostProperty = OVMS_HOST_PROPERTY != null && !OVMS_HOST_PROPERTY.isBlank();
        boolean hasPortProperty = OVMS_PORT_PROPERTY != null && !OVMS_PORT_PROPERTY.isBlank();
        if (hasHostProperty ^ hasPortProperty) {
            throw new IllegalStateException("ovms.host and ovms.port system properties must be provided together.");
        }
    }

    private static final String EXTERNAL_HOST = firstNonBlank(
            OVMS_HOST_PROPERTY,
            OVMS_HOST_ENV);
    private static final int EXTERNAL_PORT = Integer.parseInt(firstNonBlank(
            OVMS_PORT_PROPERTY,
            OVMS_PORT_ENV,
            "9001"));
    private static final boolean USE_EXTERNAL_OVMS = EXTERNAL_HOST != null && !EXTERNAL_HOST.isBlank();
    private static final boolean BGE_M3_ENABLED = Boolean.parseBoolean(firstNonBlank(
            System.getProperty("ovms.enableBgeM3"),
            System.getenv("OVMS_ENABLE_BGE_M3"),
            "false"));

    @Container
    @SuppressWarnings("resource")
    private static final GenericContainer<?> ovms = newOvmsContainer();

    private static final String PIPELINE = "minilm_pipeline";
    private static final int EXPECTED_DIMS = 384;
    private static final String MPNET_PIPELINE = "mpnet_pipeline";
    private static final int MPNET_EXPECTED_DIMS = 768;
    private static final String BGE_M3_PIPELINE = "bge_m3_pipeline";
    private static final int BGE_M3_EXPECTED_DIMS = 1024;
    private static final int TIMEOUT_MS = 30_000;

    private ManagedChannel channel;

    private static GenericContainer<?> newOvmsContainer() {
        return new OvmsContainer()
                .withExposedPorts(9000)
                .withCommand("--config_path=/models/config-cpu.json", "--port=9000")
                // OVMS actually logs "Pipeline: <name> state changed to: AVAILABLE"
                // with a colon after "Pipeline", not a space — the earlier ".*Pipeline .*"
                // regex never matched and testcontainers timed out even though OVMS was
                // fully ready in under a second. Keep the count aligned with the number of
                // pipelines baked into the image.
                .waitingFor(Wait.forLogMessage(".*Pipeline:.*state changed to: AVAILABLE.*\\n", 2)
                        .withStartupTimeout(Duration.ofMinutes(2)));
    }

    private static final class OvmsContainer extends GenericContainer<OvmsContainer> {
        private OvmsContainer() {
            super(DockerImageName.parse("ghcr.io/ai-pipestream/module-embedder-ovms-test:minilm-mpnet"));
        }

        @Override
        public void start() {
            if (USE_EXTERNAL_OVMS) {
                log.info("Using external OVMS endpoint {}:{} (container startup bypassed)", EXTERNAL_HOST, EXTERNAL_PORT);
                return;
            }
            super.start();
        }

        @Override
        public void stop() {
            if (USE_EXTERNAL_OVMS) {
                return;
            }
            super.stop();
        }
    }

    @BeforeEach
    void setUp() {
        String host = USE_EXTERNAL_OVMS ? EXTERNAL_HOST : ovms.getHost();
        int port = USE_EXTERNAL_OVMS ? EXTERNAL_PORT : ovms.getMappedPort(9000);
        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (channel != null) {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    @DisplayName("Model descriptor discovery returns expected input/output/dims")
    void testModelDescriptorDiscovery() {
        OpenVinoModelDescriptor desc = OpenVinoModelDescriptor.discover(channel, PIPELINE, TIMEOUT_MS);
        assertEquals(PIPELINE, desc.modelName());
        assertEquals(EXPECTED_DIMS, desc.dimensions());
        assertNotNull(desc.inputTensorName());
        assertNotNull(desc.outputTensorName());
        log.info("Discovered pipeline: input={} output={} dims={}",
                desc.inputTensorName(), desc.outputTensorName(), desc.dimensions());
    }

    @Test
    @DisplayName("Streaming client: single batch produces 384-dim unit vectors")
    void testStreamingSingleBatch() {
        var client = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        List<float[]> embeddings = client.embed(List.of("hello world", "openvino is fast", "quick brown fox"))
                .await().indefinitely();

        assertEquals(3, embeddings.size());
        for (float[] v : embeddings) {
            assertEquals(EXPECTED_DIMS, v.length);
            assertNoNaN(v);
            assertUnitNorm(v);
        }
    }

    @Test
    @DisplayName("Streaming client: texts exceeding batchSize are split into multiple gRPC calls")
    void testStreamingMultipleBatches() {
        var client = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 3, TIMEOUT_MS);
        List<String> texts = List.of("a", "b", "c", "d", "e", "f", "g"); // 7 texts, batchSize=3 → 3 batches
        List<float[]> embeddings = client.embed(texts).await().indefinitely();

        assertEquals(7, embeddings.size());
        for (float[] v : embeddings) {
            assertEquals(EXPECTED_DIMS, v.length);
            assertUnitNorm(v);
        }
        assertTrue(client.getTotalRequests() >= 1);
        assertEquals(7, client.getTotalTextsProcessed());
    }

    @Test
    @DisplayName("Unary client: produces the same embedding as streaming client for identical input")
    void testUnaryMatchesStreaming() {
        var streaming = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        var unary = new OpenVinoMutinyUnaryBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);

        List<String> texts = List.of("the weather is nice today");
        float[] a = streaming.embed(texts).await().indefinitely().get(0);
        float[] b = unary.embed(texts).await().indefinitely().get(0);

        assertEquals(EXPECTED_DIMS, a.length);
        assertEquals(EXPECTED_DIMS, b.length);
        double cos = cosineSimilarity(a, b);
        log.info("streaming vs unary cos sim: {}", cos);
        assertTrue(cos > 0.9999, "streaming and unary clients must produce identical embeddings, got cos=" + cos);
    }

    @Test
    @DisplayName("Pipelined client: concurrent batches return correct embeddings in order")
    void testPipelinedConcurrentBatches() {
        var client = new OpenVinoMutinyPipelinedBatchedClient(channel, PIPELINE, 3, 4, TIMEOUT_MS);
        List<String> texts = List.of(
                "alpha", "bravo", "charlie", "delta",
                "echo", "foxtrot", "golf", "hotel", "india");
        List<float[]> embeddings = client.embed(texts).await().indefinitely();

        assertEquals(9, embeddings.size());
        for (float[] v : embeddings) {
            assertEquals(EXPECTED_DIMS, v.length);
            assertUnitNorm(v);
        }
    }

    @Test
    @DisplayName("Semantic check: related sentences have higher cosine similarity than unrelated")
    void testSemanticCosineSimilarity() {
        var client = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        List<String> texts = List.of(
                "the quick brown fox jumps over the lazy dog",
                "another sentence about dogs and pets",
                "a completely different topic about quantum physics");
        List<float[]> e = client.embed(texts).await().indefinitely();

        double dogSim = cosineSimilarity(e.get(0), e.get(1));
        double unrelatedSim = cosineSimilarity(e.get(0), e.get(2));
        log.info("cos(dogs_a, dogs_b) = {}  cos(dogs, physics) = {}", dogSim, unrelatedSim);
        assertTrue(dogSim > unrelatedSim,
                "dog-related sentences should be more similar (" + dogSim
                        + ") than a dog sentence vs a physics sentence (" + unrelatedSim + ")");
        assertTrue(dogSim > 0.15, "dog-related pair should have meaningful similarity, got " + dogSim);
    }

    @Test
    @DisplayName("Compare all three clients: same text produces the same embedding from each")
    void testAllClientsAgree() {
        var streaming = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        var unary = new OpenVinoMutinyUnaryBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        var pipelined = new OpenVinoMutinyPipelinedBatchedClient(channel, PIPELINE, 8, 2, TIMEOUT_MS);

        List<String> texts = List.of("shared input across all three clients");
        float[] s = streaming.embed(texts).await().indefinitely().get(0);
        float[] u = unary.embed(texts).await().indefinitely().get(0);
        float[] p = pipelined.embed(texts).await().indefinitely().get(0);

        assertTrue(cosineSimilarity(s, u) > 0.9999);
        assertTrue(cosineSimilarity(s, p) > 0.9999);
        assertTrue(cosineSimilarity(u, p) > 0.9999);
    }

    @Test
    @DisplayName("Empty input returns empty list (no gRPC call)")
    void testEmptyInput() {
        var client = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        List<float[]> result = client.embed(List.of()).await().indefinitely();
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("MPNet pipeline: 768-dim sentence embeddings, discovery + semantic check")
    void testMpnetPipeline() {
        OpenVinoModelDescriptor desc = OpenVinoModelDescriptor.discover(channel, MPNET_PIPELINE, TIMEOUT_MS);
        assertEquals(MPNET_PIPELINE, desc.modelName());
        assertEquals(MPNET_EXPECTED_DIMS, desc.dimensions());

        var client = new OpenVinoMutinyStreamingBatchedClient(channel, MPNET_PIPELINE, 8, TIMEOUT_MS);
        List<String> texts = List.of(
                "the quick brown fox jumps over the lazy dog",
                "another sentence about dogs and pets",
                "a completely different topic about quantum physics");
        List<float[]> e = client.embed(texts).await().indefinitely();

        assertEquals(3, e.size());
        for (float[] v : e) {
            assertEquals(MPNET_EXPECTED_DIMS, v.length);
            assertNoNaN(v);
            assertUnitNorm(v);
        }

        double dogSim = cosineSimilarity(e.get(0), e.get(1));
        double unrelatedSim = cosineSimilarity(e.get(0), e.get(2));
        log.info("mpnet cos(dogs_a, dogs_b) = {}  cos(dogs, physics) = {}", dogSim, unrelatedSim);
        assertTrue(dogSim > unrelatedSim,
                "dog-related sentences should be more similar than a dog sentence vs a physics sentence");
    }

    @Test
    @DisplayName("MiniLM and MPNet produce independent embeddings of different dimensions")
    void testMinilmAndMpnetSideBySide() {
        var minilm = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        var mpnet = new OpenVinoMutinyStreamingBatchedClient(channel, MPNET_PIPELINE, 8, TIMEOUT_MS);

        assertEquals(EXPECTED_DIMS, minilm.getDescriptor().dimensions());
        assertEquals(MPNET_EXPECTED_DIMS, mpnet.getDescriptor().dimensions());

        List<String> texts = List.of("the quick brown fox", "another completely unrelated topic");
        List<float[]> a = minilm.embed(texts).await().indefinitely();
        List<float[]> b = mpnet.embed(texts).await().indefinitely();

        assertEquals(EXPECTED_DIMS, a.get(0).length);
        assertEquals(MPNET_EXPECTED_DIMS, b.get(0).length);

        // Both models should agree on which pair is more similar (cos sim magnitudes differ
        // because they are different model families, but the ordering should be consistent).
        // Here we only assert both sides of the pair are distinct (not identical embeddings).
        assertTrue(cosineSimilarity(a.get(0), a.get(1)) < 0.95,
                "minilm embeddings for different sentences should not be nearly-identical");
        assertTrue(cosineSimilarity(b.get(0), b.get(1)) < 0.95,
                "mpnet embeddings for different sentences should not be nearly-identical");
        log.info("minilm dims={} mpnet dims={}", a.get(0).length, b.get(0).length);
    }

    @Test
    @DisplayName("BGE-M3 pipeline (1024 dim) — confirms descriptor handles non-{384,768} dims")
    void testBgeM3Pipeline() {
        Assumptions.assumeTrue(BGE_M3_ENABLED,
                "BGE-M3 is not included in the default minilm+mpnet OVMS test image. "
                        + "Enable with -Dovms.enableBgeM3=true against an OVMS deployment that includes bge_m3_pipeline.");
        OpenVinoModelDescriptor desc = OpenVinoModelDescriptor.discover(channel, BGE_M3_PIPELINE, TIMEOUT_MS);
        assertEquals(BGE_M3_PIPELINE, desc.modelName());
        assertEquals(BGE_M3_EXPECTED_DIMS, desc.dimensions());

        var client = new OpenVinoMutinyStreamingBatchedClient(channel, BGE_M3_PIPELINE, 8, TIMEOUT_MS);
        List<String> texts = List.of(
                "the quick brown fox jumps over the lazy dog",
                "another sentence about dogs and pets",
                "a completely different topic about quantum physics");
        List<float[]> e = client.embed(texts).await().indefinitely();

        assertEquals(3, e.size());
        for (float[] v : e) {
            assertEquals(BGE_M3_EXPECTED_DIMS, v.length);
            assertNoNaN(v);
            assertUnitNorm(v);
        }

        double dogSim = cosineSimilarity(e.get(0), e.get(1));
        double unrelatedSim = cosineSimilarity(e.get(0), e.get(2));
        log.info("bge-m3 cos(dogs_a, dogs_b) = {}  cos(dogs, physics) = {}", dogSim, unrelatedSim);
        assertTrue(dogSim > unrelatedSim,
                "dog-related sentences should be more similar than a dog sentence vs a physics sentence");
    }

    @Test
    @DisplayName("Metrics are collected across multiple embed calls")
    void testMetricsCollection() {
        var client = new OpenVinoMutinyStreamingBatchedClient(channel, PIPELINE, 8, TIMEOUT_MS);
        client.embed(List.of("one", "two", "three")).await().indefinitely();
        client.embed(List.of("four", "five")).await().indefinitely();

        assertTrue(client.getTotalRequests() >= 2);
        assertEquals(5, client.getTotalTextsProcessed());
        assertTrue(client.getAverageLatencyMs() > 0);
        assertTrue(client.getThroughputPerSec() > 0);
    }

    private static void assertNoNaN(float[] v) {
        for (float x : v) {
            assertFalse(Float.isNaN(x), "embedding contains NaN");
        }
    }

    private static void assertUnitNorm(float[] v) {
        double sum = 0;
        for (float x : v) sum += x * x;
        double norm = Math.sqrt(sum);
        assertTrue(Math.abs(norm - 1.0) < 0.01,
                "embedding should be L2-normalized (norm=" + norm + ")");
    }

    private static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0;
        for (int i = 0; i < a.length; i++) dot += a[i] * b[i];
        return dot;  // vectors are pre-normalized, so dot product == cosine similarity
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }
}
