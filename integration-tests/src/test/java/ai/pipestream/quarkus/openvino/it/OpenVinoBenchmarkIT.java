package ai.pipestream.quarkus.openvino.it;

import ai.pipestream.quarkus.openvino.runtime.OpenVinoStreamingBatchedClient;
import inference.GRPCInferenceServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Throughput / latency sweep against a live OVMS using the post-migration
 * (virtual-threads, no Mutiny) {@link OpenVinoStreamingBatchedClient}.
 *
 * <p>Skipped unless invoked with {@code -Dovms.benchmark=true}. Requires
 * {@code -Dovms.host} and {@code -Dovms.port} pointing at an OVMS instance
 * with {@code minilm_pipeline} and {@code mpnet_pipeline} loaded:
 * <pre>
 *   ./gradlew :quarkus-openvino-embeddings-integration-tests:test \
 *     --tests OpenVinoBenchmarkIT \
 *     -Dovms.benchmark=true \
 *     -Dovms.host=localhost -Dovms.port=9002 \
 *     -Dovms.benchmark.notes=b70-vt-post-psu \
 *     -Dovms.benchmark.output=/work/main/core-services/embedders/pipestream-embedder-openvino/docs/benchmarks/sweep-vt-b70.csv
 * </pre>
 *
 * <p><b>Schema match.</b> Output CSV uses the same 18-column schema as the
 * historical {@code docs/benchmarks/sweep-5model-b70.csv} — direct row-vs-row
 * comparison is the point. Provider label is
 * {@code ovms-streaming:<pipeline>}; label is {@code <pipeline>-streaming-batch32-<dev>}
 * to match the prior sweep's naming.
 *
 * <p>Corpus is synthetic — ~64-char realistic sentences repeated to fill
 * {@value #TOTAL_SENTENCES_DEFAULT}. The historical sweep used 32,555 court
 * opinion sentences; absolute numbers will differ slightly because of text
 * length variance, but relative comparison (pre-migration vs post-migration
 * on the same machine, same OVMS, same model) is meaningful.
 */
@DisplayName("OpenVINO Throughput Benchmark")
public class OpenVinoBenchmarkIT {

    private static final Logger log = LoggerFactory.getLogger(OpenVinoBenchmarkIT.class);

    private static final boolean ENABLED = Boolean.parseBoolean(
            System.getProperty("ovms.benchmark", "false"));
    private static final String HOST = System.getProperty("ovms.host", "localhost");
    private static final int PORT = Integer.parseInt(System.getProperty("ovms.port", "9001"));
    private static final int BATCH_SIZE = Integer.parseInt(
            System.getProperty("ovms.benchmark.batch", "32"));
    private static final int TOTAL_SENTENCES_DEFAULT = 32_555;
    private static final int TOTAL_SENTENCES = Integer.parseInt(
            System.getProperty("ovms.benchmark.sentences", String.valueOf(TOTAL_SENTENCES_DEFAULT)));
    private static final int WARMUP_BATCHES = Integer.parseInt(
            System.getProperty("ovms.benchmark.warmup", "3"));
    private static final String NOTES = System.getProperty("ovms.benchmark.notes", "vt-b70");
    private static final Path OUTPUT = Path.of(System.getProperty(
            "ovms.benchmark.output",
            "build/bench/sweep-vt.csv"));
    private static final int TIMEOUT_MS = 60_000;

    private static final List<String> SEED_SENTENCES = List.of(
            "The court considered whether the defendant's actions constituted reasonable conduct under the statute.",
            "OpenVINO Model Server provides hardware-accelerated inference on Intel GPUs and CPUs alike.",
            "Sentence-transformers produce dense vector representations suited for semantic search and clustering.",
            "The plaintiff's motion for summary judgment was denied based on disputed material facts.",
            "Virtual threads allow blocking I/O code to scale without explicit asynchronous composition.",
            "The expert witness testified about industry standards for safety equipment in commercial vehicles.",
            "KServe v2 binary inference protocol minimizes serialization overhead by carrying raw tensor bytes.",
            "The appellate court remanded the case for further findings consistent with this opinion.",
            "Pipeline DAG execution inside OVMS keeps tokenization and inference on the same process.",
            "The contract's force majeure clause did not extend to ordinary supply chain disruptions.");

    private ManagedChannel channel;
    private GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub;
    private List<String> corpus;

    @BeforeEach
    void setUp() {
        Assumptions.assumeTrue(ENABLED,
                "Benchmark IT skipped — pass -Dovms.benchmark=true to enable.");
        channel = ManagedChannelBuilder.forAddress(HOST, PORT).usePlaintext().build();
        stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);
        corpus = buildCorpus(TOTAL_SENTENCES);
        log.info("Benchmark setup — host={}:{} batch={} sentences={} warmup={} notes={} output={}",
                HOST, PORT, BATCH_SIZE, corpus.size(), WARMUP_BATCHES, NOTES, OUTPUT);
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (channel != null) {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    @DisplayName("MiniLM throughput sweep")
    void benchmarkMiniLM() throws Exception {
        runOne("minilm_pipeline", 384);
    }

    @Test
    @DisplayName("MPNet throughput sweep")
    void benchmarkMPNet() throws Exception {
        runOne("mpnet_pipeline", 768);
    }

    private void runOne(String pipeline, int expectedDims) throws Exception {
        var client = OpenVinoStreamingBatchedClient.create(stub, pipeline, BATCH_SIZE, TIMEOUT_MS);
        assertTrue(client.getDescriptor().dimensions() == expectedDims,
                pipeline + " expected dims=" + expectedDims);

        int n = corpus.size();
        int totalBatches = (n + BATCH_SIZE - 1) / BATCH_SIZE;
        int warmup = Math.min(WARMUP_BATCHES, totalBatches);

        // Warmup — run + discard timings.
        for (int i = 0; i < warmup; i++) {
            client.embed(corpus.subList(i * BATCH_SIZE, Math.min((i + 1) * BATCH_SIZE, n)));
        }

        long[] latNs = new long[totalBatches];
        long failures = 0;
        long measured = 0;
        long wallStart = System.nanoTime();
        for (int i = 0; i < totalBatches; i++) {
            int from = i * BATCH_SIZE;
            int to = Math.min(from + BATCH_SIZE, n);
            List<String> batch = corpus.subList(from, to);
            long t0 = System.nanoTime();
            try {
                client.embed(batch);
                latNs[i] = System.nanoTime() - t0;
                measured += batch.size();
            } catch (Exception e) {
                latNs[i] = System.nanoTime() - t0;
                failures++;
                log.warn("Batch {} failed for {}: {}", i, pipeline, e.toString());
            }
        }
        long wallMs = (System.nanoTime() - wallStart) / 1_000_000L;

        long[] sortedMs = toSortedMs(latNs);
        double p50 = pct(sortedMs, 50), p95 = pct(sortedMs, 95), p99 = pct(sortedMs, 99);
        double min = sortedMs.length == 0 ? 0 : sortedMs[0] / 1_000_000.0;
        double max = sortedMs.length == 0 ? 0 : sortedMs[sortedMs.length - 1] / 1_000_000.0;
        double mean = meanMs(latNs);
        double sentPerSec = wallMs == 0 ? 0 : (measured * 1000.0) / wallMs;
        double batchPerSec = wallMs == 0 ? 0 : (totalBatches * 1000.0) / wallMs;

        String label = pipeline + "-streaming-batch" + BATCH_SIZE + "-" + (PORT == 9002 ? "gpu" : "cpu");
        String provider = "ovms-streaming:" + pipeline;
        appendCsvRow(OUTPUT, Instant.now(), provider, label, expectedDims, BATCH_SIZE,
                measured, totalBatches, wallMs, p50, p95, p99, min, max, mean,
                sentPerSec, batchPerSec, failures, NOTES);

        log.info("== {} == {} sentences / {} batches / {} ms — sent/s={} batch/s={} p50={}ms p95={}ms p99={}ms failures={}",
                label, measured, totalBatches, wallMs,
                String.format("%.1f", sentPerSec), String.format("%.2f", batchPerSec),
                String.format("%.2f", p50), String.format("%.2f", p95), String.format("%.2f", p99),
                failures);
    }

    private static List<String> buildCorpus(int target) {
        List<String> out = new ArrayList<>(target);
        for (int i = 0; out.size() < target; i++) {
            out.add(SEED_SENTENCES.get(i % SEED_SENTENCES.size()));
        }
        return out;
    }

    private static synchronized void appendCsvRow(Path path, Instant runAt, String provider, String label,
                                                  int dims, int batchSize, long sentences, long batches,
                                                  long wallMs, double p50, double p95, double p99,
                                                  double min, double max, double mean,
                                                  double sentPerSec, double batchPerSec,
                                                  long failures, String notes) throws Exception {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        boolean exists = Files.exists(path);
        StringBuilder sb = new StringBuilder();
        if (!exists) {
            sb.append("run_at,provider,label,dimensions,batch_size,total_sentences,total_batches,wall_ms,")
              .append("p50_ms,p95_ms,p99_ms,min_ms,max_ms,mean_ms,sentences_per_sec,batches_per_sec,failures,notes\n");
        }
        sb.append(runAt).append(',').append(provider).append(',').append(label).append(',')
          .append(dims).append(',').append(batchSize).append(',').append(sentences).append(',')
          .append(batches).append(',').append(wallMs).append(',')
          .append(fmt(p50)).append(',').append(fmt(p95)).append(',').append(fmt(p99)).append(',')
          .append(fmt(min)).append(',').append(fmt(max)).append(',').append(fmt(mean)).append(',')
          .append(fmt(sentPerSec)).append(',').append(fmt(batchPerSec)).append(',')
          .append(failures).append(',').append(notes).append('\n');
        Files.writeString(path, sb.toString(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static String fmt(double v) {
        return String.format("%.3f", v);
    }

    private static long[] toSortedMs(long[] ns) {
        long[] copy = Arrays.copyOf(ns, ns.length);
        Arrays.sort(copy);
        return copy;
    }

    private static double pct(long[] sortedNs, double p) {
        if (sortedNs.length == 0) return 0;
        int idx = (int) Math.ceil((p / 100.0) * sortedNs.length) - 1;
        idx = Math.max(0, Math.min(sortedNs.length - 1, idx));
        return sortedNs[idx] / 1_000_000.0;
    }

    private static double meanMs(long[] ns) {
        if (ns.length == 0) return 0;
        long sum = 0;
        for (long v : ns) sum += v;
        return (sum / (double) ns.length) / 1_000_000.0;
    }
}
