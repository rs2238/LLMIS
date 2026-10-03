#METRICS (per-request latency histogram)
from prometheus_client import Histogram

from config import latency_buckets_seconds

#defines the request latency histogram
#native prometheus_client histogram for request latency -> Histogram() registers it with the default registry automatically -> its why we don't need REGISTRY.register(REQUEST_LATENCY) in app.py
REQUEST_LATENCY = Histogram(
    "llmis_request_latency_seconds",
    "Time to complete a /generate request, labeled by which serving levers were active.",
    ["cache", "batching", "quantization"],
    buckets=latency_buckets_seconds,
)

#function to record the request latency in the histograma
#.lables and .observe are both native Histogram methods
def record(latency_s, use_cache, use_batching, use_quantization):
    REQUEST_LATENCY.labels(
        cache=str(bool(use_cache)).lower(),
        batching=str(bool(use_batching)).lower(),
        quantization=str(bool(use_quantization)).lower(),
    ).observe(latency_s)
