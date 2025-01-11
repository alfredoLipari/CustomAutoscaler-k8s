import os
import math
import time
from collections import deque
from fastapi import FastAPI, Request, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from threading import Lock
import asyncio
from pydantic import BaseModel
from contextlib import asynccontextmanager


app = FastAPI()

# Define all metrics at module level
REQUEST_COUNTER = Counter('example_requests_total', 'Total number of requests')
LATENCY_HISTOGRAM = Histogram('request_latency_seconds', 'Request latency in seconds', buckets=[0.1, 0.5, 0.7, 1])
REQUEST_RATE_GAUGE = Gauge('request_rate_per_second', 'Request rate per second over the last minute')
AVG_LATENCY_GAUGE = Gauge('average_latency_seconds', 'Average latency over the last minute')
PERCENTILE_LATENCY_GAUGE = Gauge('percentile_95_latency_seconds', '95th percentile latency over the last minute')
PREDICTED_RATE_GAUGE = Gauge('predicted_request_rate_per_minute', 'Predicted request rate per minute')

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'handler'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 1.0, 2.5, 5.0, 7.5, 10.0]
)

class MetricUpdate(BaseModel):
    value: float

class TimeWindowedMetrics:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.requests = deque(maxlen=1000)
        self.latencies = deque(maxlen=1000)
        self.lock = Lock()

    def add_request(self, timestamp):
        with self.lock:
            self.requests.append(timestamp)
            self._cleanup()

    def add_latency(self, latency):
        with self.lock:
            self.latencies.append((time.time(), latency))
            self._cleanup()

    def _cleanup(self):
        current_time = time.time()
        while self.requests and current_time - self.requests[0] > self.window_size:
            self.requests.popleft()
        while self.latencies and current_time - self.latencies[0][0] > self.window_size:
            self.latencies.popleft()

    def get_request_rate(self):
        with self.lock:
            if len(self.requests) < 2:
                return 0
            time_diff = self.requests[-1] - self.requests[0]
            if time_diff <= 0:
                return 0
            return (len(self.requests) - 1) / time_diff

    def get_average_latency(self):
        with self.lock:
            if not self.latencies:
                return 0
            return sum(l for _, l in self.latencies) / len(self.latencies)

    def get_95th_percentile_latency(self):
        with self.lock:
            if not self.latencies:
                return 0
            sorted_latencies = sorted(l for _, l in self.latencies)
            index = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]

windowed_metrics = TimeWindowedMetrics()

@app.middleware("http")
async def http_middleware(request: Request, call_next):
    if request.url.path in ["/metrics", "/health", "/update_metric"]:
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    HTTP_REQUEST_DURATION.labels(
        method=request.method,
        handler=request.url.path
    ).observe(duration)
    
    return response

def complex_computation(n: int):
    """
    More predictable CPU-intensive computation that scales linearly
    and avoids sudden spikes in CPU usage.
    """
    result = 0
    chunk_size = 1000
    num_chunks = n // chunk_size

    for chunk in range(num_chunks):
        # Process data in smaller chunks to maintain consistent CPU usage
        start_idx = chunk * chunk_size
        end_idx = start_idx + chunk_size
        
        # Mix of arithmetic and trigonometric operations for realistic load
        for i in range(start_idx, end_idx):
            # Use modulo to keep values in a reasonable range
            normalized_i = i % 360
            # Mix of different operations for more realistic CPU usage
            result += (normalized_i * 0.1) + math.sin(math.radians(normalized_i))
    
    return result / num_chunks

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await asyncio.sleep(5)

@app.get("/")
async def cpu_intensive_task(predicted_rate: float = None):
    REQUEST_COUNTER.inc()

    if predicted_rate is not None:
        PREDICTED_RATE_GAUGE.set(predicted_rate)
    
    start_time = time.time()
    # Reduced iterations for more predictable behavior
    result = await asyncio.get_event_loop().run_in_executor(
        None, complex_computation, 780000 
    )
    end_time = time.time()
    
    duration = end_time - start_time
    LATENCY_HISTOGRAM.observe(duration)
    
    windowed_metrics.add_request(start_time)
    windowed_metrics.add_latency(duration)
    
    return {
        "computation_result": f"{result:.2f}", 
        "time_taken_seconds": f"{duration:.2f}"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
def get_metrics():
    try:
        REQUEST_RATE_GAUGE.set(windowed_metrics.get_request_rate())
        AVG_LATENCY_GAUGE.set(windowed_metrics.get_average_latency())
        PERCENTILE_LATENCY_GAUGE.set(windowed_metrics.get_95th_percentile_latency())
        
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        print(f"Error generating metrics: {str(e)}")
        raise

@app.post("/update_metric")
async def update_metric(request: Request, update: MetricUpdate):
    body = await request.json()
    print(f"Received data: {body}")
    
    try:
        PREDICTED_RATE_GAUGE.set(update.value)
        return {"status": "success", "value": update.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    