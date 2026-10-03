import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest #REGISTRY is how prometheus tracks the metrics we created
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from config import block_size, device
from serve import load_model

#METRICS
import metrics #this line runs metrics.py which creates and sets up the latency histogram in the Prometheus registry
from drift import DriftMonitor, DriftCollector #DriftMonitor does the PSI computation and DriftCollector hands it to Prometheus

#LOGGING
logging.basicConfig(level=logging.INFO, format="%(message)s") #sets the basic logging configuration for the entire application
request_logger = logging.getLogger("llmis.requests") #creates a named log for request-related logging

#METRICS
drift_monitor = DriftMonitor()
REGISTRY.register(DriftCollector(drift_monitor)) #this registers the drift collector with the Prometheus registry so that drift metrics can be scraped by Prometheus

#batching initialization
from batching import BatchScheduler
model, encode, decode, stoi = load_model()
scheduler = BatchScheduler(model)

from quantize import quantize_model

#the following are kept separate from the main model for A/B comparison
model_int8 = quantize_model(model) #create a quantized INT8 version of the model 
scheduler_int8 = BatchScheduler(model_int8) #create a separate batch scheduler for the INT8 model

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scheduler.run_forever()) #starts the main batch scheduler in the background (asynctio.create_task is the worker thread that loops once the server starts; scheduler.run_forever() is the loop that collects requests, groups them into batches, and sends them off)
    task_int8 = asyncio.create_task(scheduler_int8.run_forever()) #starts the INT8 batch scheduler in the background (same pattern as the main scheduler)
    yield #yield indicates the server is still running
    task.cancel() #cancels the main batch scheduler task when the server shuts down
    task_int8.cancel() #cancels the INT8 batch scheduler task when the server shuts down

app = FastAPI(title="llmis", lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 500
    use_cache: bool = True #toggle for  kv cache
    use_batching: bool = True #toggle for batching
    use_quantization: bool = False #toggle for quantization

class GenerateResponse(BaseModel):
    completion: str

@app.get("/health")
def health():
    return {"status": "ok"}

#METRICS
@app.get("/metrics") #when /metrics is visited, run the following
def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST) #generate_latest() goes through the registry and renders all the metrics in the format expected by Prometheus; the response is then collected by Prometheus when it scrapes the /metrics endpoint

#non batched run
def _run_generate(model_to_use, idx, max_new_tokens, use_cache):
    with torch.no_grad():
        return model_to_use.generate(idx, max_new_tokens=max_new_tokens, use_cache=use_cache)

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    clean = "".join(c for c in req.prompt if c in stoi) #removes characters not in the vocabulary
    if req.prompt and not clean:
        raise HTTPException(status_code=400, detail="prompt has no in-vocabulary characters")

    #select the appropriate model, scheduler, and device based on the quantization flag
    active_model = model_int8 if req.use_quantization else model
    active_scheduler = scheduler_int8 if req.use_quantization else scheduler
    active_device = "cpu" if req.use_quantization else device

    if clean == "":
        idx = torch.zeros((1, 1), dtype=torch.long, device=active_device)
    else:
        idx = torch.tensor([encode(clean)], dtype=torch.long, device=active_device)

    uses_cache_bound = req.use_cache or req.use_batching
    if uses_cache_bound and idx.shape[1] + req.max_new_tokens > block_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"prompt ({idx.shape[1]} tokens) + max_new_tokens ({req.max_new_tokens}) "
                f"exceeds block_size ({block_size}) for the cached generation path. "
                f"reduce max_new_tokens, or set use_cache=false and use_batching=false."
            ),
        )

    #METRICS
    #perf_counter() is a pure stopwatch that only counts forward -> this is what we use to track latency
    #note: the location is important -> it comes after all of the validation steps so only valid accepted requests are timed 
    start = time.perf_counter()

    #this is where the request is either submitted to the batching scheduler or processed directly depending on the use_batching flag
    if req.use_batching:
        out_row = await active_scheduler.submit(idx[0], req.max_new_tokens) #out_row is the generated token ids for this request after it has been processed in a batch
        completion = decode(out_row.tolist()) #decoded out_row
    else:
        out = await run_in_threadpool(_run_generate, active_model, idx, req.max_new_tokens, req.use_cache)
        completion = decode(out[0].tolist())

    #METRICS
    latency_s = time.perf_counter() - start #calculate the latency in seconds for this request
    metrics.record(latency_s, req.use_cache, req.use_batching, req.use_quantization) #record the latency and request metadata in the metrics
    drift_monitor.observe(idx.shape[1]) #observe the prompt length for drift monitoring; idx.shape[1] gives the number of tokens in the prompt

    #LOGGING
    #creates the structured JSON log line for the named log we created earlier (request_logger)
    request_logger.info(json.dumps({
        "event": "generate",
        "prompt_len": idx.shape[1],
        "max_new_tokens": req.max_new_tokens,
        "use_cache": req.use_cache,
        "use_batching": req.use_batching,
        "use_quantization": req.use_quantization,
        "latency_ms": round(latency_s * 1000, 1),
        "prompt_length_psi": drift_monitor.psi(),
    }))

    return GenerateResponse(completion=completion)