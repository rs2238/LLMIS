#batching.py collects requests into batches for efficient model generation. app.py passes requests here and they are either processed when max_batch_size is reached or after max_batch_wait_ms has elapsed, whichever comes first.
import asyncio

import torch

from config import max_batch_size, max_batch_wait_ms

class BatchScheduler:
    #initializes the batch depot (where the requests are collected before being processed in batches)
    def __init__(self, model, max_batch_size=max_batch_size, max_wait_ms=max_batch_wait_ms):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue = asyncio.Queue()

    #this is what app.py calls to submit a request for batching -> this is where the requests are dropped off
    #future is just a placeholder for the result of this request's batch processing (the generated tokens)
    async def submit(self, idx, max_new_tokens):
        #idx: 1D LongTensor (a single prompt's token ids). called from the FastAPI request handler; blocks (async-ly) until this request's turn in a batch has been run.
        loop = asyncio.get_running_loop()
        fut = loop.create_future() #creates a future that will hold the result of this specific request's batch processing
        await self.queue.put((idx, max_new_tokens, fut)) #puts the request into the batch queue along with its future
        return await fut #wait until this request's batch has been processed (result is available) before return

    #depot worker thread that continuously forms and runs batches
    async def run_forever(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()] #waits until the first request shows up

            deadline = loop.time() + self.max_wait_ms / 1000 #once the first request shows up, the timer starts
            while len(batch) < self.max_batch_size: #while the length of the batch is less than the maximum allowed batch size
                remaining = deadline - loop.time() #compute remaining time before the deadline
                if remaining <= 0:
                    break #stop collecting when the remaining time is up
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining)) #wait for a new request to show up within the remaining time and add it to the batch if it arrives in time
                except asyncio.TimeoutError:
                    break #if no new request arrives within the remaining time, stop collecting

            #every request in batch consists of 3 things:
            #1. the token vector of the request (idx)
            #2. the maximum number of new tokens to generate (max_new_tokens)
            #3. a future to hold the result of this request's batch processing (fut)
            #the following lines bundle each component of the requests into separate lists for processing
            idx_list = [item[0] for item in batch]
            max_new_tokens_list = [item[1] for item in batch]
            futures = [item[2] for item in batch]

            #the actual generation is a long-running, CPU-bound, blocking call
            #this runs it in a separate worker thread so it doesn't freeze the event loop (which needs to keep accepting new requests into the next batch)
            try:
                results = await loop.run_in_executor(None, self._run_batch, idx_list, max_new_tokens_list) #run_in_executor runs the batch processing in a separate thread
                for fut, result in zip(futures, results): #zip(futures, results) pairs envelope 0 with answer 0, envelope 1 with answer 1, and so on
                    if not fut.done(): #only fill in the result if the future hasn't been completed yet
                        fut.set_result(result)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e) #puts an error into the future so the worker thread doesn't stall in case of failure

    #this is where the batches are run through the model (the actual generation happens here)
    def _run_batch(self, idx_list, max_new_tokens_list):
        with torch.no_grad():
            return self.model.generate_batched(idx_list, max_new_tokens_list)
