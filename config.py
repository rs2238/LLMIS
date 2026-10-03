import torch

#--------------------------------
#architecture hyperparameters
#--------------------------------
block_size = 256 #window length
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

#--------------------------------
#training hyperparameters
#--------------------------------
batch_size = 64 #number of windows
learning_rate = 3e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

#--------------------------------
#serving/batching hyperparameters (see batching.py)
#--------------------------------
max_batch_size = 8   #largest group of concurrent requests processed as one batched generate() call
max_batch_wait_ms = 50 #how long the scheduler waits for more requests to join a batch before running it

#--------------------------------
#observability hyperparameters (see metrics.py, drift.py)
#--------------------------------
latency_buckets_seconds = [0.5, 1, 2, 4, 8, 12, 16, 20, 30] #histogram bucket boundaries for /metrics,
                                                             #sized around the request latencies actually
                                                             #observed in benchmarks/ (0.1s-27s range)
drift_reference_size = 100 #requests used to establish the "normal" input distribution
drift_window_size = 100    #most recent requests compared against that reference
