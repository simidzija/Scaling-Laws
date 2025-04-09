Experiments on T4 GPU:

| params | n_layers | d_model | batch_size | seq_len | can run? | iter/s |
|--------|----------|---------|------------|---------|----------|--------|
| 262M   | 20       | 1024    | 32         | 128     | YES      | 1.7    |
| 485M   | 24       | 1280    | 32         | 128     | YES      | 1.0    |
| 585M   | 24       | 1408    | 32         | 128     | NO       | N/A    |

So the largest model we can run is roughly 485M params. 
Let's check if this makes sense.
This model is 1.94 GB (4 bytes / parameter). 
During LLM training the main contributions to GPU usage are:
- model: 1.94 GB
- gradients: same as model (1.94 GB)
- Adam optimizer: 2x model size (3.88 GB) 
- Attention activations: n_layers * batch_size * seq_len^2 numbers at 2 byte precision (0.03 GB)
- Other activations: O(10) * n_layers * n_tokens * d_model numbers at 2 byte precision (~1.3 GB)

This adds up to ~9 GB. 
This is reasonable given that the T4 has 16 GB of VRAM, 
that some of it has to go towards overheads costs,
and that we probably aren't pushing it to it's absolute limit.
Note that although the attention activation memory footprint grows quadratically with the sequence length
and can therefore dominate, in our case the sequence length is fairly small (128),
so this part of the memory footprint is minor.

Chinchilla taught us that the number of tokens for compute optimal training should be
roughly 20x the model size.
So for the 485M model we should train on ~10B tokens.
Each batch has 32 * 128 = 4096 tokens so this amounts to ~2.5M batches.
Each batch takes ~1s so this would take about a month of runtime, which is reasonable: 
the T4 costs ~$7/day and Google gives $300 in credits, which amounts to ~40 days of continuous runtime.

The fact that we are saturating both our GPU memory and making full use of our GPU hours is a good sign
that we have made good decisions for our hyperparameters.
However there is one degree of freedom that we can still tune, without much affect on the memory or runtime:
we can change batch_size and seq_len while keeping their product (n_tokens) fixed.
This will affect the memory footprint of the attention activations, and so it is generally not a luxery one has,
but because our attention activations are relatively small we can get away with it (to an extent).
The smart thing to do therefore is to increase seq_len at the expense of batch_size, 
because this will, essentially for free, result in a model with a larger context window.
Indeed we find that decreasing the batch_size to 16 while increasing seq_len to 256 still allows the training
to fit on the GPU and it has no affect on the runtime (still 1.0 s/batch).
Going to batch_size = 8 and seq_len = 512 however is too much; the training run crashes due to insufficient memory,
presumably because the minor increase in attention activation memory pushes us past the memory capacity of the GPU.
