## Hyperparameters

Experiments on Google Colab T4 GPU:

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


### Learning rate

In the Chinchilla scaling experiments they use a learning rate of 2e-4 for their smallest model (73M parameters) and 1.25e-4 for their largest model (6.8B parameters).
My models will be on the smaller end of this range, so I will use the 2e-4 as a reference.
However the batch size used to train the Chinchilla models is 0.5M tokens, whereas I will use a much smaller batch size of 4096 tokens, to save GPU memory (see discussion above).
Therefore I will scale the learning rate by a factor of 4096/0.5M, giving a batch size of 1.6e-6.

This lr seems to give worse results than bigger lrs.
I think the reason is that Chinchilla used Maximal Update Parameterization while I'm using Standard Parameterization, and therefore using their lr to inform my lr doesn't make sense.

### Experiments with batch size

| params | n_layers | d_model | bs  | seq_len | tokens | VRAM (max 15360 MiB) | iter/s |
|--------|----------|---------|-----|---------|--------|----------------------|--------|
| 14M    | 8        | 256     | 16  | 256     | 4096   | 2515 MiB             | 7.0    |
| 14M    | 8        | 256     | 32  | 256     | 8192   | 5673 MiB             | 4.5    |
| 14M    | 8        | 256     | 16  | 512     | 8192   | 5107 MiB             | 3.9    |
| 14M    | 8        | 256     | 64  | 256     | 16384  | 9059 MiB             | 2.5    |
| 14M    | 8        | 256     | 32  | 512     | 16384  | 9969 MiB             | 2.1    |
| 14M    | 8        | 256     | 16  | 1024    | 16384  | 11571 MiB            | 1.6    |
| 14M    | 8        | 256     | 128 | 256     | 32768  | Out of memory        | N/A    |