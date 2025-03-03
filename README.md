# Scaling Laws

## Constraints

There are many constraints when training LLMs.
The ones that will affect us are:
- Money
- Time
- Compute
- Memory
- Data
Let us determine how these constraints constrain us personally.

### Money

An elementary constraint when training LLMs is money.
We have access to ~$250 in Google Cloud credits.

### Time

The most affordable GPU offered by Google Cloud is the T4, which costs ~$8/day.
Therefore with our budget we can run the T4 for about 30 days.

### Compute

The T4 has a theoretical peak performance of 65 TFLOPS (terafloating-point operations per second) when using float16 precision and 8.1 TFLOPS when using float32 precision.
We'll use float16 precision wherever we can get away with it, but to avoid stability issues when training, certain quantities will have to be stored in float32 (see [Automatic Mixed Precision](#automatic-mixed-precision) section below).
Let's conservatively estimate that we can achieve an average performance of 20 TFLOPS. 
With our estimate of 30 days of GPU access this gives a total compute budget of $5.4\times 10^{19}$ flops.

### Memory

To maximize GPU performance it's important to minimize data transfer between the GPU and the CPU.
However this is constrained by the memory of the GPU, called the VRAM, which for the T4 is 16 GB.
There are various quantities that eat into this VRAM:

**Model parameters:** For a transformer model this is dominated by the parameters in the multi-head attention layers and the feed forward layers. 
With the standard GPT architecture, each attention layer has four matrices (WQ, WK, WV, WO) each of size $d_\text{model} \times d_\text{model}$, where $d_\text{model}$ is the size of the model's internal feature dimension, while each feed forward layerr has two matrices of size $4 \times d_\text{model} \times d_\text{model}$, for a total parameter count of $12 \times n_\text{layers} \times d_\text{model}^2$.
Each parameter is stored in float32 (4 byte) precision.

**Activations:** Which activations get stored depends somewhat on the specific implementation of the transformer architecture, and also on the details of how pytorch saves intermediate tensors during the forward pass.
We estimate the number of activations as:
- $4Bd_\text{model}$ from outputs of WQ, WK, WV, WO matrices in self attention layer, where $B$ is tokens in each batch
- $2S^2*B/S = 2BS$ activations from unnormalized and normalized (post-softmax) attention weights, where $S$ is sequence length
- $4Bd_\text{model}$ from output of first feed forward layer
- $4Bd_\text{model}$ from output of feed forward activation function
- $Bd_\text{model}$ from output of second feed forward layer
- $2Bd_\text{model}$ from pre-attention and pre-FFNN layer norms
- $2Bd_\text{model}$ from attention and FFNN residual connections
We will keep $S < d_\text{model}$ to avoid the quadratic attention cost. 
This gives a total activation count of roughly $17\times n_\text{layers} \times B \times d_\text{model}$.
Most activations are stored in float16 (2 byte) precision.

**Data:** In order to be able to store larger models on the GPU it is prudent to save VRAM by only using it to store one batch of training data at a time.
Therefore the data only takes up $B \times \text{(bytes per token)}$ amount of memory, which is negligible compared to the memory taken up by the activations.

**Gradients:** The gradients computed during backpropagation have to be stored in memory.
Since the number of gradients is the same as the number of model parameters, the amount of VRAM required for gradients is also $12 \times n_\text{layers} \times d_\text{model}^2$ times the precision of storing one gradient (4 bytes).

**Optimizer parameters:** We will use the AdamW optimizer, which is simply Adam followed by weight decay.
Therefore AdamW, like Adam, must store running averages of all gradients and squared gradients.
Thus the memory footprint of AdamW is twice that of the gradients, namely $2\times 12 \times n_\text{layers} \times d_\text{model}^2$ times the precision of storing one gradient (4 bytes).
Therefore the model weights, gradients, and AdamW optimizer require $192 \times n_\text{layers} \times d_\text{model}^2$ bytes of VRAM.

**Total:** Summing up the individual components, ignoring data since it is negligible, and using the formula $N \approx 12\times n_\text{layers}\times d_\text{model}^2$, we find:
$$\text{Total VRAM} = N\cdot\left(16 + \frac{3 B}{2d_\text{model}}\right) \text{   bytes}$$

**Comment on checkpointing:** Checkpointing is a technique in which certaint activations don't get saved during the forward pass, thus reducing memory usage.
However these activations then need to be recomputed during the backward pass.
Thus checkpointing decreases memory usage at the cost of requiring more compute.
Because our goal is to train the biggest model we can compute optimally, and since this model *is* able to fit on VRAM if we make the batch size small enough, the correct approach here is to *not* use checkpointing, and instead make the batch size small enough given the VRAM constraints. 
We will therefore not use activation checkpointing.


### Data

An emerging constraint for practitioners training increasingly larger LLMs is the amount of available high quality training data that is required to optimally train such large models.
Since our largest model will still be several orders of magnitude smaller than fronteir models, this won't be an issue for us.

## Hyperparameters

There are various hyperparameters that must be specified when defining and training LLMs.
Some of these hyperparameters are naturally fixed by our constraints, while others are chosen based on empirical heuristics.

### Model and data size

When multiplying a matrix with $N$ entries by a vector, one has to perform $N$ multiplications and $N$ additions. 
Since most of the operations in a transformer are matrix multiplications, a forward pass of a transformer requires $2N$ elementary operations (flops) per input token.
The number of flops per token for a backward pass is twice the amount for a forward pass.
This is because during the forward pass we mostly perform binary operations (addition and multiplication), and so during the backward pass gradients generally need to flow in *two* directions.
Thus the total compute for training on $D$ total tokens is roughly $6ND$.

Let $N_\text{max}$ denote the size of the largest model we plan to train.
We will also train models of sizes $0.1N_\text{max}$, $0.01N_\text{max}$, and $0.001N_\text{max}$.
We will train each model on varying amounts of data $D$ in order to determine the optimal balance between model size and data size.
Of course the answer is already known from the Chinchilla scaling laws, where it was shown that compute optimal training of a model with $N$ parameters requires $D\approx 20N$. 
We will therefore train with $D\in\{5N, 10N, 20N, 40N\}$ for each $N$ and compare our results to the expectation from Chinchilla.

Equating our compute budget of $5.4\times 10^{19}$ with the sum of $6ND$ for the various combinations of $N$ and $D$, we find $N_{max}\approx 344$M.
We will round $N_{max}$ to 300M and train models of size 0.3M, 3M, 30M, and 300M on data that is 5, 10, 20, and 40 times the size of the model. 


### Model and data shape



### Batch size

### Optimizer


## Automatic mixed precision


## Learning rate schedule