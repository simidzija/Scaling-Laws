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

The most elementary constraint when training LLMs (and when doing many other things) is money.
We have access to $300 in free Google Cloud credits, although I already blew ~$50 when I left the GPU running for a few days, leaving ~$250.

### Time

The most affordable GPU offered by Google Cloud is the T4, which (as I found out the hard way) costs ~$8/day.
Therefore with our budget we can run the T4 for about 30 days.

### Compute

The T4 has a theoretical peak performance of 65 TFLOPS (terafloating-point operations per second) when using float16 precision and 8.1 TFLOPS when using float32 precision.
Therefore we will use float16 precision wherever we can get away with it, but to avoid stability issues when training, certain quantities will have to be stored in float32 (see [Automatic Mixed Precision](#automatic-mixed-precision) section below).
Let us therefore conservatively estimate that we can achieve an average performance of 20 TFLOPS. 
With our estimate of 30 days of GPU access this gives a total compute budget of $5.4\times 10^{19}$ flops.

### Memory

To maximize GPU performance it is very important to minimize data transfer between the GPU and the CPU.
However this is constrained by the memory of the GPU, called the VRAM, which for the T4 is 16 GB.
There are various quantities that eat into this VRAM:

**Model parameters:** For a transformer model this is dominated by the parameters in the multi-head attention layers and the feed forward layers. 
With the standard GPT architecture, the former has four matrices each of size $d_\text{model} \times d_\text{model}$, while the latter has two matrices of size $4 \times d_\text{model} \times d_\text{model}$, for a total parameter count of $12 \times n_\text{layers} \times d_\text{model}^2$.
Each parameter will be stored in float32 (4 byte) precision.

**Data:**

**Activations:**

**Gradients:**

**Optimizer parameters:**





### Data


## Parameters

### Model size

### Batch size

### Optimizer


## Automatic Mixed Precision