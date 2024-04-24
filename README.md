# Mixture of Experts (MoE)

Authors: Ding Zhang, Yuchen Wang, Chiao-Wei Hsu, and Yufeng Zou 

## 

## MoE Introduction

## Outline:
1. History: 
    1. Introduction: Experts as Components, Conditional Computation
    3. Models:
        1. GShard (yf)
        2. Switch (yf)
        3. Mistral

Mistral:
2. Architectural differences between the vanilla Transformer and Mistral (zd)
3. Sliding Window Attention (zd)
4. KV-Cache (yuchen)
5. Sparse Mixture of Experts (zd)
6. Model Sharding / Expert Parallelism (yuchen)
7. Stabilizing training with router Z-loss (Chiao)
8. Capacity Factor and Communication costs (Chiao)

9. Understanding the Mistral model's code

## Switch Transformers

The [Switch Transformer](https://arxiv.org/abs/2101.03961) model, published by Google in 2022, uses a sparse [T5](https://en.wikipedia.org/wiki/T5_(language_model)) encoder-decoder architecture, where the MLP are replaced by a Mixture of Experts (MoE). A routing mechanism (top 1 in this case) associates each token to one of the expert, where each expert is a dense MLP. While switch transformers have a lot more weights than their equivalent dense models, the sparsity allows better scaling and better finetuning performance at scale. During a forward pass, only a fraction of the weights are used. The routing mechanism allows the model to select relevant weights on the fly which increases the model capacity without increasing the number of operations. The largest version contains 1.6T parameters with 2048 experts, while the base version contains <7B parameters.

There are several important aspects of this work:

**Switch routing**&ensp; Routes the input to the top-1 expert only, so an MoE layer is also called a Switch layer in this work. Suppose the gate value for expert $i$ is $p_i(x)$, then the output of the Switch layer is $y=p_j(x)E_j(x)$, where $j=\text{arg}\max_i p_i(x)$. This reduces the computation of experts and communication costs.

**Expert capacity**&ensp; Sets a limit on the number of inputs each expert can process, determined by $\text{expert capacity}=\frac{\text{# tokens per batch}}{\text{# experts}}\times\text{capacity factor}$. When an input is routed to an overloaded expert, the token representation is passed directly to the next layer through the residual connection. A capacity factor greater than 1 creates additional buffer to accommodate for when tokens are not perfectly balanced across experts. Increasing the capacity improves the quality but leads to more communication costs and memory of activations. Switch Transformers perform well at low capacity factors (1-1.25).

**Load balancing loss**&ensp; A differentiable load balancing loss is introduced to encourage a balanced load across experts. For each Switch layer, the auxiliary loss is added to the total model loss during training. This loss encourages uniform routing and can be weighted using a hyperparameter.

**Training and finetuning techniques** 
* Selective precision, such as training the experts with [*bfloat16*](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) while using full precision for the rest of the computations.
* Smaller parameter initialization for training stability.
* Regularization during finetuning by setting a smaller dropout rate (0.1) at non-expert layers and a much larger dropout rate (0.4) at expert layers.

**Parallel computation**&ensp;Data, model, and expert parallelism will be  explained in a future section of this blog.

performance ...

## Mistral and Mixtral
**Mistral**, or more formally, **Mistral-7B**, was first introduced in this [blogpost](https://mistral.ai/news/announcing-mistral-7b/) by Albert Jiang, et al. The model is open-source, and it is also the first large language model (LLM) released by the company, [mistral.ai](https://mistral.ai/). 

Mistral-7B is a transformer model designed for handling fast inference and longer sequences. It is a decoder-only Transformer with the following architectural choices:
* Sliding Window Attention
* Grouped Query Attention (GQA)
* Rolling Buffer Cache
* Pre-fill and Chunking

With these carefully designed architectures, Mistral-7B is able to handle tasks with longer sequences more effectively at a reduced cost. It takes a significant step in balancing the goals of achieving high performances while at the same time keep the large language model efficient. 

The company then takes one step further, introducing **Mixtral 8x7B**, which is a Sparse Mixture of Experts language model. It employs a mixture-of-experts architecture that dynamically selects a certain number of experts for processing each token based on a gating mechanism, which also allows it to handle a large number of parameters efficiently during inference. Specifically, while the model has a total of 46.7 billion parameters, it only uses around 13 billion active parameters per token, which enhances both speed and efficiency compared to other large models like GPT-3.5 and Llama 2 (side note, according to the name of the model, technically there should be a total number of 8\*7=56 billion parameters; the reason is MoE is not simply just an ensemble of 8 models with 7B parameters, rather, only some layers of the model are replicated). 

In the next few sections of this blog, we will provide a detailed explanation on the components of the MoE architecture, and the reasons behind these designs. Both Mistral and Mixtral models are open-source, available for download on HuggingFace. 


## Sliding Window Attention
### Problem with long input tokens
Recall that the success of transformers is highly dependent on the self-attention mechanism. However, the nature of the Transformer architecture suffers from the maximum limitation of input size to 512 tokens. The input tokens are used as "keys" in the self-attention layers, which are the sequence representations, and "queries" that can attend to these keys, thus attends to itself. For example, let's assume a 5-token input sequence; for each token in the input sequence to be able to attend all keys (fully connected), this requires a quadratic $O(n^2)$ memory complexity per attention layer. This type of attention layer is known as the full attention or quadratic attention layer. A good way of thinking this is to represent the layer connectivity as an n\*n matrix. The memory requirment for this attention layer is the number of rows (n) times the number of columns (n), which is indeed $O(n^2)$. Thus, when the attention layer receives a large input sequence, the quadratic complexity makes it significantly inefficient for the transformer model computations. In some cases, the output may depend on long-distance attention between the document tokens (a word in the first chapter has been referenced in the fifth chapter, for example). Such long attention is not achievable in BERT-like models. 

### Sliding Window Attention
Sliding window attention is an attention pattern for attention-based models. It is first being purposed in the [LongFormer's paper](https://arxiv.org/abs/2004.05150v2) as an attention mechanism. The mechanism tries to overcome the issue of limited input sequence length in aforementioned classical transformer models like BERT, by suggesting a convolution-like architecture for the attention mechanism. It defines a window of width $W$, such that the query node 

Please refer to this [blog](https://ahelhady.medium.com/understanding-longformers-sliding-window-attention-mechanism-f5d61048a907) for more details.

## Grouped Query Attention

## KV-Cache
In this section, we are going to explain what is a KV-Cache. Here, K stands for key value and V stands for V value. So KV cache is a key-value chaching systems. data is stored in the from of key-value pairs where each key is unique and maps to a pecific value. When a key-value pair is cached, the key acts as an identifier that is used to quickly retrieve the corresponding value from the cache, without needing to compute the value again or retrieve it from a slower data storage. 


![KV cache](https://hackmd.io/_uploads/BJyji6H-0.gif)

So a KV Cache mechanism is often emoployed by the generative process of Large Language Models (LLMs) often employs to speed up output generation. This technique involves storing previously computed Key/Value vectors from the attention calculation and reusing them when generating new tokens, thus bypassing the need to recalculate for past tokens. 

We could verify how the KV cache speed up the process through the ***computations***. The calculations are provided by [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/#kv-cache). 

But before we start our computations, we need to first familiar oueselves with some basic terminology that used in subsequent discussions:

* ***Floating point operations per second*** ([flops](https://en.wikipedia.org/wiki/FLOPS)): A
flop serves as a basic unit of computation, which could denote one addition, subtraction, multiplication or
division of floating point numbers. Note that, the flop count is just a rough measure of how expensive an
algorithm can be. 
* ***flop bound***: It would then mean that there is time when nothing is being passed through memory.
* ***memory bound:*** It would mean that no floperations are occuring. Loading weights could consume memory bandwidth.

Per token, the numebr of bytes we store is 

$$
2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}}
$$
, where 2 is to account for the two vectors, k and v. We store KV paris per layer, and each of thoes values is a n_heads * d_heads matrix. 

The flops to compute ka dn v for all our layer is 
$$
2 \cdot 2 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2
$$

It takes $2 \cdot d_{\text{model}}^2$ to multiply each token embedding by each token weight. We have another factor of 2 as we do that twice, once each for k and v and then repeat for $n_{\text{layer}}$.

This means for a ***52B*** parameters model(taking Antrropic's, where $d_{\text{model}}$ = ***8192*** and $n_{\text{layer}}$ = ***64***). The flops are 
$$
2 \cdot 2 \cdot 64 \cdot 8192^2 = 17,179,869,184
$$

Say we have a NVIDIA A100 GPU, whcih does ***312e12*** flops per second and ***1.5e12*** bytes per second of memory badwidth. The following are numbers for just the kv weights and computations.
$$
memory = \frac{2 \cdot 2 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2}{1.5e12 }
$$

$$
compute = \frac{2 \cdot 2 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2}{312e12 }
$$

So here we get a ratio of 208 (result of $\frac{312e12}{1.5e12}$) given this hardware specification. This means if we're going to compute kv for one token, it'll take the same amount of time to compute for up to 208 tokens! For fewer than 208 tokens, the system is memory bandwidth-bound, implying that we cannot fully leverage the computational operations. Beyond 208 tokens, we are computation-bound, meaning that memory is not fully utilized. 

The intersection of the below diagram is at 208, though in reality the memory line does have a slight slope due to memory cost of intermediate calculations.

![roofline](https://hackmd.io/_uploads/BkuoYRS-0.png)

Assume that the context length is 6, then for a 52B model full forwards pass, that's $\frac{12 \cdot 2 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2}{1.5e12} \approx 69$ milliseconds for up to 208 tokens. If we had 416 (double) tokens in the context, then it would take twice as long, and 312 tokens would take 1.5 times as long.

Calculating for a kv cache token is exactly 1/6 of the compute of passing the token through the model. In general, these forwards passes (what we experience in getting logits, embeddings and training) are very cheap because of the parallelism that is possible as opposed to sampling where we're forced to read through all the weights for each token and do the autoregression.

This doesn't mean that only 1/6 of the time is saved! Let's assume we are flops bound. Then at each sample step, we save $\frac{12 \cdot 2 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2}{312e12}$ flops while the decoding steps costs $\frac{12 \cdot 2 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2}{312e12}$. Thus at each step we save 1/6 of the slops time multiplied by the number of tokens in our sequence (big!) — which increases as we sample tokens. Without a kv cache, sampling would be quadratic in time complexity as we increase the number of tokens.

However, while KV Cache is an effective strategy for efficiency, it significantly raises memory usage. This becomes more pronounced with larger models and longer text generations, leading to substantial demands on device memory resources [MODEL TELLS YOU WHAT TO DISCARD:
ADAPTIVE KV CACHE COMPRESSION FOR LLMS](https://arxiv.org/pdf/2310.01801.pdf).

## Model Sharding / Expert Parallelism
Model sharding is a technique used in deep learning to handle large models that cannot fit entirely into the memory of a single computing device, such as a GPU or CPU. This situation often arises with large-scale deep learning models, such as those found in natural language processing (e.g., GPT-3) and computer vision. Sharding involves dividing the model's parameters across multiple devices, allowing parallel processing and memory distribution.

The model's parameters are split into distinct subsets, called shards. Each shard contains a portion of the model's total parameters. Each computing device or node in a distributed system handles onr or more shards. This seyup means that each part of the model is processed in parallel across different devices, effectively utilizing the combined memory and computational power of the systm. While model sharding allows for the training of very large models by leveraging multiple devices, it also introduces the need for these devices to communicate with each other. This communication typically involves synchronizing the gradients or parameters during the training process, which can be a significant overhead.

And then we need the model parallelism to parallel the data we divide. For more information, you can refer to [How to Parallelize Deep Learning on GPUs Part 2/2: Model Parallelism](https://timdettmers.com/2014/11/09/model-parallelism-deep-learning/).

Model parallelism is, when you split the model among GPUs and use the same data for each model; so each GPU works on a part of the model rather than a part of the data. In deep learning, one approach is to do this by splitting the weights, e.g. a 1000×1000 weight matrix would be split into a 1000×250 matrix if you use ***four*** GPUs.

However, model parallelism is not the best way to do the data paralelism. This is discussed in [Model Parallelism](https://huggingface.co/transformers/v4.10.1/parallelism.html). The problem is there is one GPU is idle at any given moment. So if 4 GPUs are used, it's almost identical to quadrupling the amount of memory of a single GPU, and ignoring the rest of the harware. Plus there is the overhead of copying the data between devices. So 4x 6GB cards will be able to accommodate the same size as 1x 24GB card using naive MP, except the latter will complete the training faster, since it doesn’t have the data copying overhead. But, say, if you have 40GB cards and need to fit a 45GB model you can with 4x 40GB cards (but barely because of the gradient and optimizer states)

This photo is from [Introducing GPipe, an Open Source Library for Efficiently Training Large-scale Neural Network Models](https://research.google/blog/introducing-gpipe-an-open-source-library-for-efficiently-training-large-scale-neural-network-models/), on the top is model parallelism(MP) and the bottom is pipeline parallel(PP). 

![image2](https://hackmd.io/_uploads/B1aOdx8Z0.png)

We could see that PP has less zones where GPUs are idel. The idle parts are referred to as the "bubble".

Here GPU0 performs the same forward path on chunk 0,1,2, and 3(F0,0, F0,1, F0,2, F0,3) and then it waits for other GPUs to do their work and only when theri work is starting to be completed, GPU0 starts to work again doing the backward path for chunks 3, 2, 1(B0,3, B0,2, B0,1, B0,0).

Because of the chunks, PP introduces the concept of micro-batches (MBS). DP splits the global data batch size into mini-batches, so if you have a data prallel degree of 4 (which means we have 4 GPUs), a global batch size of 1024 gets split up into 4 mini-batches of 256 each (1024/4). And if the number of chunks is 32 we end up with a micro-batch size of 8 (256/32). Each Pipeline stage works with a single micro-batch at a time.

With chunk = 1, you get the naive MP. With a very large chunk number, you end up with tiny micro-batch sizes which could be not very efficient either. So we need to finetune the chunk number to achieve the highest efficient utilization of the GPUs.





## Stabilizing training with router Z-loss (Chiao)


## Capacity Factor and Communication costs (Chiao)

