# Mixture of Experts (MoE)

Authors: Ding Zhang, Yuchen Wang, Chiao-Wei Hsu, and Yufeng Zou 

## MoE Introduction

Mixture of Experts (MoE) is a neural network architecture that combines multiple expert networks to solve complex problems. Each expert is trained to leverage specialized knowledge for different aspects of the task. A gating network determines which expert to activate based on the input data, ensuring that the most appropriate expert handles each input. MoE models have shown promise in various domains, including natural language processing, computer vision, and reinforcement learning. This blog post explores the notable pieces of work, architecture, and training methodologies of MoE models, highlighting their potential for improving model performance and efficiency.

<!-- ## Outline:
1. History: 
    1. Introduction: Experts as Components, Conditional Computation
    2. Models:
        1. GShard (yf)
        2. Switch (yf)
        3. Mistral
2. Architectural differences between the vanilla Transformer and Mistral (zd)
3. Sliding Window Attention (zd)
4. KV-Cache (yuchen)
5. Sparse Mixture of Experts (zd)
6. Model Sharding / Expert Parallelism (yuchen)
7. Stabilizing training with router Z-loss (Chiao)
8. Capacity Factor and Communication costs (Chiao)
 -->
## Outline

First, we will discuss the architecture of MoE models, with a focus on one notable model, Switch Transformers. Next, we will delve into Mixtral models by MistralAI, exploring their unique features and design choices. We will also cover key concepts such as sliding window attention, KV-cache, and model sharding, which are essential components of MoE models. Finally, we will discuss training methodologies for MoE models, including specialized loss functions, regularization techniques, and fine-tuning strategies. By examining these aspects of MoE models, we aim to provide a comprehensive overview of this innovative neural network architecture and its potential applications in machine learning.

## Switch Transformers

The [Switch Transformer](https://arxiv.org/abs/2101.03961) model, published by Google in 2022, uses a sparse [T5](https://en.wikipedia.org/wiki/T5_(language_model)) encoder-decoder architecture, where the MLP are replaced by a Mixture of Experts (MoE). A routing mechanism (top 1 in this case) associates each token to one of the expert, where each expert is a dense MLP. While switch transformers have a lot more weights than their equivalent dense models, the sparsity allows better scaling and better finetuning performance at scale. During a forward pass, only a fraction of the weights are used. The routing mechanism allows the model to select relevant weights on the fly which increases the model capacity without increasing the number of operations. The large version contains 1.6T parameters with 2048 experts, while the base version contains <7B parameters. Experiments show that Switch Transformers have significant speedup in pretraing over T5 counterparts and achieve better performance on downstream tasks.


There are several important aspects of this work:

**Switch routing**&ensp; Routes the input to the top-1 expert only, so an MoE layer is also called a Switch layer in this work. Suppose the gate value for expert $i$ is $p_i(x)$, then the output of the Switch layer is $y=p_j(x)E_j(x)$, where $j=\text{arg}\max_i p_i(x)$. This reduces the computation of experts and communication costs.

**Expert capacity**&ensp; Sets a limit on the number of inputs each expert can process, determined by $\text{expert capacity}=\frac{\text{# tokens per batch}}{\text{# experts}}\times\text{capacity factor}$. When an input is routed to an overloaded expert, the token representation is passed directly to the next layer through the residual connection. A capacity factor greater than 1 creates additional buffer to accommodate for when tokens are not perfectly balanced across experts. Increasing the capacity improves the quality but leads to more communication costs and memory of activations. Switch Transformers perform well at low capacity factors (1-1.25).

**Load balancing loss**&ensp; A differentiable load balancing loss is introduced to encourage a balanced load across experts. For each Switch layer, the auxiliary loss is added to the total model loss during training. This loss encourages uniform routing and can be weighted using a hyperparameter.

**Training and finetuning techniques** 
* Selective precision, such as training the experts with [*bfloat16*](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) while using full precision for the rest of the computations.
* Smaller parameter initialization for training stability.
* Regularization during finetuning by setting a smaller dropout rate (0.1) at non-expert layers and a much larger dropout rate (0.4) at expert layers.

**Parallel computation**&ensp;Data, model, and expert parallelism will be  explained in a future section of this blog.



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

## Sparsity in MoEs

**Sparsity?**

What do we mean "Sparsity" in MoE models like Mixtral 8x7B? Sparsity uses the idea of conditional computation, where in dense models all the parameters are used for the inputs, sparsity allows only specific parts of the network to be activated for each input. This set up allows one to scale up the size of the model without increasing the computation costs. 

How do we choose which experts to use? Well, in the Mixtral paper, a learned gating network $G$ decides which experts $E$ to process the input:
$$
y = \sum_{i=1}^{n} G(x)_i E_i(x), \text{ where } G_\sigma(x) := \text{Softmax}(\text{TopK}(x \cdot W_g))
$$
Here, $G(x)_i$ denotes the $n$ dimensional output of the gating network for the $i$-th expert, and the $E_i(x)$ is the output for the $i$-th expert. This is a weighted multiplication, and all experts are run for all the inputs. If $G$ equals to 0, then we don't need to compute the respective expert operations and save computation time. The gating function used in the Mixtral paper is a softmax, and it takes in the top-K logits of a linear layer:
$$
\text{TopK}(v, k)_i = 
\begin{cases} 
v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v, \\
-\infty & \text{otherwise}.
\end{cases}
$$

**Top-K experts in gating**

Having a lower $k$ number (e.g. $k=2$ in Mixtral 8x7B), this kind of routing ensures that the computational cost is controlled because only a small subset of the experts are activated for each input, which makes the model more efficient during both training and inference times compared to dense models where all parameters are always active. Why don't we just choose the top performing expert (i.e. $k=1$) instead of choosing two experts? It is likely chosen as a balance between model complexity and computational efficiency. The use of Top-2 routing in MoE allows for a sparse activation pattern where only the top two experts (in terms of gating network output) are utilized for a given input. This offers a trade-off that allows for more diversity and expert utilization than Top-1 routing, potentially increasing the representational capacity of the model without the full computational load that would come from activating all experts or a larger number of them. In practical applications, such as improving large language models (LLMs), Top-k routing with k=2 is used to merge domain-specific expert models that have been trained separately on specialized data sets into a single model that can utilize the specialized knowledge of each expert where applicable. This means that for any given input, the two most relevant experts are utilized, allowing the model to leverage specialized knowledge without overwhelming computational costs.


## Sliding Window Attention
**Problem with long input tokens**

Recall that the success of transformers is highly dependent on the self-attention mechanism. However, the nature of the Transformer architecture suffers from the maximum limitation of input size to 512 tokens. The input tokens are used as "keys" in the self-attention layers, which are the sequence representations, and "queries" that can attend to these keys, thus attends to itself. For example, let's assume a 5-token input sequence; for each token in the input sequence to be able to attend all keys (fully connected), this requires a quadratic $O(n^2)$ memory complexity per attention layer. This type of attention layer is known as the full attention or quadratic attention layer. A good way of thinking this is to represent the layer connectivity as an n\*n matrix. The memory requirment for this attention layer is the number of rows (n) times the number of columns (n), which is indeed $O(n^2)$. Thus, when the attention layer receives a large input sequence, the quadratic complexity makes it significantly inefficient for the transformer model computations. In some cases, the output may depend on long-distance attention between the document tokens (a word in the first chapter has been referenced in the fifth chapter, for example). Such long attention is not achievable in BERT-like models. 

**Sliding Window Attention**

Sliding window attention is an attention pattern for attention-based models. It is first being purposed in the [LongFormer's paper](https://arxiv.org/abs/2004.05150v2) as an attention mechanism. The mechanism tries to overcome the issue of limited input sequence length in aforementioned classical transformer models like BERT, by suggesting a convolution-like architecture for the attention mechanism. It defines a window of size $W$, such that the query node is allowed to attend only to $W$ of its neighbours inside the window. In the figure below, we show an attention window of size 3, where the highlighted node in green is allowed to attend to the peer key (middle node) and its immediate neighbours on the left and on the right. 


![Screenshot 2024-04-24 141713](https://hackmd.io/_uploads/H1yR9R8ZA.png)

The key assumption behind sliding window attention is that the most important information to the word is its local neighbours, with size $k$. This results in a memory complexity reduction to $O(nW)$, which is significantly efficient for $W << n$. 

However, one may be wondering that: didn't apply this sliding window attention losing information from key nodes outside the window size $W$? How would the sliding window problem solves the afore-mentioned problem when two words are far apart with each other in the chapters but still have unnegligble relationships? Well, if you look at the level of a single attention layer you may think so. But, when we stack multiple attention layers together, at higher layers, the query node gains attention information from far neighbors but in different representation way. The idea is very similar to the **receptive field** in CNN. In the level of a single attention layer, the key nodes sitting outside the window size of $W$ are discarded. But as we move on to the next layer, each node contains aggregated information of the nodes propagated from the previous layer. Thus, we end up with a conical structure for each token’s attention, starting with the local attentive nodes to their $W$ neighbors, but at higher layers, the attention gains information from tokens far away from it (global attention).
![1_lWBOpmmaXoPkqch3ToMA2g](https://hackmd.io/_uploads/ryWtZyDbR.gif)

Please refer to this [blog](https://ahelhady.medium.com/understanding-longformers-sliding-window-attention-mechanism-f5d61048a907) for more details.


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





## How were different MoEs trained?
The Mixture of Experts (MoE) model is a paradigm shift in machine learning, offering a divide-and-conquer approach to complex problems. By dividing the task among multiple specialized sub-networks, or "experts," an MoE model can process data more efficiently and effectively. This section delves into the training methodologies for different types of MoEs, providing insights into the intricacies of this innovative architecture.

### Understanding the MoE Architecture

As mentioned before, at its core, an MoE model consists of two primary components: the expert networks and the gating network. The expert networks are specialized neural networks trained on subsets of the data, each becoming adept at handling specific aspects of the overall task. The gating network, often another neural network, acts as a traffic controller, directing input data to the most appropriate expert based on learned parameters.

### Training Expert Networks

The training of expert networks follows a standard deep learning approach, typically involving backpropagation and gradient descent methods. Each expert is independently trained on designated data subsets, allowing it to develop a unique specialization. This process ensures that when the model encounters a specific type of input, it can leverage the expertise of the most qualified sub-network. A common training loss for such purpose is the cross-entropy loss, which measures the difference between the predicted and actual output probabilities.

The code snippet below illustrates the implementation of the cross-entropy loss function in PyTorch:

```python
import torch
import torch.nn as nn

# Define the cross-entropy loss function
loss_function = nn.CrossEntropyLoss()

# Generate sample data
inputs = torch.randn(3, 5, requires_grad=True)
targets = torch.empty(3, dtype=torch.long).random_(5)

# Calculate the loss
loss = loss_function(inputs, targets)
print(loss)
```

This code snippet demonstrates how to calculate the cross-entropy loss using PyTorch, a popular deep learning framework. By optimizing the expert networks through loss minimization, the MoE model can enhance its performance and adapt to a wide range of tasks. If we were to use numpy instead of PyTorch, the code would look like this:

```python
import numpy as np

# Define the cross-entropy loss function
def cross_entropy_loss(inputs, targets):  # inputs is the output probability of the expert network
    return -np.sum(targets * np.log(inputs))

# Generate sample data
inputs = np.random.rand(3, 5)
targets = np.random.randint(5, size=3)

# Calculate the loss
loss = cross_entropy_loss(inputs, targets)
print(loss)
```

### Gating Network Training

The gating network's training is crucial as it determines the efficiency of the MoE model. It learns to weigh the contributions of each expert, deciding which expert or combination of experts should be activated for a given input. This decision-making process is based on a probability distribution calculated by the gating function for each input, ensuring that the task is handled by the best-suited expert or experts.

### Balancing and Scaling MoEs

One of the challenges in training MoEs is balancing the load among experts. This involves ensuring that no single expert becomes a bottleneck, which could lead to inefficiencies. Additionally, scaling the number of experts impacts pretraining, as it requires careful consideration of the model's capacity and the computational resources available. The goal is to achieve a balance where the model scales effectively without compromising performance.

This target of load balancing presents itself as an auxiliary loss that is added to the loss introduced above. There are multiple lines of work implementing different variants. Essentially, most of them involve reducing the variance (e.g., by minimizing the coefficient of variance, CV) of the probability of the activation of each experts. Furthermore, specialized loss functions and regularization techniques play a crucial role in enhancing the training process of Mixture of Experts (MoE) models. Below introduce one notable technique - Z-loss function.

### Specialized Loss Functions and Regularization

To enhance the training process, specialized loss functions and regularization techniques are employed. For instance, the router Z-loss function (ST-MoEs) helps distribute the workload evenly among experts, preventing the "rich-get-richer" phenomenon. This loss function can be weighted to adjust its impact on expert utilization, ensuring a fair distribution of tasks. Regularization techniques, such as dropout or L2 regularization, are also used to prevent overfitting and improve the generalization of the MoE model.

<!-- include the z-loss.png from paper-->
![z-loss](https://hackmd.io/_uploads/rk27wCI-0.png)


Below is a simplified example of how the router Z-loss function can be implemented in PyTorch:

```python
import torch
import torch.nn as nn

# Define the router Z-loss function
def router_z_loss(logits):  # logits is the input to the gating network, ranging from -inf to inf
    return torch.mean(torch.log(torch.sum(torch.exp(logits), dim=1) ** 2))

# Generate sample data
logits = torch.randn(3, 5, requires_grad=True)  # assuming a batch of 3 samples and 5 experts

# Calculate the loss
loss = router_z_loss(logits)
print(loss)
```

, where [the original code snippet](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py) is from the paper [ST-MOE: DESIGNING STABLE AND TRANSFERABLE
SPARSE EXPERT MODELS](https://arxiv.org/pdf/2202.08906.pdf) and shown below for your reference:

```python
def _router_z_loss(logits, experts_dim, num_microbatches, importance=None):
  """Loss that encourages router logits to remain small and improves stability.

  Args:
    logits: a tensor with shape [<batch_dims>, experts_dim]
    experts_dim: a Dimension (the number of experts)
    num_microbatches: number of microbatches
    importance: an optional tensor with shape [<batch_dims>, group_size_dim]

  Returns:
    z_loss: scalar loss only applied by non-padded tokens and normalized by
      num_microbatches.
  """
  log_z = mtf.reduce_logsumexp(logits, experts_dim)
  z_loss = mtf.square(log_z)
  if importance is not None:
    z_loss *= mtf.cast(mtf.equal(importance, 1.0), dtype=z_loss.dtype)
  denom = mtf.reduce_sum(
      mtf.cast(mtf.equal(importance, 1.0), dtype=z_loss.dtype))
  z_loss = mtf.reduce_sum(z_loss) / (denom * num_microbatches)
  return z_loss
```

Note how the router Z-loss function has a form of L2 regularization, but different from the traditional L2 regularization applied to the weights of the model ([-inf, +inf]), it is applied to the probability output ([0, 1]) of the gating network. This is designed to encourage a more uniform distribution of tasks among experts, thereby improving the overall efficiency of the MoE model. By incorporating such specialized loss functions, MoEs can achieve better load balancing and performance optimization. 


### Fine-Tuning MoEs

Fine-tuning MoEs presents its own set of challenges. It involves adjusting the pretrained model to perform well on specific tasks or datasets. Recent advancements, such as MoE instruction-tuning, show promise in addressing these challenges, allowing MoEs to maintain their efficiency and effectiveness during the fine-tuning stage.

### Capacity Factor and Communication costs

The capacity factor is a critical parameter in MoE models that determines the maximum number of tokens that can be processed by an expert. By setting an appropriate capacity factor, the model can balance the computational load across experts, preventing bottlenecks and ensuring efficient resource utilization. This section delves into the concept of the capacity factor and its impact on communication costs in MoE models.

The capacity factor in MoE models represents the maximum number of tokens that an expert can process effectively. The definition of the capacity factor is shown below

![capacity](https://hackmd.io/_uploads/HJYNvRLb0.png)


A capacity factor greater than 1.0 creates additional buffer to accommodate for when to-
kens are not perfectly balanced across experts. If too many tokens are routed to an expert
(referred to later as dropped tokens), computation is skipped and the token representa-
tion is passed directly to the next layer through the residual connection. Increasing the
expert capacity is not without drawbacks, however, since high values will result in wasted
computation and memory.



## Future Work
The future of MoE models is promising, with ongoing research focusing on enhancing their performance and efficiency. Some key areas of interest include:

1. **Scalability**: Developing techniques to scale MoE models to handle even larger datasets and more complex tasks.
2. **Interpretability**: Improving the interpretability of MoE models to understand how decisions are made by the experts and the gating network.
3. **Quantization**: Exploring quantization techniques to reduce the memory and computational requirements of MoE models while maintaining performance.
4. **Transfer Learning**: Investigating transfer learning methods to leverage pretrained MoE models for a wide range of tasks and domains.
5. **Efficient Training**: Developing efficient training methodologies for MoE models to reduce training time and resource consumption.

By addressing these challenges and opportunities, researchers and practitioners can unlock the full potential of MoE models and leverage their capabilities to solve a wide range of machine learning tasks.

## Conclusions

Mixture of Experts (MoE) models represent a significant advancement in neural network architecture, offering a powerful approach to handling complex tasks. By combining multiple expert networks with a gating mechanism, MoE models can leverage specialized knowledge and improve performance. The architecture and training methodologies of MoE models have been refined over time, leading to notable advancements in natural language processing, computer vision, and other domains. As MoE models continue to evolve, they hold great promise for enhancing model efficiency, scalability, and interpretability, making them a valuable tool for machine learning practitioners.