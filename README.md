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


## Mistral and Mixtral
**Mistral**, or more formally, **Mistral-7B**, was first introduced in this [blogpost](https://mistral.ai/news/announcing-mistral-7b/) by Albert Jiang, et al. The model is open-source, and it is also the first large language model (LLM) released by the company, [mistral.ai](https://mistral.ai/). 

Mistral-7B is a transformer model designed for handling fast inference and longer sequences. It is a decoder-only Transformer with the following architectural choices:
* Sliding Window Attention
* Grouped Query Attention (GQA)
* Rolling Buffer Cache
* Pre-fill and Chunking

With these carefully designed architectures, Mistral-7B is able to handle tasks with longer sequences more effectively at a reduced cost. It takes a significant step in balancing the goals of achieving high performances while at the same time keep the large language model efficient. 

The company then takes one step further, introducing **Mixtral 8x7B**, which is a Sparse Mixture of Experts language model. It employs a mixture-of-experts architecture that dynamically selects exerp

Both Mistral and Mixtral models are open-source, available for download on HuggingFace. 


## Sliding Window Attention

## KV-Cache
The generative process of Large Language Models (LLMs) often employs a KV Cache mechanism to speed up output generation. This technique involves storing previously computed Key/Value vectors from the attention calculation and reusing them when generating new tokens, thus bypassing the need to recalculate for past tokens. However, while KV Cache is an effective strategy for efficiency, it significantly raises memory usage. This becomes more pronounced with larger models and longer text generations, leading to substantial demands on device memory resources [MODEL TELLS YOU WHAT TO DISCARD:
ADAPTIVE KV CACHE COMPRESSION FOR LLMS](https://arxiv.org/pdf/2310.01801.pdf).

## Model Sharding / Expert Parallelism

## Stabilizing training with router Z-loss (Chiao)


## Capacity Factor and Communication costs (Chiao)

