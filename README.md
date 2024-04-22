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


## Architectural differences between the vanilla Transformer and Mistral

## Sliding Window Attention

## Stabilizing training with router Z-loss (Chiao)


## Capacity Factor and Communication costs (Chiao)

