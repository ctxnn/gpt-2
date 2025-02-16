# gpt2 from scratch 

## overview

this repository contains code and experiments for training gpt-2 from scratch on the fineweb edu dataset. the goal is to explore the performance of a transformer-based language model when trained on educational web data. various optimizations, techniques, and insights related to transformer training are documented here.

---

## training details

so i finished the training on 5000 epochs on 1 gpu for around ~10$, and loss was 3.370812, validation loss was 3.3529 and hellaswag accuracy was 2785/10042=0.2773 which is of course greater than random(25) but if i ran it for more epochs it would’ve been perfect. for now, it’s done.

---

### about the initialization of the gpt2 model
- the std deviation of the linear layer should be 0.02 (it is 0.02 cause if it is calculated by 1/sqrt(d_model) and the average value of the d_model of the gpt2 series is 0.02)
- we want to initialize the bias with zeros
- we scale the weights of residual path by the factor of 1/sqrt(N), where N is the no of layers

---

### more about the residual path/stream problem
- in transformer architectures, each layer adds its output to the residual stream (output = input + layer_output), which causes the variance of activations to accumulate and grow linearly with network depth.
- without scaling, after n layers, the variance becomes n*V (where V is the initial variance), leading to potential activation explosions and training instability.
- to solve this, gpt-2 scales down each layer's output by 1/√n, where n = 2*num_layers (doubled because each transformer layer has two residual connections: one after attention and one after mlp).
- this scaling factor ensures that after n layers, the total variance remains constant (V) instead of growing to nV, as the scaled variance becomes n(V/n) = V.
- the result is more stable training, balanced layer contributions, and predictable behavior regardless of model depth, which is crucial for deep transformers that can be hundreds of layers.

---

### intuition

you're adding dice rolls together:
- layer 1: you roll a die (random output) → some random number
- layer 2: you add another die roll → more randomness added
- layer 3: add another die roll → even more randomness
- ...and so on

just like adding more dice rolls makes your total number more unpredictable and likely to get larger, each transformer layer adds its own "random" contribution to the residual stream, making the signal increasingly noisy and potentially too large.

---

## using mixed precision

when using bfloat16 with mixed precision pytorch will use bfloat16 for some things (like logits) and then it will use torch.float32 for other things (like wte, wpe etc).

---

## using torch.compile

increases the performance (reduces computation time).

- torch.compile sees all the code at the same time (unlike the python interpreter which does not know which operation will come next) and optimizes it (there is no python interpreter involved).
- it makes the going back and forth with memory less (optimizes the round trips) and we will not pay the memory bandwidth cost anymore.
- kernel fusion - allows you to keep the data on the chip and it makes the back and forth less.

for more details → [Google Doc](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0)

---

## flash attention

so what flash attention does is that the $N \times N$ attention matrix never gets read or written to the hbm (it stays on the chip).

---

### sometimes the flops don't matter; knowing about memory hierarchy matters

---

## one silly optimization

so we want the `vocab_size` to be in the power of 2, cause many cuda kernels work in sort of power of 2.

---

## about weight decay

weight decay is a regularization technique used in machine learning to prevent overfitting by discouraging excessively large weights in a model. it works by adding a penalty term to the loss function that depends on the magnitude of the model’s weights.

### mathematical formulation

weight decay is typically implemented using L2 regularization, where the modified loss function becomes:

$$L = L_{\text{original}} + \lambda \sum_{i} w_i^2$$

- $L_{\text{original}}$ is the original loss (e.g., cross-entropy, MSE).
- $w_i$ are the model’s parameters (weights).
- $\lambda$ (weight decay factor) controls the strength of regularization.

this penalty encourages smaller weights, reducing model complexity and improving generalization.

#### pytorch implementation example

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## gradient accumulation and batch size

gpt-3 used a batch size of 0.5M, but even with 4-8 A100s, that would be impossible. so we use gradient accumulation.

### why use gradient accumulation?
- **overcome memory constraints**: instead of using a large batch size (which may not fit in memory), gradients from multiple smaller batches are accumulated and applied at once.
- **improve stability**: larger effective batch sizes result in more stable updates and better generalization.
- **match performance of large-batch training**: allows training with small batches while mimicking the behavior of larger batches.

---

## about using hellaswag eval

this evaluation is somewhat outdated but still useful. openai also used this while evaluating gpt-3.

- it has a random score of 25% (meaning any model will get around 25% if it is just guessing). if our model is good, we should be able to gradually increase past this baseline.

---