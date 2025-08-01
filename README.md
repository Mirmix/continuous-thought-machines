> **Note:**  
> This repository is our attempt to test homeostatic (intrinsic) normalization in the Continuous Thought Machine (CTM) framework.  
> For a discussion of the normalization method, implementation details, and experimental results, **please refer to [README_NORMALIZATION.md](README_NORMALIZATION.md)**.  
>  
> The content below is the original CTM README for reference.

# 🕰️ The Continuous Thought Machine

📚 [PAPER: Technical Report](https://arxiv.org/abs/2505.05522) | 📝 [Blog](https://sakana.ai/ctm/) | 🕹️ [Interactive Website](https://pub.sakana.ai/ctm) | ✏️ [Tutorial](examples/01_mnist.ipynb)

![Activations](assets/activations.gif)

We present the Continuous Thought Machine (CTM), a model designed to unfold and then leverage neural activity as the underlying mechanism for observation and action. Our contributions are:

1. An internal temporal axis, decoupled from any input data, that enables neuron activity to unfold.

2. Neuron-level temporal processing, where each neuron uses unique weight parameters to process a history of incoming signals, enabling fine-grained temporal dynamics.

3. Neural synchronisation, employed as a direct latent representation for modulating data and producing outputs, thus directly encoding information in the timing of neural activity.

We demonstrate the CTM's strong performance and versatility across a range of challenging tasks, including ImageNet classification, solving 2D mazes, sorting, parity computation, question-answering, and RL tasks.

We provide all necessary code to reproduce our results and invite others to build upon and use CTMs in their own work.

## [Interactive Website](https://pub.sakana.ai/ctm)
Please see our [Interactive Website](https://pub.sakana.ai/ctm) for a maze-solving demo, many demonstrative videos of the method, results, and other findings. 

