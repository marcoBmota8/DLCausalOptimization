# DLCausalOptimization
CS 8395: Deep Learning - Final project - Spring 2024
Solutions from wide minima in the loss function have been shown to provide
deep learning models with better generalizability. These flatter regions of the loss
landscape are promoted by some optimizers such as stochastic gradient descent
(SGD) in which this behavior is controlled by the ratio of learning rate to batch size.
Recent studies have proposed variants of SGD that further promote wide minima
in the pursuit of more robust models that can withstand environment changes such
as covariate shift. For any data generating process, models that leverage causal
relationships are, by definition, the most robust to these kinds of data distribution
changes. However, there has been little assessment in the literature on whether (all
or some) of the wide local minima correspond to finding solutions that use causal
relationships for its predictions. In this study we explore such hypothesis for a
variety of optimizers which include SGD, entropy-SGD, (adaptive) sharpness aware
minimization (SAM), and vanilla gradient descend as naive baseline. We sample
synthetic data of different sizes and dimensionality from artificial directed acyclic
graphs for which we know the ground truth. As architecture, we use the previously
published TabNet which inherently provides with feature importance measure
through its sequential-attention mechanism. Our objective is to determine the
benefits of each optimizer in training deep learning model from a causal inference
perspective
