# micrograd

Building Andrej Karpathy's micrograd from scratch

A tiny Autograd engine. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification.

## summary

A neural network takes in inputs, does a forward pass to make predictions, then does backpropagation to update the weights of the neural network so that it makes even more accurate predictions (known as gradient descent). This is done iteratively until the neural network is good enough to make accurate predictions.

This is done in many different ways, but all of them make use of these fundamentals detailed in Micrograd. The only difference is how they are implemented, and how they are used to train neural networks. Micrograd is a great way to understand the fundamentals of neural networks and how they are trained.

## references

[karpathy's project](https://github.com/karpathy/micrograd) <br>
[karpathy's tutorial video](https://www.youtube.com/watch?v=VMj-3S1tku0)
