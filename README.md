# Perceptron

This model is a implemented version of perceptron (https://psycnet.apa.org/record/1959-09865-001). Basically, this code focuses on classify Iris dataset (https://archive.ics.uci.edu/ml/datasets/iris), but can be widely adjusted to other classifying task.

## Quick start
you can start with brief command below.
```
python train.py [--parameters]
```

Default parameters are already fitted to expected best performance, and following parameters can be controlled by various perpose.

## Hyperparameters
* hidden_size: setting the size of hidden state vector. (default 64)
* batch_size: In regard of mini-batch gradient descent, it set the size of mini-batch. (default 16)
* n_layers: The number of input layers. If greater than 2, the model will be MLP (default 4)
* epochs: The number of epochs. (default 100)
* optimizer: Only 3 optimzers are supported, Adam, SGD, RMSprop. Capital doesn't matter. (default: Adam)
* lr: Learning rate during gradient descent (default 2e-2)
* verbose: Print training logs every n epochs. For example, if this is set to 10, then print logs every 10 epochs. 