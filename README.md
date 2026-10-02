# Convolutional neural network from scratch (NumPy)

A small CNN written without any deep learning framework: every layer has its own forward and backward pass in NumPy/SciPy, trained on MNIST digits.

## What's implemented

- `Convolutional`: valid cross-correlation forward pass, with kernel and input gradients computed by hand (`scipy.signal.correlate2d` / `convolve2d`)
- `Dense`, `Reshape`: fully connected and reshape layers
- `Sigmoid`, `Softmax`: activations with their derivatives
- Categorical cross-entropy loss and its gradient
- A plain SGD training loop on a subset of MNIST digit classes

Training error falls from about 0.095 to 0.027 over 20 epochs (`main.py` plots the curve).

## Running it

```bash
pip install numpy scipy matplotlib keras
cd Convolution_from_scratch/digit_recognizer
python cifar10.py   # trains on MNIST (keras is used only to download the data)
python main.py      # plots the training error curve
```
