# NumPy-Digit-Recognizer

A Multi-Layer Perceptron (MLP) neural network built entirely from scratch in Python using only the NumPy library. This project is designed to classify and recognize handwritten digits from the MNIST dataset.

## Features
- **Built from Scratch**: No external machine learning frameworks (like TensorFlow or PyTorch) are used. All mathematical operations are implemented purely with `NumPy`.
- **Backpropagation Algorithm**: Custom `forward pass` and `backward pass` implementation using matrix operations.
- **Activations & Loss**: Uses ReLU for hidden layers, Softmax for the output layer, and Cross-Entropy for calculating the loss.
- **Flexibility**: Easy configuration of hyperparameters (number and size of layers, learning rate, batch size, number of epochs) directly within `train.py`.

## Project Structure
- `train.py` — The main script that loads the data, initializes the network, and runs the training and testing loops.
- `neural_network.py` — The core implementation of the network architecture, layers, forward/backward math, gradients, and weights.
- `data_loader.py` — A utility explicitly for unpacking and reading the binary IDX files of the MNIST dataset.
- `mnistData/` — Directory intended to store the binary MNIST dataset files.

## Usage
1. Make sure `numpy` is installed:
   ```bash
   pip install numpy
   ```
2. Place the binary MNIST files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`) inside the `mnistData/` folder (create it if it doesn't exist).
3. Run the training process:
   ```bash
   python train.py
   ```
The console will display the training progress across epochs as well as the final accuracy of the network on the test dataset.
