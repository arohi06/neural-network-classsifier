# 🧠 Neural Network from Scratch – Image Recognition of Letters A, B, and C

## 📌 Project Overview
This project demonstrates a simple feedforward neural network using only NumPy to classify synthetic 5×6 binary images of the letters A, B, and C. No external machine learning libraries like TensorFlow or PyTorch were used.

We implemented:
- A two-layer neural network
- Sigmoid activation functions
- Manual backpropagation
- Training via gradient descent
- Loss and accuracy tracking over epochs
- Testing and visualization using matplotlib

## 🖼️ Input Data
Each character (A, B, and C) is represented using binary pixel patterns in a 5×6 grid format. These grids are flattened to 1D arrays with 30 elements:
- `A = [0,1,1,1,0,...]`
- `B = [1,1,1,1,0,...]`
- `C = [0,1,1,0,0,...]`

A few noisy variants of each letter were added to simulate real-world imperfections and improve generalization.

## 🔧 Methodology

### 1. Network Architecture
- **Input layer:** 30 neurons (each pixel)
- **Hidden layer:** 16 neurons, sigmoid activation
- **Output layer:** 3 neurons (one-hot encoding for A, B, C), sigmoid activation

### 2. Activation Function
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
