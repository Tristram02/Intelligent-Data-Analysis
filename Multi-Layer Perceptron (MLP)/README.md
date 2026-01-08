# Intelligent Data Analysis: Multi-Layer Perceptron (MLP) with Backpropagation

## Objective

The objective of this assignment is to implement and analyze a **Multi-Layer Perceptron (MLP)** neural network trained with the **backpropagation algorithm**. The task involves:

1. Implementing a two-layer MLP with sigmoid activation functions
2. Training the network to learn an identity mapping (autoencoder)
3. Comparing network performance with and without bias neurons
4. Analyzing convergence behavior and training dynamics
5. Understanding the role of bias in neural network learning

## Solution

### MLP Architecture

The implementation uses a simple two-layer feedforward neural network:

**Network Structure:**

- **Input Layer**: 4 neurons (one-hot encoded patterns)
- **Hidden Layer**: 2 neurons (bottleneck layer)
- **Output Layer**: 4 neurons (reconstructed patterns)
- **Activation Function**: Sigmoid (logistic function)

**Two Configurations:**

1. **With Bias**: Hidden layer includes a bias neuron (3 inputs to output layer)
2. **Without Bias**: No bias neuron (2 inputs to output layer)

### Backpropagation Algorithm

The training process follows the standard backpropagation algorithm:

1. **Forward Pass**:

   - Compute hidden layer activations: `h = W1 · x`
   - Add bias if enabled: `h_with_bias = [h, 1]`
   - Compute output layer activations: `s = W2 · h_with_bias`
   - Apply sigmoid: `y = σ(s)`

2. **Backward Pass**:

   - Output layer error: `δ_out = σ'(s) · (z - y)`
   - Hidden layer error: `δ_hidden = W2^T · δ_out`
   - Weight updates:
     - `W2 = W2 + η · δ_out ⊗ h_with_bias`
     - `W1 = W1 + η · δ_hidden ⊗ x`

3. **Error Calculation**:
   - Sum of squared errors: `E = 0.5 · Σ(z - y)²`

### Algorithm Parameters

- **Learning Rate (η)**: 0.5
- **Maximum Epochs**: 10,000
- **Convergence Tolerance**: 1e-9
- **Weight Initialization**: Uniform random in [-0.5, 0.5]
- **Hidden Layer Size**: 2 neurons
- **Training Patterns**: 4 one-hot encoded vectors

### Training Task

The network learns an **identity mapping** (autoencoder task):

**Input Patterns (X):**

```
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

**Target Outputs (Z):** Same as input (Z = X)

This creates a dimensionality reduction challenge: the network must compress 4-dimensional patterns into a 2-dimensional hidden representation and then reconstruct them.

### Conducted Experiments

The code performs two main experiments:

1. **MLP with Bias**: Network includes bias neuron in hidden layer
2. **MLP without Bias**: Network excludes bias neuron

For each configuration, the system:

- Trains the network for up to 10,000 epochs
- Tracks error evolution across epochs
- Evaluates final reconstruction accuracy
- Compares convergence speed and final performance

### Results

For each experiment, the following are generated:

- **Training Error Plot**: `mlp_errors.png` showing error evolution (log scale) for both configurations
- **Final Outputs**: Console output showing reconstructed patterns for each input
- **CSV Export**: `mlp_final_outputs.csv` containing final network outputs for both configurations
- **Convergence Analysis**: Epoch count and final error values

## Project Structure

```
Multi-layer perceptron with backpropagation/
├── main.py
└── README.md
```

> **Note:** This repository contains only the source code (`main.py`). All other files are generated outputs from running the experiments.

## Running the Code

### Requirements

- Python 3.x
- NumPy
- Matplotlib
- Pandas

### Installing Dependencies

```bash
pip install numpy matplotlib pandas
```

### Running the Experiment

```bash
python main.py
```

The program automatically:

1. Trains both MLP configurations (with and without bias)
2. Displays training progress every 500 epochs
3. Generates error evolution plot (`mlp_errors.png`)
4. Exports final outputs to CSV (`mlp_final_outputs.csv`)
5. Prints final reconstruction results to console

## Key Functions

### `sigmoid(s)`

Computes the sigmoid activation function.

**Parameters:**

- `s`: input value or array

**Returns:**

- Sigmoid output: `1 / (1 + e^(-s))`

### `sigmoid_prime(s)`

Computes the derivative of the sigmoid function.

**Parameters:**

- `s`: input value or array

**Returns:**

- Sigmoid derivative: `σ(s) · (1 - σ(s))`

### Class: `SimpleMLP`

#### `__init__(eta, with_bias, w_init_range)`

Initializes the MLP with random weights.

**Parameters:**

- `eta`: learning rate
- `with_bias`: whether to include bias neuron in hidden layer
- `w_init_range`: tuple defining weight initialization range

#### `forward(x)`

Performs forward propagation through the network.

**Parameters:**

- `x`: input pattern vector

**Returns:**

- `h`: hidden layer activations (before bias)
- `h_with_bias`: hidden layer activations (with bias if enabled)
- `s`: output layer pre-activation values
- `y`: final output (after sigmoid)

#### `train_epoch(X, Z)`

Trains the network for one epoch using backpropagation.

**Parameters:**

- `X`: input patterns (matrix)
- `Z`: target outputs (matrix)

**Returns:**

- Total epoch error (sum of squared errors)

#### `train(X, Z, max_epochs, tol, verbose)`

Trains the network for multiple epochs until convergence or max epochs.

**Parameters:**

- `X`: input patterns
- `Z`: target outputs
- `max_epochs`: maximum number of training epochs
- `tol`: convergence tolerance (error threshold)
- `verbose`: whether to print training progress

**Returns:**

- List of errors for each epoch

## Observations and Conclusions

1. **Bias Importance**: The bias neuron significantly improves learning capability and convergence speed
2. **Convergence**: Both configurations can learn the identity mapping, but with different convergence rates
3. **Dimensionality Reduction**: The 2-neuron bottleneck successfully compresses 4D patterns into 2D representations
4. **Reconstruction Quality**: Networks with bias typically achieve better reconstruction accuracy
5. **Training Dynamics**: Error decreases exponentially (visible in log-scale plot), showing effective gradient descent

## Author

Kamil Włodarczyk (259413)

## License

Educational project - Intelligent Data Analysis
