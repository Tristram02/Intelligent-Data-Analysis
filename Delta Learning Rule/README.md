# Intelligent Data Analysis: Delta Learning Rule

## Objective

The objective of this assignment is to implement and analyze the **delta learning rule** for a simple linear neuron. The task involves:

1. Implementing the delta learning algorithm
2. Conducting experiments with different configurations of parameters N (number of inputs) and M (number of training examples)
3. Analyzing algorithm convergence through visualization of error evolution
4. Investigating the impact of weight initialization on the learning process

## Solution

### Delta Learning Algorithm

The delta learning rule is a fundamental supervised learning algorithm for a linear neuron. The algorithm operates according to the following scheme:

1. **Initialization**: Weights `w` are randomly initialized from the interval `[-1, 1]`
2. **Iterative learning**: For each epoch `k` and each example `μ`:
   - Output calculation: `y = w · x^μ`
   - Error calculation: `e = z^μ - y`
   - Weight update: `w = w + η · e · x^μ`
3. **Monitoring**: Tracking the mean squared error (MSE) in each epoch

### Algorithm Parameters

- **η (eta)**: learning rate = 0.01
- **K**: number of epochs = 10000
- **N**: number of neuron inputs (input vector dimension)
- **M**: number of training examples
- **w_range**: weight initialization range = (-1, 1)

### Conducted Experiments

The code performs three experiments with different configurations:

1. **Case 1**: N=2, M=6 (more examples than dimensions)
2. **Case 2**: N=4, M=4 (equal number of examples and dimensions)
3. **Case 3**: N=6, M=3 (fewer examples than dimensions)

Each experiment is repeated 5 times with different weight initializations to investigate the impact of random initialization on the learning process.

### Results

For each experiment, the following are generated:

- **Error evolution plot**: shows how MSE changes across epochs for different initializations
- **Final weights**: weight values after training completion for each repetition
- **Final average error**: MSE calculated on the training set after learning completion

## Project Structure

```
Training and testing a linear neuron with multiple training patterns/
├── main.py
└── README.md
```

> **Note:** This repository contains only the source code (`main.py`). All other files are generated outputs from running the experiments.

## Running the Code

### Requirements

- Python 3.x
- NumPy
- Matplotlib

### Installing Dependencies

```bash
pip install numpy matplotlib
```

### Running Experiments

```bash
python main.py
```

The program automatically:

1. Conducts all three experiments
2. Generates plots in PNG format
3. Displays results in the console (final weights and MSE)

## Key Functions

### `delta_training(X, Z, eta, K, w_range)`

Implements the delta learning algorithm.

**Parameters:**

- `X`: matrix of training examples (M × N)
- `Z`: vector of expected outputs (M)
- `eta`: learning rate
- `K`: number of epochs
- `w_range`: weight initialization range

**Returns:**

- `w`: weight vector after training
- `errors`: list of MSE errors for each epoch

### `run_experiment(N, M, eta, K, repeats)`

Conducts a complete experiment with multiple repetitions.

**Parameters:**

- `N`: number of inputs
- `M`: number of training examples
- `eta`: learning rate
- `K`: number of epochs
- `repeats`: number of repetitions with different initializations

## Observations and Conclusions

1. **Convergence**: The algorithm demonstrates convergence for all investigated cases
2. **M/N Impact**: The ratio of the number of examples to the number of dimensions affects approximation quality
3. **Initialization**: Different initializations can lead to different local solutions
4. **Stability**: The algorithm is stable with an appropriately chosen learning rate

## Author

Kamil Włodarczyk (259413)

## License

Educational project - Intelligent Data Analysis
