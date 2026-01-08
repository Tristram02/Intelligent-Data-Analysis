# Intelligent Data Analysis - Part 1: Neural Network Fundamentals

This repository contains implementations of fundamental neural network algorithms and learning techniques, developed as part of the Intelligent Data Analysis course. The assignments progress from basic single-neuron learning to advanced multi-layer networks and self-organizing systems.

## Overview

Part 1 focuses on classical neural network architectures and learning algorithms, covering:

- **Supervised Learning**: Delta rule, backpropagation
- **Pattern Recognition**: MADALINE-based OCR
- **Unsupervised Learning**: Self-Organizing Maps (SOM)
- **Vector Quantization**: Image compression techniques

Each assignment is self-contained with its own implementation, documentation, and experimental results.

## Assignments

### 1. Delta Learning Rule

**Directory**: `Delta Learning Rule/`

**Objective**: Implement and analyze the delta learning rule for training a simple linear neuron.

**Key Concepts**:

- Single-layer perceptron learning
- Gradient descent optimization
- Weight initialization impact
- Convergence analysis

**Problem**: Train a linear neuron to learn an identity mapping with varying ratios of training examples (M) to input dimensions (N).

**Experiments**:

- N=2, M=6 (overdetermined system)
- N=4, M=4 (square system)
- N=6, M=3 (underdetermined system)

**Learning Outcomes**:

- Understanding basic supervised learning
- Analyzing convergence behavior
- Impact of dataset size on learning quality

---

### 2. MADALINE OCR

**Directory**: `MADALINE OCR/`

**Objective**: Implement a MADALINE (Many Adaptive Linear Neurons) network for optical character recognition with noise robustness analysis.

**Key Concepts**:

- Pattern matching using normalized vectors
- Similarity metrics (dot product)
- Noise robustness in pattern recognition
- Vector normalization techniques

**Problem**: Recognize handwritten characters from noisy binary images using a template-matching approach.

**Experiments**:

- Training on clean characters (0% noise)
- Testing with varying noise levels: 0%, 10%, 30%, 50%, 70%
- Analyzing confidence scores and recognition accuracy

**Learning Outcomes**:

- Template-based pattern recognition
- Understanding noise impact on recognition
- Similarity-based classification

---

### 3. Multi-Layer Perceptron (MLP)

**Directory**: `Multi-Layer Perceptron (MLP)/`

**Objective**: Implement a two-layer MLP with backpropagation algorithm and analyze the role of bias neurons.

**Key Concepts**:

- Multi-layer feedforward networks
- Backpropagation algorithm
- Sigmoid activation functions
- Bias neurons and their importance
- Autoencoder architecture

**Problem**: Train a 4-2-4 autoencoder to learn identity mapping through a bottleneck layer (dimensionality reduction).

**Experiments**:

- MLP with bias neuron in hidden layer
- MLP without bias neuron
- Comparison of convergence speed and reconstruction quality

**Learning Outcomes**:

- Understanding backpropagation mechanics
- Role of bias in neural networks
- Dimensionality reduction and reconstruction
- Non-linear feature learning

---

### 4. SOM-Based Image Compression

**Directory**: `SOM-Based Image Compression/`

**Objective**: Implement a Self-Organizing Map using Winner-Takes-All algorithm for lossy image compression through vector quantization.

**Key Concepts**:

- Self-Organizing Maps (SOM)
- Winner-Takes-All (WTA) learning
- Vector quantization
- Compression ratio vs quality trade-off
- Codebook learning

**Problem**: Compress grayscale images by learning a codebook of representative patches and encoding images as indices + magnitudes.

**Experiments**:

- Varying codebook size (K = 4, 8, 16, 32, 64, 128, 256)
- Different patch sizes (2×2, 4×4)
- Parameter tuning (learning rate, quantization factor, epochs)
- Quality metrics (PSNR, MSE, Compression Ratio)

**Learning Outcomes**:

- Unsupervised learning with SOM
- Vector quantization principles
- Compression-quality trade-offs
- Dead neuron phenomenon

---

## Common Themes

### Progressive Complexity

1. **Assignment 1**: Single neuron, linear learning
2. **Assignment 2**: Multiple neurons, pattern matching
3. **Assignment 3**: Multi-layer network, non-linear learning
4. **Assignment 4**: Self-organizing network, unsupervised learning

### Learning Paradigms

- **Supervised Learning**: Assignments 1, 3 (with labeled data)
- **Template Matching**: Assignment 2 (similarity-based)
- **Unsupervised Learning**: Assignment 4 (no labels, self-organization)

### Key Algorithms

- **Delta Rule**: Gradient descent for single-layer networks
- **Backpropagation**: Error propagation in multi-layer networks
- **Winner-Takes-All**: Competitive learning in SOM

## Running the Assignments

Each assignment is independent and can be run separately. General requirements:

### Common Dependencies

```bash
pip install numpy matplotlib pandas scikit-image pillow
```

### Execution

Navigate to each assignment directory and run:

```bash
cd "Delta Learning Rule"
python main.py

cd "../MADALINE OCR"
./prepare_data_and_run.sh times.ttf "A B C D E F"
python madeline_ocr.py training_patterns test_patterns_30

cd "../Multi-Layer Perceptron (MLP)"
python main.py

cd "../SOM-Based Image Compression"
python main.py
```

Refer to individual README files for detailed instructions and parameter customization.

## Key Takeaways

### Theoretical Insights

1. **Learning Rate**: Critical parameter affecting convergence speed and stability
2. **Network Capacity**: More neurons/layers increase representational power but require more data
3. **Bias Neurons**: Essential for learning offset transformations
4. **Normalization**: Improves training stability and comparison metrics
5. **Trade-offs**: Compression vs quality, speed vs accuracy, complexity vs interpretability

### Practical Skills

- Implementing neural networks from scratch (no high-level frameworks)
- Analyzing convergence and performance metrics
- Visualizing training dynamics and results
- Parameter tuning and experimental design
- Understanding algorithmic trade-offs

## Technologies Used

- **Python 3.x**: Primary programming language
- **NumPy**: Numerical computations and linear algebra
- **Matplotlib**: Visualization and plotting
- **scikit-image**: Image processing (Assignment 4)
- **Pillow (PIL)**: Image generation and manipulation (Assignment 2)
- **Pandas**: Data export and analysis (Assignment 3)

## Author

Kamil Włodarczyk (259413)

## Course Information

**Course**: Intelligent Data Analysis  
**Institution**: [University Name]  
**Academic Year**: 2024/2025  
**Part**: 1 - Neural Network Fundamentals

## License

Educational project - All implementations are for learning purposes as part of the Intelligent Data Analysis course.
