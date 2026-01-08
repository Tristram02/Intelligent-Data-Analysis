# Intelligent Data Analysis: SOM-Based Image Compression

## Objective

The objective of this exam assignment is to implement and analyze a **Self-Organizing Map (SOM)** based image compression system using the **Winner-Takes-All (WTA)** learning algorithm. The task involves:

1. Implementing a SOM neural network for vector quantization
2. Compressing grayscale images by encoding them as patch codebooks
3. Analyzing compression performance across different parameters
4. Evaluating the trade-off between compression ratio (CR) and image quality (PSNR)
5. Understanding the relationship between network size, training parameters, and compression efficiency

## Solution

### SOM-Based Compression Algorithm

The implementation uses a Self-Organizing Map with Winner-Takes-All learning for lossy image compression through vector quantization:

**Compression Pipeline:**

1. **Preprocessing**:

   - Convert image to grayscale (if needed)
   - Resize to 128×128 pixels
   - Split into n×n patches (typically 2×2 or 4×4)

2. **Training Phase (Codebook Generation)**:

   - Normalize all patch vectors to unit length
   - Initialize K random neuron weight vectors
   - Train using WTA algorithm:
     - For each patch, find the neuron with highest dot product (winner)
     - Update winner weights: `W_winner = W_winner + η(x - W_winner)`
     - Normalize updated weights to unit length
   - Remove "dead" neurons (never activated)

3. **Encoding Phase**:

   - For each patch, find the best matching neuron (index L)
   - Store the patch magnitude quantized by factor q (value T)
   - Compressed representation: (Codebook W, Indices L, Magnitudes T)

4. **Decoding Phase**:
   - Reconstruct each patch: `patch = q × T × W[L]`
   - Reassemble patches into full image

### Algorithm Parameters

- **Image Size (N)**: 128×128 pixels
- **Patch Size (n)**: 2×2 or 4×4 pixels
- **Number of Neurons (K)**: [4, 8, 16, 32, 64, 128, 256]
- **Learning Rate (η)**: 0.2 (default)
- **Quantization Factor (q)**: 10 (default)
- **Training Epochs**: 20 (default)
- **Random Seed**: 42 for reproducibility

### Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio)**:

```
PSNR = 10 × log₁₀(255² / MSE)
```

Higher PSNR indicates better reconstruction quality (typically 20-40 dB).

**Compression Ratio (CR)**:

```
CR = Original_Bits / Compressed_Bits

where:
- Original_Bits = 8 × N × N
- Compressed_Bits = N_W + N_L + N_T
  - N_W = bits for codebook weights
  - N_L = bits for patch indices
  - N_T = bits for magnitude values
```

Higher CR indicates better compression.

### Conducted Experiments

The code performs systematic experiments varying:

1. **Number of Neurons (K)**: Tests K ∈ {4, 8, 16, 32, 64, 128, 256}
2. **Patch Size (n)**: Experiments with 2×2 and 4×4 patches
3. **Learning Rate (η)**: Tests η ∈ {0.05, 0.2, 0.5}
4. **Quantization Factor (q)**: Tests q ∈ {8, 10, 16}
5. **Training Epochs**: Tests epochs ∈ {20, 50, 100}

For each configuration, the system:

- Trains the SOM network
- Compresses and reconstructs the image
- Computes PSNR and CR metrics
- Generates CR vs PSNR plots
- Visualizes original and reconstructed images

### Test Images

The implementation is tested on standard grayscale images:

- `zelda.png` - Test image with various textures
- `cameraman.png` - Classic test image
- `peppers.png` - High-detail test image

### Results

For each experiment, the following are generated:

- **CR vs PSNR Plot**: Shows the trade-off between compression and quality
- **Reconstruction Comparison**: Side-by-side original vs reconstructed images
- **Console Output**: Detailed metrics for each K value (alive neurons, PSNR, CR)
- **Parameter Analysis**: Impact of η, q, epochs, and patch size on performance

## Project Structure

```
SOM-based image compression using WTA algorithm/
├── main.py
├── README.md
│
├── zelda.png
├── cameraman.png
├── peppers.png
│
└── plots/
    ├── plot_frame_4.png
    ├── image_frame_4.png
    └── [various experiment plots]
```

> **Note:** This repository contains only the source code (`main.py`) and test images. The `plots/` directory and result files are generated when running experiments.

## Running the Code

### Requirements

- Python 3.x
- NumPy
- scikit-image
- Matplotlib

### Installing Dependencies

```bash
pip install numpy scikit-image matplotlib
```

### Running the Default Experiment

```bash
python main.py
```

This will:

1. Load `zelda.png` and resize to 128×128
2. Train SOM with K ∈ {4, 8, 16, 32, 64, 128, 256}
3. Use 4×4 patches, η=0.2, q=10, epochs=20
4. Generate plots in `plots/` directory
5. Display metrics for each configuration

### Customizing Experiments

Edit the parameters in `main.py`:

```python
# Change test image
img_sk = io.imread("cameraman.png")

# Modify parameters
epochs = 50        # More training iterations
q = 16            # Different quantization factor
eta = 0.5         # Higher learning rate
frame_n = 2       # Use 2×2 patches instead of 4×4

# Test different K values
Ks = [8, 16, 32, 64, 128]
```

## Key Functions

### `split_into_patches(img, n)`

Divides an image into non-overlapping n×n patches.

**Parameters:**

- `img`: input image (N×N array)
- `n`: patch size

**Returns:**

- Array of flattened patches (num_patches × n²)

### `normalize_vectors(X)`

Normalizes vectors to unit length.

**Parameters:**

- `X`: matrix of vectors (num_vectors × dim)

**Returns:**

- `Xn`: normalized vectors
- `norms`: original vector norms

### `train_som_wta(X_normalized, num_neurons, epochs, eta, rng)`

Trains SOM using Winner-Takes-All algorithm.

**Parameters:**

- `X_normalized`: normalized training vectors
- `num_neurons`: number of neurons (K)
- `epochs`: training iterations
- `eta`: learning rate
- `rng`: random number generator

**Returns:**

- `W`: trained weight matrix (K × dim)
- `touch_count`: activation count for each neuron

### `encode_image(patches, W, norms_original, q)`

Encodes image patches using trained codebook.

**Parameters:**

- `patches`: image patches
- `W`: trained codebook weights
- `norms_original`: original patch norms
- `q`: quantization factor

**Returns:**

- `L`: winner indices for each patch
- `T`: quantized magnitude values

### `decode_image(L, T, W, n)`

Reconstructs image from compressed representation.

**Parameters:**

- `L`: winner indices
- `T`: magnitude values
- `W`: codebook weights
- `n`: patch size

**Returns:**

- Reconstructed image (N×N array)

### `compute_mse_psnr(img_orig, img_rec)`

Computes reconstruction quality metrics.

**Parameters:**

- `img_orig`: original image
- `img_rec`: reconstructed image

**Returns:**

- `mse`: Mean Squared Error
- `psnr`: Peak Signal-to-Noise Ratio (dB)

### `compute_cr(N, n, K, W, L, T)`

Computes compression ratio.

**Parameters:**

- `N`: image size
- `n`: patch size
- `K`: number of active neurons
- `W`, `L`, `T`: compressed representation

**Returns:**

- Compression ratio (CR)

## Observations and Conclusions

1. **Trade-off**: Clear trade-off between compression ratio and image quality (CR vs PSNR)
2. **Codebook Size**: Larger K improves quality but reduces compression ratio
3. **Patch Size**: Smaller patches (2×2) provide better quality but lower compression
4. **Dead Neurons**: Some neurons may never activate; removing them improves efficiency
5. **Learning Rate**: Higher η speeds convergence but may reduce final quality
6. **Quantization**: Lower q increases compression but introduces more quantization error
7. **Convergence**: 20-50 epochs typically sufficient for good codebook learning

## Author

Kamil Włodarczyk (259413)

## License

Educational project - Intelligent Data Analysis
