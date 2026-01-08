# Intelligent Data Analysis: MADALINE OCR - Character Recognition

## Objective

The objective of this assignment is to implement and analyze a **MADALINE-based Optical Character Recognition (OCR)** system for recognizing handwritten characters. The task involves:

1. Implementing a MADALINE neural network for character recognition
2. Generating training and test datasets with varying noise levels
3. Analyzing the network's robustness to noise
4. Evaluating recognition accuracy under different noise conditions
5. Understanding the relationship between pattern similarity and recognition confidence

## Solution

### MADALINE OCR Algorithm

The implementation uses a simplified MADALINE (Many Adaptive Linear Neurons) approach for character recognition:

1. **Training Phase**:

   - Load clean character images (noise level 0%)
   - Convert each image to a normalized binary vector
   - Store these normalized vectors as neuron weights
   - Each training pattern becomes a reference template

2. **Recognition Phase**:
   - Load test image and convert to normalized binary vector
   - Compute dot product (similarity) with all training patterns
   - Select the pattern with highest similarity as the recognized character
   - Output confidence score based on the similarity value

### Key Features

- **Normalization**: All pattern vectors are normalized to unit length for consistent comparison
- **Similarity Metric**: Uses dot product as a measure of pattern similarity
- **Noise Robustness**: Tests recognition performance across multiple noise levels (0%, 10%, 30%, 50%, 70%)
- **Binary Patterns**: Works with black-and-white character images (threshold at 128)

### Algorithm Parameters

- **Image Size**: 128×128 pixels
- **Noise Levels**: 0%, 10%, 30%, 50%, 70%
- **Font**: Times New Roman (times.ttf)
- **Character Set**: Configurable (typically uppercase letters)
- **Threshold**: 128 for binary conversion

### Data Generation Pipeline

The `font_generator.py` script creates character images:

1. Renders a character using specified font
2. Centers the character in a 128×128 canvas
3. Converts to binary (black/white) image
4. Adds specified percentage of random pixel noise
5. Saves image and appends metadata to `description.txt`

### Conducted Experiments

The system generates and tests characters across multiple noise levels:

1. **Training Set**: Clean characters (0% noise) in `training_patterns/`
2. **Test Sets**: Characters with varying noise levels:
   - `test_patterns_0/` - 0% noise (baseline)
   - `test_patterns_10/` - 10% noise
   - `test_patterns_30/` - 30% noise
   - `test_patterns_50/` - 50% noise
   - `test_patterns_70/` - 70% noise

### Results

For each test pattern, the system outputs:

- **Predicted Character**: The recognized letter
- **Confidence Score**: Similarity value (0.0 to 1.0)
- **Accuracy**: Comparison between expected and predicted labels

## Project Structure

```
MADALINE OCR for character recognition/
├── madeline_ocr.py
├── font_generator.py
├── prepare_data_and_run.sh
├── times.ttf
├── README.md
│
├── training_patterns/
│   ├── description.txt
│   └── [letter].png files
│
└── test_patterns_[0,10,30,50,70]/
    ├── description.txt
    └── [letter].png files
```

> **Note:** This repository contains only the source code. Pattern directories and generated images are created when running the scripts.

## Running the Code

### Requirements

- Python 3.x
- NumPy
- Pillow (PIL)
- Bash (for automation script)

### Installing Dependencies

```bash
pip install numpy pillow
```

### Quick Start - Automated Pipeline

Use the provided shell script to generate data and run experiments:

```bash
chmod +x prepare_data_and_run.sh
./prepare_data_and_run.sh times.ttf "A B C D E F G H I J"
```

This will:

1. Generate training patterns (0% noise)
2. Generate test patterns for all noise levels (0%, 10%, 30%, 50%, 70%)
3. Create `description.txt` files for each dataset

### Manual Usage

#### 1. Generate Training Data

```bash
python3 font_generator.py 128 128 28 10 times.ttf A 0 training_patterns
python3 font_generator.py 128 128 28 10 times.ttf B 0 training_patterns
# ... repeat for all letters
```

#### 2. Generate Test Data with Noise

```bash
python3 font_generator.py 128 128 28 10 times.ttf A 30 test_patterns_30
python3 font_generator.py 128 128 28 10 times.ttf B 30 test_patterns_30
# ... repeat for all letters and noise levels
```

#### 3. Run OCR Recognition

```bash
python3 madeline_ocr.py training_patterns test_patterns_0
python3 madeline_ocr.py training_patterns test_patterns_10
python3 madeline_ocr.py training_patterns test_patterns_30
python3 madeline_ocr.py training_patterns test_patterns_50
python3 madeline_ocr.py training_patterns test_patterns_70
```

## Key Functions

### `madeline_ocr.py`

#### `read_description_file(desc_path)`

Parses the description file to extract image filenames and labels.

**Parameters:**

- `desc_path`: path to description.txt file

**Returns:**

- List of tuples: `(filename, label)`

#### `load_image_vector(path)`

Loads an image and converts it to a normalized binary vector.

**Parameters:**

- `path`: path to image file

**Returns:**

- `vec`: normalized binary vector (flattened)
- `norm`: original vector norm (before normalization)

### `font_generator.py`

#### `generate_letter_image(w, h, x, y, font_path, letter)`

Generates a binary image of a single character.

**Parameters:**

- `w`, `h`: image dimensions
- `x`, `y`: character position offsets
- `font_path`: path to TrueType font file
- `letter`: character to render

**Returns:**

- Binary PIL Image object

#### `add_noise_binary(img, noise_percent, seed)`

Adds random pixel-flip noise to a binary image.

**Parameters:**

- `img`: input PIL Image
- `noise_percent`: percentage of pixels to flip (0-100)
- `seed`: random seed for reproducibility

**Returns:**

- Noisy PIL Image object

## Observations and Conclusions

1. **Noise Robustness**: The MADALINE approach shows good robustness to moderate noise levels (up to 30%)
2. **Confidence Degradation**: Recognition confidence decreases as noise level increases
3. **Pattern Similarity**: Normalized dot product effectively measures pattern similarity
4. **Simplicity vs. Performance**: Despite its simplicity, the algorithm performs well for clean and moderately noisy patterns
5. **Limitations**: Performance degrades significantly at high noise levels (50%+) where pattern structure is heavily distorted

## Author

Kamil Włodarczyk (259413)

## License

Educational project - Intelligent Data Analysis
