import os
import argparse
import numpy as np
from PIL import Image

def read_description_file(desc_path):
    entries = []
    with open(desc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            fname, rest = line.split(':', 1)
            label = rest.strip()
            entries.append((fname.strip(), label))
    return entries

def load_image_vector(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    vec = (arr < 128).astype(np.float64)
    flat = vec.flatten()
    norm = np.linalg.norm(flat)
    if norm == 0:
        return flat, norm
    return flat / norm, norm

def main():
    parser = argparse.ArgumentParser(description='MADALINE OCR using normalized pattern vectors as neuron weights')
    parser.add_argument('train_directory', type=str)
    parser.add_argument('test_directory', type=str)
    args = parser.parse_args()


    train_list = read_description_file(os.path.join(args.train_directory, 'description.txt'))
    test_list = read_description_file(os.path.join(args.test_directory, 'description.txt'))


    weights = []
    for fname, label in train_list:
        path = os.path.join(args.train_directory, fname)
        if not os.path.exists(path):
            print('Warning: training image not found:', path)
            continue
        vec, norm = load_image_vector(path)
        if norm == 0:
            print('Warning: zero-vector for training image', path)
            continue
        weights.append((vec, label, fname))

    if not weights:
        print('No training data found.')
        return

    for fname, test_label in test_list:
        test_path = os.path.join(args.test_directory, fname)
        if not os.path.exists(test_path):
            print('Warning: test image not found:', test_path)
            continue
        test_vec, test_norm = load_image_vector(test_path)
        if test_norm == 0:
            print(f"{test_label} -> (empty test vector), confidence: 0.000")
            continue
        best_val = -1.0
        best_label = 'UNKNOWN'
        for wvec, wlabel, _ in weights:
            val = float(np.dot(test_vec, wvec))
            if val > best_val:
                best_val = val
                best_label = wlabel
        print(f"{test_label} -> {best_label}, confidence: {best_val:.3f}")

if __name__ == '__main__':
    main()