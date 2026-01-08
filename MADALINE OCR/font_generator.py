import os
import sys
import argparse
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_letter_image(w, h, x, y, font_path, letter):
    img = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, size=int(min(w,h)))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (w - text_w) // 2
    y = (h - text_h) // 4

    draw.text((x, -y), letter, font=font, fill=0)

    arr = np.array(img)
    bw = (arr < 128).astype(np.uint8) * 255
    return Image.fromarray(bw, mode='L')

def add_noise_binary(img, noise_percent, seed=None):
    if (noise_percent <= 0):
        return img

    if (noise_percent >= 100):
        w, h = img.size
        rand = np.random.RandomState(seed)
        arr = (rand.rand(h, w) > 0.5).astype(np.uint8) * 255
        return Image.fromarray(arr, mode='L')

    arr = np.array(img)
    h, w = arr.shape
    total = w * h
    flips = int(total * (noise_percent / 100.0))
    if (flips <= 0):
        return img
    
    rng = np.random.RandomState(seed)
    inds = rng.choice(total, size=flips, replace=False)
    flat = arr.flatten()
    flat[inds] = 255 - flat[inds]
    arr_noisy = flat.reshape((h, w))
    return Image.fromarray(arr_noisy, mode='L')

def main():
    parser = argparse.ArgumentParser(description='Generate single-letter binary png with noise and append description.txt')
    parser.add_argument('w', type=int)
    parser.add_argument('h', type=int)
    parser.add_argument('x', type=int)
    parser.add_argument('y', type=int)
    parser.add_argument('font_file', type=str)
    parser.add_argument('letter', type=str)
    parser.add_argument('noise_level', type=int)
    parser.add_argument('output_directory', type=str)
    args = parser.parse_args()


    os.makedirs(args.output_directory, exist_ok=True)
    img = generate_letter_image(args.w, args.h, args.x, args.y, args.font_file, args.letter)
    img_noisy = add_noise_binary(img, args.noise_level)


    filename = f"{args.letter}.png"
    outpath = os.path.join(args.output_directory, filename)
    img_noisy.save(outpath)


    descr_path = os.path.join(args.output_directory, 'description.txt')
    line = f"{filename}:letter {args.letter}, noise level {args.noise_level}%\n"
    with open(descr_path, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"Saved {outpath}")
    print(f"Appended description: {line.strip()}")

if __name__ == '__main__':
    main()