# Ai-CDD



# Project File Overview

## dataset.py
Combines three separate datasets into a single unified dataset for training and evaluation.

## explore.py
Analyzes the datasets by:
- Checking dataset sizes
- Counting the number of images
- Providing basic dataset statistics

## xray_preprocessing.py
Performs **Stage 1 preprocessing**, including:
- Converting images to grayscale
- Resizing images to a standard size
- Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement

## xray_unet_preprocessing.py
Handles **Stage 2 preprocessing**, preparing the data specifically for U-Net model input.





```bash
📁 dataset
  └── test
      ├── NORMAL: 1163
      ├── TUBERCULOSIS: 555
      └── Total: 1718
  └── train
      ├── NORMAL: 4998
      ├── TUBERCULOSIS: 2359
      └── Total: 7357
  └── val
      ├── NORMAL: 935
      ├── TUBERCULOSIS: 523
      └── Total: 1458

📁 dataset1
  └── test
      ├── NORMAL: 525
      ├── TUBERCULOSIS: 106
      └── Total: 631
  └── train
      ├── NORMAL: 2450
      ├── TUBERCULOSIS: 489
      └── Total: 2939
  └── val
      ├── NORMAL: 525
      ├── TUBERCULOSIS: 105
      └── Total: 630

📁 dataset2
  └── test
      ├── NORMAL: 234
      ├── TURBERCULOSIS: 41
      └── Total: 275
  └── train
      ├── NORMAL: 1341
      ├── TURBERCULOSIS: 650
      └── Total: 1991
  └── val
      ├── NORMAL: 8
      ├── TURBERCULOSIS: 12
      └── Total: 20

📁 dataset3
  └── test
      ├── Normal: 404
      ├── Tuberculosis: 408
      └── Total: 812
  └── train
      ├── Normal: 1207
      ├── Tuberculosis: 1220
      └── Total: 2427
  └── val
      ├── Normal: 402
      ├── Tuberculosis: 406
      └── Total: 808
```