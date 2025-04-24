# Glacial Lake Segmentation: TensorFlow vs PyTorch

## Introduction

This project implements a semantic segmentation workflow for glacial lake imagery. It loads, preprocesses, and splits a custom dataset, then trains and evaluates lightweight U-Net-inspired models using both TensorFlow (Keras) and PyTorch. The script outputs quantitative comparison metrics and visualizes segmentation results, providing a baseline for future enhancements.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Features

- Data loading and preprocessing from a directory structure of images and masks.
- Data split into training and validation sets.
- TensorFlow (Keras) and PyTorch model definitions and training loops.
- Metrics: Precision, Recall, F1, IoU, training/inference time.
- Side-by-side validation loss curve comparison.
- Visualization of segmentation results from both frameworks.
- CSV export of training histories.

## Installation

1. Clone this repository and ensure you have Python 3.7+.
2. Install dependencies:

```bash
pip install numpy pandas torch torchvision tensorflow scikit-learn matplotlib pillow
```

## Usage

1. Place your dataset in the expected directory structure:

    /kaggle/input/glacial-lake-dataset/glacial-lake-dataset/
        ├── images/
        └── masks/

2. Update the dataset path in the script if your paths differ:

    IMAGE_PATH = '/path/to/images'
    MASK_PATH = '/path/to/masks'

3. Run the script:

    python main.py

4. Outputs:
    - Training logs and metrics in the console.
    - CSV files: `tensorflow_training_history.csv`, `pytorch_training_history.csv`
    - Comparison plots and segmentation visualization.

## Configuration

Modify the following constants in the script as needed:

    IMAGE_SIZE = (400, 400)     # Image and mask resize dimensions
    IMAGE_PATH = '...'          # Path to images
    MASK_PATH = '...'           # Path to masks

You may adjust epochs, batch size, and other hyperparameters directly in the script.

## Dependencies

- NumPy
- Pandas
- TensorFlow (Keras)
- PyTorch
- scikit-learn
- Matplotlib
- Pillow

## Project Structure

main.py                         # Main script for loading data, training, and evaluation  
tensorflow_training_history.csv # Saved TensorFlow training metrics  
pytorch_training_history.csv    # Saved PyTorch training metrics  

