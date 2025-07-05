# Computer Vision Assignment

This repository contains the implementation of the individual assignment for the course "ΜΥΕ046 – Computer Vision" (Winter Semester 2024–2025) at the University of Ioannina (UOI).

The assignment covers both classical and deep learning methods for image classification using the MNIST dataset. All code is implemented in Python and organized in a Jupyter Notebook.

## Repository Contents


## Assignment Overview

The assignment is divided into **two major parts**:

### Part 1: Machine Learning (15 points)

- Use of Scikit-learn to build classifiers
- Comparison of:
  - Random classifier
  - Logistic Regression
  - k-Nearest Neighbors (sklearn & custom implementation)
  - PCA + k-NN with manual SVD
- Confusion matrix analysis and accuracy evaluation

### Part 2: Deep Learning (15 points + 5 bonus)

- Manual training loop with PyTorch
- Classifiers implemented:
  - Linear Classifier (Single-Layer Perceptron)
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
- Weight visualization and filter interpretation
- Evaluation via confusion matrices

## Technologies Used

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- PyTorch
- Jupyter Notebook / Google Colab

## Setup Instructions

> This project was designed to run on either **Google Colab** or locally using **Anaconda** or a virtual environment.

### Option 1: Google Colab (Recommended)

1. Upload the notebook and related files to your Google Drive
2. Open `assignment.ipynb` with Colab
3. Install any missing dependencies using:
   ```python
   !pip install torch torchvision matplotlib numpy scipy scikit-learn

### Option 2: Anaconda Environment (Local)
conda create -n mye046 python=3.7
conda activate mye046
conda install jupyter numpy matplotlib scipy scikit-learn pytorch torchvision cpuonly -c pytorch
jupyter notebook


