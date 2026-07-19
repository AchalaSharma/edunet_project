# Mango Leaf Disease Classification

## Overview
This repository contains a computer vision and machine learning pipeline for classifying mango leaf diseases. The project extracts handcrafted features (color, shape, texture, and edge) from leaf images and evaluates multiple machine learning models to identify the disease category accurately.

**Author:** Achala Sharma 

## Dataset
The dataset (`Resize mango leaf disease`) consists of images categorized into five distinct classes, with 600 images per class:
*   **Anthracnose**
*   **Bacterial Canker**
*   **Die Back**
*   **Gall Midge**
*   **Healthy**

## Feature Extraction Pipeline
The pipeline processes the images (resized to 256x256) and extracts a comprehensive set of 123 features across four distinct categories. These features are initially saved as separate CSV files and ultimately merged into a single `features.csv` dataset.

1.  **Color Features (`color_features.csv`):** 
    * RGB and HSV means and standard deviations.
    * Green pixel percentage.
    * Flattened RGB histograms (32 bins per channel).
2.  **Shape Features (`shape_features.csv`):** 
    * Contour area and perimeter.
    * Bounding box aspect ratio.
    * Circularity and contour point counts.
3.  **Texture Features (`texture_features.csv`):** 
    * Gray-Level Co-occurrence Matrix (GLCM) features: contrast, correlation, and homogeneity.
    * Histogram entropy.
    * Local Binary Patterns (LBP) histogram mean.
    * Gabor filter mean response.
    * Discrete Wavelet Transform (DWT) energy.
4.  **Edge Features (`edge_features.csv`):** 
    * Canny edge count and edge pixel ratio.
    * Sobel gradient magnitude (mean and standard deviation).
    * Laplacian variance.

## Models and Performance
After standardizing the combined feature set using `StandardScaler`, the dataset is split (70% train, 30% test) and evaluated across five different machine learning classifiers. 

### Accuracy Results:
*   **Support Vector Machine (SVM - Linear Kernel):** ~92.2% *(Best Performing)*
*   **Random Forest (100 Estimators):** ~91.1%
*   **K-Nearest Neighbors (KNN - k=5):** ~87.7%
*   **Decision Tree:** ~83.6%
*   **Naive Bayes (Gaussian):** ~66.3%

## Installation and Requirements
To run the `cv_mango_leaf.ipynb` notebook, you will need the following Python libraries installed:

```bash
pip install numpy pandas matplotlib seaborn opencv-python-headless pillow scipy scikit-image scikit-learn pywavelets tqdm flask keras tensorflow

