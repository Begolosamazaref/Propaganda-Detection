# BackEndCode AMD Classification

## Overview
This directory contains the backend implementation for Age-related Macular Degeneration (AMD) classification using a hybrid deep learning approach. The code implements a novel architecture that combines a Scale Adaptive (SA) model with transfer learning using ResNet50 to achieve high accuracy in AMD classification.

## Technical Architecture
- **Scale Adaptive Model**: Custom encoder model using MSE and MSLE loss functions
- **Transfer Learning**: Integration with ResNet50 pre-trained on ImageNet
- **Classification Layers**: Multi-layer neural network with batch normalization and dropout
- **Input Shape**: 448x448x3 for the SA model, 224x224x3 for ResNet50
- **Output**: 4-class classification (GA, Intermediate, Normal, Wet)

## Implementation Details
- **Model Building (01_model_building.ipynb)**:
  - Custom loss functions: RMSE, SSIM, and hybrid MSE-MSLE loss
  - Feature extraction using Scale Adaptive model + ResNet50
  - Classification layers with dropout for regularization
  - Global Average Pooling and Flattening for feature reduction

- **Training & Evaluation (02_model_training &Eval.ipynb)**:
  - Dataset splitting (70% training, 15% validation, 15% testing)
  - Data augmentation with horizontal/vertical flips and rotation
  - SGD optimizer with learning rate of 0.001
  - Categorical cross-entropy loss function
  - Batch size of 64 with 10 epochs

- **Prediction (03_model_prediction.ipynb)**:
  - Image preprocessing pipeline (resizing, normalization)
  - Model loading and inference
  - Classification into 4 AMD categories

## Dependencies
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-image
- TensorFlow Probability
- Matplotlib
- Seaborn

## Environment Setup
1. Install Anaconda from https://www.anaconda.com/download
2. Create environment using the provided YAML file:
   ```
   conda env create -f Requirments_capstone.yml
   ```
3. Activate the environment:
   ```
   conda activate capstone
   ```
4. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

## Model Performance
The hybrid architecture achieves 97% accuracy in classifying AMD stages, outperforming traditional CNN approaches through the combination of scale-adaptive features and transfer learning. 