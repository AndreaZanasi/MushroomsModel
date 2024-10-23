# 🍄 Mushroom Classification Model

This repository contains code for training and evaluating a deep learning model to classify different types of mushrooms. The model is based on the ResNet-18 architecture and is trained using PyTorch.

## 📁 Repository Structure
### Files

- **main.py**: This is the main script for training the model. It includes functions for setting up the device, loading data, training the model, and evaluating the model on the test set.
  - `set_device()`: Determines whether to use a GPU or CPU for training.
  - `train_nn(model, train_loader, test_loader, criterion, optimizer, epochs)`: Trains the neural network.
  - `evaluate_model_on_test_set(model, test_loader)`: Evaluates the model on the test set.
  - `pil_loader(path)`: Custom image loader to handle truncated images.
  - Data transformations and data loaders for training and testing datasets.
  - Model setup and training loop.

- **mean_and_std.py**: Contains a function to calculate the mean and standard deviation of the dataset.
  - `get_mean_and_std(loader)`: Calculates the mean and standard deviation for each batch and averages them.

- **testing.py**: Script for testing the trained model on individual images.
  - `predict_image(image_path, class_names)`: Predicts the class of a given image.
  - Loads a pre-trained model and evaluates it on a set of test images.

- **weights/**: Directory containing pre-trained model weights saved during training.

## 🚀 Usage

### 🏋️‍♂️ Training the Model

1. Ensure you have the required dependencies installed:
   ```sh
   pip install torch torchvision pillow colorama
   ```
2. Prepare your dataset:
   - Place your training images in the `Mushrooms/` directory, organized by class.
   - Ensure the `Mushrooms/` directory is not tracked by Git by adding it to `.gitignore`.

3. Run the `main.py` script to start training:
   ```sh
   python main.py
   ```

### 🧪 Testing the Model

1. Ensure you have a `Test/` folder containing the images you want to test.

2. Run the `testing.py` script to test the trained model on individual images:
   ```sh
   python testing.py
   ```

3. The `predict_image(image_path, class_names)` function will load a pre-trained model and predict the class of the given images.

## 📂 Dataset Folder Structure

- `Mushrooms/`: Contains subdirectories for each mushroom class with corresponding images.
- `Test/`: Contains images for testing the trained model.

## 📊 Model Weights

- The `weights/` directory contains pre-trained model weights saved during training.
