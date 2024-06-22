# Fashion MNIST Classification

This project demonstrates the use of a Convolutional Neural Network (CNN) with data augmentation techniques to classify images from the Fashion MNIST dataset. The model is built using TensorFlow and Keras, and includes custom callbacks for progress tracking and early stopping.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Overview

This repository contains code to train a CNN on the Fashion MNIST dataset, using various techniques such as data augmentation, learning rate scheduling, and custom callbacks to enhance model performance and training experience. The code is provided in a single Jupyter Notebook for easy execution and experimentation.

## Dataset

The Fashion MNIST dataset is a collection of 70,000 grayscale images of 28x28 pixels, each belonging to one of 10 fashion categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Model Architecture

The model architecture consists of the following layers:
- Data augmentation layers
- Three convolutional layers with batch normalization and max pooling
- A dense layer with dropout for regularization
- An output layer with 10 units (one for each class)

## Data Augmentation

Data augmentation techniques used include:
- Random horizontal flip
- Random rotation
- Random zoom

These augmentations help in improving the generalization of the model.

## Training

The model is trained using:
- Sparse Categorical Crossentropy loss
- Adam optimizer with an exponential decay learning rate schedule
- Early stopping callback to prevent overfitting
- Custom progress bar callback for better training visualization

## Evaluation

The model's performance is evaluated on the test set, and the training history is visualized through accuracy and loss plots.

## Results

The trained model achieves high accuracy on the test set, demonstrating the effectiveness of the chosen architecture and training strategies.

## Usage

To use this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/J3lly-Been/fashion-mnist-classification.git
    cd fashion-mnist-classification
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Open and run the Jupyter Notebook:
    ```sh
    jupyter notebook fashion_mnist_classification.ipynb
    ```

## Requirements

- TensorFlow
- NumPy
- Matplotlib
- TQDM

You can install the required packages using:
```sh
pip install tensorflow numpy matplotlib tqdm
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
