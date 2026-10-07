# Waste Classification Using Deep Learning

## About the Project

Waste Classification is a Deep Learning project that classifies waste images into different categories.

The model takes an image of waste as input and predicts which category it belongs to. This project shows how image classification can be done using Deep Learning.

## Objective

The main objectives of this project are:

- To classify different types of waste using images.
- To understand the basics of Deep Learning.
- To learn how image datasets are used for training a model.
- To train a model that can recognize different waste categories.
- To visualize the training results.

## Waste Categories

The dataset can contain different types of waste, such as:

- Plastic
- Paper
- Organic Waste
- Metal
- Glass

The categories depend on the images available in the dataset.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- VS Code

## Deep Learning Model

This project uses a Convolutional Neural Network (CNN) for image classification.

CNN is commonly used for image-based tasks because it can learn important features from images such as shapes, patterns, and textures.

## How the Project Works

The basic workflow of the project is:

1. Collect waste images.
2. Organize images into different folders according to their categories.
3. Load the image dataset.
4. Resize the images.
5. Normalize the image values.
6. Split the dataset into training and validation data.
7. Build the CNN model.
8. Train the model.
9. Test the model using images.
10. Display the prediction and training results.

## Project Structure

```text
Waste-Classification/
│
├── dataset/
│   ├── plastic/
│   ├── paper/
│   ├── organic/
│   ├── metal/
│   └── glass/
│
├── waste_classification.py
├── graph.png
├── README.md
└── requirements.txt