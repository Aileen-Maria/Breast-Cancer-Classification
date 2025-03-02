# Breast Cancer Classification

This repository contains a Jupyter notebook that implements various machine learning classification algorithms for the task of breast cancer classification using the breast cancer dataset available in the sklearn library.

## Objective

The goal of this project is to apply supervised learning techniques to a real-world dataset (breast cancer dataset) and compare the performance of five classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. k-Nearest Neighbors (k-NN)

## Dataset

The dataset used in this project is the **Breast Cancer Dataset** from the `sklearn.datasets` module. It contains data about cell nuclei features for each sample, where each sample is labeled as either malignant (cancerous) or benign (non-cancerous).

The dataset consists of:
- **Features**: 30 numerical features describing the characteristics of the cell nuclei.
- **Target**: Binary classification (Malignant/Benign).

## Preprocessing

The following preprocessing steps were performed on the dataset:
1. **Missing Value Handling**: Checked for and handled any missing values in the dataset.
2. **Feature Scaling**: Applied **StandardScaler** for scaling the feature values to a standard range, ensuring better performance of some algorithms like SVM and k-NN.

## Algorithms Implemented

The following machine learning algorithms were used to train the model and classify the data:

### 1. Logistic Regression
Logistic Regression is a linear model for binary classification. It works by estimating the probability that a given input point belongs to a particular class. It is simple and works well with linearly separable data.

### 2. Decision Tree Classifier
Decision Trees work by splitting the data into subsets based on the most significant feature at each node. This process continues recursively, forming a tree-like structure. It is suitable for both classification and regression tasks.

### 3. Random Forest Classifier
Random Forest is an ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction. It helps reduce the risk of overfitting which is common in decision trees.

### 4. Support Vector Machine (SVM)
SVM is a powerful classification technique that works by finding the hyperplane that best divides the data into different classes. It works well with high-dimensional data and is effective in cases where the number of dimensions exceeds the number of samples.

### 5. k-Nearest Neighbors (k-NN)
k-NN is a non-parametric, lazy learning algorithm that classifies a data point based on the majority class among its nearest neighbors. It is easy to implement and works well when the decision boundary is non-linear.

## Model Evaluation

The performance of each model was evaluated using accuracy as the primary metric. The following steps were taken:

1. Split the data into training and test sets (80-20%).
2. Train each model on the training set.
3. Evaluate the model's accuracy on the test set.

## Results

A comparison of the models' accuracy will help identify which algorithm performed the best and which one performed the worst.

## Requirements

To run the code, make sure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
