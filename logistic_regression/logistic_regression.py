import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:

    def __init__(self, data2=False, learning_rate=0.01, iterations=1000):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.data2 = data2
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None


    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels

        """

        X = X.to_numpy()
        y = y.to_numpy()

        sample_count = X.shape[0]
        feature_count = X.shape[1]

        if self.data2:
            list = []
            for x in X:
                list.append(np.sqrt(x[0] ** 2 + x[1] ** 2))
            X = np.array(list)
            feature_count = 1

        self.weights = np.zeros(feature_count + 1)


        X = np.c_[np.ones(sample_count), X]

        for i in range(self.iterations):
            for j in range(sample_count):
                self.weights = self.weights + self.learning_rate * (y[j] - sigmoid(np.dot(self.weights, X[j])))*X[j]

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions

        """

        if not isinstance(X, (np.ndarray) ):
            X = X.to_numpy()

        if self.data2:
            list = []
            for x in X:
                list.append(np.sqrt(x[0] ** 2 + x[1] ** 2))
            X = np.array(list)

        sample_count = X.shape[0]

        X = np.c_[np.ones(sample_count), X]

        array = np.zeros(sample_count)
        for i in range(sample_count):
            array[i] = np.dot(self.weights, X[i])

        return sigmoid(array)



# --- Some utility functions

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input
    """
    return 1. / (1. + np.exp(-x))
