import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
sns.set_style('darkgrid') # Seaborn plotting style


class LogisticRegression:

    def __init__(self, learning_rate=0.01, iterations=10):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
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

        sample_count = len(X.index)
        feature_count = len(X.columns)

        # Preprocess data into an array with euclidean distance if it is the dataset nr. 2
        """if self.use_dataset2:
            X = np.array([np.sqrt(x[0] ** 2 + x[1] ** 2) for x in X])
            feature_count = 1
        """

        # Initialize the thetas/weights
        self.weights = np.zeros(feature_count + 1)

        # Add one column with ones for the bias theta/weight
        X['Bias'] = np.ones(sample_count)
        #X = np.c_[np.ones(sample_count), X]

        # Iterate through training data epochs nr. of times.
        for i in range(self.iterations):

            # Update thetas/weights using gradient decent for every sample. See the equation in the report.
            for j in range(sample_count):
                self.weights = self.weights + self.learning_rate * (y[j] - sigmoid(np.matmul(self.weights, X.iloc[j]))) * X.iloc[j]

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

        try:
            Xnum = X.to_numpy()
        except AttributeError:
            Xnum = X
            X = pd.DataFrame(Xnum)
        """
        X = pd.DataFrame(X)

        sample_count = len(X.index)
        # Add one column with ones for the bias theta/weight
        #X = np.c_[np.ones(sample_count), X]
        X['Bias'] = np.ones(sample_count)

        # Calculate the dot product between every sample and thetas/weights
        temp = np.zeros(sample_count)
        for i in range(sample_count):
            temp[i] = np.matmul(self.weights, Xnum[i])

        # Return float value between 0 and 1 with the use of sigmoid function
        return sigmoid(temp)



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
