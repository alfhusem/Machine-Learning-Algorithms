import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logistic_regression as lr   # <-- Your implementation

#sns.set_style('darkgrid') # Seaborn plotting style

data_1 = pd.read_csv('data_1.csv')
data_1.head()

X = data_1[['x0', 'x1']]
y = data_1['y']

"""
n_samples = len(X.index)
feature_count = len(X.columns)

#n_features = X.shape[1]
#n_samples = X.shape[0]

print(X)
print(y)

print(n_features)
print(n_samples)


plt.figure(figsize=(5, 5))
sns.scatterplot(x='x0', y='x1', hue='y', data=data_1);

# Partition data into independent (feature) and depended (target) variables
X = data_1[['x0', 'x1']]
y = data_1['y']


weights = np.zeros(feature_count)

print(y)
print('---')
print(y[1])
print(weights)
print()
print(X)
print('---')
print(X.iloc[1])
print('-----')

for j in range(n_samples):
    weights = weights + 0.01 * (y[j] - lr.sigmoid(np.matmul(weights, X.iloc[j]))) * X.iloc[j]



# Create and train model.
model_1 = lr.LogisticRegression() # <-- Should work with default constructor
model_1.fit(X, y)

# Calculate accuracy and cross entropy for (insample) predictions
y_pred = model_1.predict(X)
print(f'Accuracy: {lr.binary_accuracy(y_true=y, y_pred=y_pred, threshold=0.5) :.3f}')
print(f'Cross Entropy: {lr.binary_cross_entropy(y_true=y, y_pred=y_pred) :.3f}')
"""

X = X.to_numpy()
sample_count = X.shape[0]
print(sample_count)
