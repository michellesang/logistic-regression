import numpy as np
import pandas as pd

# Load data
train_data = pd.read_csv('sonar_train.data', header=None)
test_data = pd.read_csv('sonar_test.data', header=None)
valid_data = pd.read_csv('sonar_valid.data', header=None)

y_train = train_data.iloc[:, -1].values
X_train = train_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_valid = valid_data.iloc[:, -1].values
X_valid = valid_data.iloc[:, :-1].values

# Convert labels to binary (1 → 0, 2 → 1)
y_train = (y_train == 2).astype(int)
y_test = (y_test == 2).astype(int)
y_valid = (y_valid == 2).astype(int)


# Feature scaling
def feature_scaling(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


# Scale training and test data using training set statistics
X_train, mean, std = feature_scaling(X_train)
X_test, _, _ = feature_scaling(X_test, mean, std)
X_valid, _, _ = feature_scaling(X_valid, mean, std)


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Gradient descent with regularization
def gradient_descent_l2(X, y, theta, learning_rate, iterations, lambda_):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y)) + (lambda_ / m) * np.hstack(([0], theta[1:]))
        theta -= learning_rate * gradient
    return theta


def gradient_descent_l1(X, y, theta, learning_rate, iterations, lambda_):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y)) + (lambda_ / m) * np.sign(theta)
        gradient[0] -= (lambda_ / m) * np.sign(theta[0])  # No regularization for bias term
        theta -= learning_rate * gradient
    return theta


# Add intercept term to X
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_valid = np.hstack((np.ones((X_valid.shape[0], 1)), X_valid))

# Initialize parameters
theta = np.zeros(X_train.shape[1])
learning_rate = 0.001
iterations = 10000
lambda_ = 0.1

# Train the model
theta = gradient_descent_l2(X_train, y_train, theta, learning_rate, iterations, lambda_)


# Predict function
def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5


# Make predictions on the test set
y_pred = predict(X_test, theta)

print(f'Logistic regression classifier:')
# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# Initialize parameters
learning_rate = 0.01
iterations = 10000
lambda_values = [0.01, 0.1, 1, 10, 100]
best_lambda = None
best_accuracy = 0
best_theta = None

# Train the model with different regularization constants
for lambda_ in lambda_values:
    theta = np.zeros(X_train.shape[1])
    theta = gradient_descent_l2(X_train, y_train, theta, learning_rate, iterations, lambda_)
    y_pred_valid = sigmoid(X_valid @ theta) >= 0.5
    accuracy = np.mean(y_pred_valid == y_valid)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_lambda = lambda_
        best_theta = theta

print(f'\nLogistic regression classifier with l2 penalty:')
# Report the selected constant, learned weights, and bias
print(f'Selected regularization constant: {best_lambda}')
print(f'Learned weights: {best_theta[1:]}')
print(f'Learned bias: {best_theta[0]}')

# Calculate accuracy on the test set
y_pred_test = sigmoid(X_test @ best_theta) >= 0.5
test_accuracy = np.mean(y_pred_test == y_test)
print(f'Accuracy on test set: {test_accuracy * 100:.2f}%')


# Initialize parameters
learning_rate = 0.01
iterations = 10000
lambda_values = [0.01, 0.1, 1, 10, 100]
best_lambda = None
best_accuracy = 0
best_theta = None

for lambda_ in lambda_values:
    theta = np.zeros(X_train.shape[1])
    theta = gradient_descent_l1(X_train, y_train, theta, learning_rate, iterations, lambda_)
    y_pred_valid = sigmoid(X_valid @ theta) >= 0.5
    accuracy = np.mean(y_pred_valid == y_valid)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_lambda = lambda_
        best_theta = theta

print(f'\nLogistic regression classifier with l1 penalty:')
# Report the selected constant, learned weights, and bias
print(f'Selected regularization constant: {best_lambda}')
print(f'Learned weights: {best_theta[1:]}')
print(f'Learned bias: {best_theta[0]}')

# Calculate accuracy on the test set
y_pred_test = sigmoid(X_test @ best_theta) >= 0.5
test_accuracy = np.mean(y_pred_test == y_test)
print(f'Accuracy on test set: {test_accuracy * 100:.2f}%')

