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


# Calculate MLE for Gaussian NB parameters
def calculate_gaussian_nb_params(X, y):
    classes = np.unique(y)
    params = {}
    for c in classes:
        X_c = X[y == c]
        params[c] = {
            'mean': np.mean(X_c, axis=0),
            'var': np.var(X_c, axis=0)
        }
        print(f'Class {c} MLE:')
        print(f'Mean: {params[c]["mean"]}')
        print(f'Variance: {params[c]["var"]}\n')
    return params


# Calculate log-likelihood
def log_likelihood(X, y, params):
    log_likelihood = 0
    for i in range(len(y)):
        c = y[i]
        mean = params[c]['mean']
        var = params[c]['var']
        log_likelihood += np.sum(-0.5 * np.log(2 * np.pi * var) - ((X[i] - mean) ** 2) / (2 * var))
    return log_likelihood


# Fit Gaussian NB model
params = calculate_gaussian_nb_params(X_train, y_train)
train_log_likelihood = log_likelihood(X_train, y_train, params)
print(f'Log-likelihood on training set: {train_log_likelihood}')


# Predict function
def predict(X, params):
    classes = list(params.keys())
    log_probs = []
    for c in classes:
        mean = params[c]['mean']
        var = params[c]['var']
        log_prob = np.sum(-0.5 * np.log(2 * np.pi * var) - ((X - mean) ** 2) / (2 * var), axis=1)
        log_probs.append(log_prob)
    return np.argmax(log_probs, axis=0)


# Make predictions on the test set
y_pred = predict(X_test, params)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')
