import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegressionGD:
    """
    Linear regression solver with Simple gradien descent optimizer
    """
    def __init__(self, learning_rate=0.01, n_epochs=1000, random_state=None, eps=1e-06):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.J_history = []
        self.weights_history = []
        self.eps = eps

    def cost(self, X, y, w, b):
        m = X.shape[0]
        cost_i = np.dot(w, X.T) + b - y
        return 1/(2*m) * np.dot(cost_i, cost_i.T)
    
    def gradient(self, X, y, w, b):
        m = X.shape[0]
        f_X = np.dot(w, X.T) + b
        dJ_w = 1/m * np.dot(f_X - y.T, X)
        dJ_b = 1/m * np.dot(f_X - y.T, np.ones(m))
        return dJ_w, dJ_b

    def fit(self, X, y, verbose=True):
        X = X.to_numpy()
        y = y.to_numpy()

        if self.random_state:
            np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        for i in range(self.n_epochs):
            # Calculate the gradient
            dJ_w, dJ_b = self.gradient(X, y, self.weights, self.bias)

            # Update weights
            self.weights = self.weights - self.learning_rate * dJ_w
            self.bias    = self.bias - self.learning_rate * dJ_b

            # Save cost at each iteration
            if i<1000:
                cost_J =  self.cost(X, y, self.weights, self.bias)
                self.J_history.append(cost_J)

            # Print cost every at intervals 10 times
            if verbose and i%math.ceil(self.n_epochs/10) == 0:
                self.weights_history.append(self.weights)
                print(f"Epoch: [{i}{' '*(len(str(self.n_epochs))-len(str(i)))}/{self.n_epochs}]\t Cost: {float(self.J_history[-1]):8.2f}\t weights: {self.weights_history[-1]}")

            # Tolerance
            #print(type(dJ_w))
            #if np.linalg.norm(dJ_w) < self.eps:
            #    break

    def predict(self, X):
        X = X.to_numpy()
        predictions = np.dot(X, self.weights) + self.bias
        return pd.DataFrame(predictions)
    
    def summary(self):
        print("MODEL SUMMARY:")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of Epochs: {self.n_epochs}")
        print(f"Random State: {self.random_state}")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")

        # Plot cost function over epochs
        plt.plot(range(self.n_epochs), self.J_history)
        plt.title('Cost Function Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()


class LinearRegressionSGD:
    """
    Linear regression solver with Stochastique gradien descent optimizer
    """
    def __init__(self, learning_rate=0.01, n_epochs=1000, random_state=None, batch_size=1): 
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.J_history = []
        self.weights_history = []
        self.batch_size = batch_size

    def compute_cost(self, X, y, w, b):
        m = X.shape[0]
        cost_i = np.dot(w, X.T) + b - y
        return 1/(2*m) * np.dot(cost_i, cost_i.T)
    
    def compute_gradient(self, data, labels, weight, b=0):
        """
        NB: We change this to increase its versatility, and we add bias term
        ---
        def compute_gradient(data, labels, weight):
            predictions = data * weight
            errors = predictions - labels
            gradient = 2 * np.dot(data, errors) / len(data)
            return gradient
        ---
        """
        m = data.shape[0]
        predictions = np.dot(weight, data.T) + b
        errors = predictions - labels.T
        dJ_w = 2/m * np.dot(errors, data)
        dJ_b = 2/m * np.dot(errors, np.ones(m))
        return dJ_w, dJ_b
    

    def fit(self, X, y, verbose=True):
        X = X.to_numpy()
        y = y.to_numpy()

        # random
        if self.random_state:
            np.random.seed(self.random_state)

        # initilization
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        # concatenate training set
        rng = np.random.default_rng(seed=self.random_state)
        train_set = np.c_[X.reshape(n_samples, -1), y.reshape(n_samples, 1)]

        for i in range(self.n_epochs):
            # shuffle training set
            rng.shuffle(train_set)

            # Batch 
            for start in range(0, n_samples, self.batch_size):    
                end = start + self.batch_size
                X_batch, y_batch = train_set[start:end, :-1], train_set[start:end, -1:]

                # Calculate the gradient
                dJ_w, dJ_b = self.compute_gradient(X_batch, y_batch, self.weights, self.bias)

                # Update weights
                self.weights = self.weights - self.learning_rate * dJ_w
                self.bias    = self.bias - self.learning_rate * dJ_b

            # Save cost at each iteration
            if i<1000:
                cost_J =  self.compute_cost(X, y, self.weights, self.bias)
                self.J_history.append(cost_J)

            # Print cost every at intervals 10 times
            if verbose and i%math.ceil(self.n_epochs/10) == 0:
                self.weights_history.append(self.weights)
                print(f"Epoch: [{i}{' '*(len(str(self.n_epochs))-len(str(i)))}/{self.n_epochs}]\t Cost: {float(self.J_history[-1]):8.2f}\t weights: {self.weights_history[-1]}")

    def predict(self, X):
        X = X.to_numpy()
        predictions = np.dot(X, self.weights) + self.bias
        return pd.DataFrame(predictions)
    
    def summary(self):
        print("MODEL SUMMARY:")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of Epochs: {self.n_epochs}")
        print(f"Random State: {self.random_state}")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")

        # Plot cost function over epochs
        plt.plot(range(self.n_epochs), self.J_history)
        plt.title('Cost Function Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()