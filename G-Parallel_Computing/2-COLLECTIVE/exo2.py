from mpi4py import MPI
import numpy as np
import pandas as pd
from sympy import sympify
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

# Initialize comm utils
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Data generator
def generate_dataset(relation="2*x1 + 3*x2 - 9*x3", n_samples=100, with_noise=True, random_state=None) -> pd.DataFrame:
    if random_state:
        np.random.seed(random_state)

    # Parse the relation string
    expr = sympify(relation)
    variables = expr.free_symbols

    # Generate feature and target
    X = np.random.rand(n_samples, len(variables))
    coefficients = [expr.coeff(v) for v in variables]
    noise = np.random.randn(n_samples) * 0.1 if with_noise else 0
    y = np.dot(X, coefficients) + noise

    # Create DataFrame
    columns = [f"x{i+1}" for i in range(len(variables))]
    data = pd.DataFrame(X, columns=columns)
    data['y'] = y
    return data

# Parallel Regressor
class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, n_epochs=1000, random_state=None, batch_size=1): 
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)
        self.weights = None
        self.bias = None
        self.J_history = []
        self.weights_history = []
        self.batch_size = batch_size
        self.computation_time = 0

    def compute_cost(self, X, y, w, b):
        m = X.shape[0]
        cost_i = np.dot(w, X.T) + b - y
        return 1/(2*m) * np.dot(cost_i, cost_i.T)
    
    def compute_gradient(self, X, y, w, b):
        m = X.shape[0]
        predictions = np.dot(w, X.T) + b
        errors = predictions - y.T
        dJ_w = 1/m * np.dot(errors, X)
        dJ_b = 1/m * np.dot(errors, np.ones(m))
        return dJ_w.flatten(), dJ_b
    
    def fit(self, X, y, verbose=True):
        self.computation_time = MPI.Wtime()
        X = X.to_numpy()
        y = y.to_numpy()

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

            # Split data
            if RANK == 0:
                training_batch = np.array_split(train_set, SIZE, axis=0)
            else:
                training_batch = None
            local_training_set = COMM.scatter(sendobj=training_batch, root=0)
            
            # Slip target and features locally
            X_batch, y_batch = local_training_set[:, :-1], local_training_set[:, -1:]

            # Compute local gradient
            dJ_w_loc, dJ_b_loc = self.compute_gradient(X_batch, y_batch, self.weights, self.bias)

            # Sum all gradient and Update weights
            COMM.Barrier()
            dJ_w = COMM.allreduce(dJ_w_loc, op=MPI.SUM)
            dJ_b = COMM.allreduce(dJ_b_loc, op=MPI.SUM)
            self.weights = self.weights - self.learning_rate * dJ_w
            self.bias    = self.bias - self.learning_rate * dJ_b

            # Save cost at each iteration
            if RANK==0:
                if i<1000:
                    cost_J =  self.compute_cost(X, y, self.weights, self.bias)
                    self.J_history.append(cost_J)

                # Print cost every at intervals 10 times
                if verbose and i%math.ceil(self.n_epochs/10) == 0:
                    self.weights_history.append(self.weights)
                    print(f"Epoch: [{i}{' '*(len(str(self.n_epochs))-len(str(i)))}/{self.n_epochs}]\t Cost: {float(self.J_history[-1]):8.2f}\t weights: {self.weights_history[-1]}")

        # Timeit
        self.computation_time = MPI.Wtime() - self.computation_time
                        
    def predict(self, X):
        if RANK==0:
            X = X.to_numpy()
            predictions = np.dot(X, self.weights) + self.bias
            return pd.DataFrame(predictions)
    
    def summary(self):
        if RANK==0:
            print("\nMODEL SUMMARY:")
            print(f"Computation time: {self.computation_time}")
            print(f"Nbr Processor\t: {SIZE}")
            print(f"Learning Rate\t: {self.learning_rate}")
            print(f"Number of Epochs: {self.n_epochs}")
            print(f"Model Weights\t: {self.weights}")
            print(f"Model Bias\t: {self.bias}")
            # Plot
            plt.plot(range(self.n_epochs), self.J_history)
            plt.title('Cost Function Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.show()

# Synthetic dataset
dataset = generate_dataset(relation="2*x1 - 8*x2 + 5*x3 + x4", n_samples=10000, with_noise=True, random_state=42)
X = dataset.drop(columns=['y'])
y = dataset['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Main
model = LinearRegressionSGD(learning_rate=0.1, n_epochs=100, random_state=42)
model.fit(X_train, y_train)
model.summary()

# Prediction
if RANK == 0:
    y_test.reset_index(drop=True, inplace=True)
    predictions = model.predict(X_test)
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Comparison between Actual and Predicted values')
    plt.legend()
    plt.show()