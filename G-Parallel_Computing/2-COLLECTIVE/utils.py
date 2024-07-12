import numpy as np
import pandas as pd
from sympy import sympify

"""
Objective: This function build synthetic dataset for simple linear regression model
"""

def generate_dataset(relation="2*x1 + 3*x2 - 9*x3", n_samples=100, with_noise=True, random_state=None) -> pd.DataFrame:
    if random_state:
        np.random.seed(random_state)

    # Parse the relation string
    expr = sympify(relation)
    variables = expr.free_symbols

    # Generate feature and target
    X = np.random.rand(n_samples, len(variables))
    coefficients = [expr.coeff(v) for v in variables]
    noise = np.random.randn(n_samples) * 1 if with_noise else 0
    y = np.dot(X, coefficients) + noise

    # Create DataFrame
    columns = [f"x{i+1}" for i in range(len(variables))]
    data = pd.DataFrame(X, columns=columns)
    data['y'] = y
    return data

