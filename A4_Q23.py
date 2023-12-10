import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

A = np.array([[0.8, 0.6], [-0.7, 0.3]])  # system matrix A
B = np.eye(2)  # input matrix
C = np.eye(2)  # output weighting matrix
D = np.eye(2)  # input weighting matrix

# continuous-time algebraic Riccati equation
P = solve_continuous_are(A, B, C, D)
print(P)
# ellipsoidal set parameters
theta = np.linspace(0, 2 * np.pi, 100)
ellipse = np.array([np.cos(theta), np.sin(theta)])
# boundary of set
boundary = np.linalg.inv(np.linalg.cholesky(P)).dot(ellipse)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(boundary[0, :], boundary[1, :], label=r'$\varepsilon: x^T P x \leq 1$')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Ellipsoidal Invariant Set')
plt.legend()
plt.grid(True)
plt.show()
