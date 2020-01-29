import numpy as np
J = np.ones((30, 300))/100
Q = np.eye(300)
lmbda = 0.1
delta_V = np.ones(30) /1000
capacity_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T), delta_V)
print(capacity_predict)