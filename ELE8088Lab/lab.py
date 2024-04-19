import numpy as np
import matplotlib.pyplot as plt

Ts = 0.1

C = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 1]])

G = np.eye(6)
Q = np.eye(6)
R = np.eye(3)

alpha = 1
belta = 1
m = 1

zt = 0
vt = np.random.multivariate_normal(np.zeros(3), R).reshape(-1, 1)
alpha_t0 = 0
alpha_t1 = 0
d_tof = 0
d_bar = 0

y_gnss = 0
y_tof = 0
y_bar = 0

x0 = np.vstack((zt, vt, alpha_t0, alpha_t1, d_tof, d_bar))
y0 = np.vstack((y_gnss, y_tof, y_bar))


def matrix_a(tau):
    A = np.array(
        [[1, Ts, 0, 0, 0, 0],
         [0, 1, Ts, Ts * tau, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]
    )
    return A


def dynamics(x, tau):
    A = matrix_a(tau)
    w = np.random.multivariate_normal(np.zeros(6), Q).reshape(-1, 1)
    x_next = A @ x + w
    return x_next


def output(x):
    y = C @ x + vt
    return y


def measurement_update(x, sigma, y):
    lhs = C @ sigma @ (C.T) + R
    rhs = y - C @ x
    x_meas = x - sigma @ (C.T) @ np.linalg.solve(lhs, rhs)

    rhs1 = C @ sigma
    sigma_meas = sigma - sigma @ (C.T) @ np.linalg.solve(lhs, rhs1)
    return x_meas, sigma_meas


def time_update(x_meas, sigma_meas, tau):
    A = matrix_a(tau)
    x_time = A @ x_meas
    sigma_time = A @ sigma_meas @ (A.T) + G @ Q @ (G.T)
    return x_time, sigma_time


x_true = np.array([[1], [0.5], [-10], [20], [10], [10]])
x_pred = np.array([[0], [0], [-8], [25], [9], [8]])
sigma_pred = np.diag([5, 1, 25, 100, 200, 200])

x_true_arr, x_pred_arr, x_meas_arr = [], [], []
N = 50
for t in range(N):
    y = output(x_true)
    tau = 0.5
    x_meas, sigma_meas = measurement_update(x_pred, sigma_pred, y)
    x_pred, sigma_pred = time_update(x_meas, sigma_meas, tau)
    x_true = dynamics(x_true, tau)

    x_true_arr.append(x_true)
    x_pred_arr.append(x_pred)
    x_meas_arr.append(x_meas)

# plot
x_plot = np.arange(N)
y_plot, y_plot2, y_plot3 = [], [], []

for i in range(len(x_true_arr)):
    y_plot.append(x_true_arr[i][0])
    # y_plot2.append(x_meas_arr[i][0])
    # y_plot3.append(x_pred_arr[i][0])

plt.plot(x_plot, y_plot, label="x_true")
# plt.plot(x_plot, y_plot2, label="x_meas")
# plt.plot(x_plot, y_plot3, label="x_pred")
plt.legend()
plt.show()
