import numpy as np
import matplotlib.pyplot as plt

Ts = 0.1

C = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 1]])

G = 0.01 * np.eye(6)
Q = 0.01 * np.eye(6)
R = 0.01 * np.eye(3)


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
    vt = np.random.multivariate_normal(np.zeros(3), R).reshape(-1, 1)
    y = C @ x + vt
    return y


def measurement_update(x, sigma, y):
    lhs = C @ sigma @ (C.T) + R
    rhs = y - C @ x
    x_meas = x + sigma @ (C.T) @ np.linalg.solve(lhs, rhs)

    rhs1 = C @ sigma
    sigma_meas = sigma - sigma @ (C.T) @ np.linalg.solve(lhs, rhs1)
    return x_meas, sigma_meas


def time_update(x_meas, sigma_meas, tau):
    A = matrix_a(tau)
    x_time = A @ x_meas
    sigma_time = A @ sigma_meas @ (A.T) + G @ Q @ (G.T)
    return x_time, sigma_time


def height_control2(arr, tau):
    if arr[-1] > 1.01 * arr[-2]:
        tau -= 0.1
        if arr[-1] > 1.05 * arr[-2]:
            tau -= 0.25
        if arr[-1] > 1.1 * arr[-2]:
            tau = 0.02

    elif arr[-1] < 0.99 * arr[-2]:
        tau += 0.1
        if arr[-1] < 0.95 * arr[-2]:
            tau += 0.25
        if arr[-1] < 0.9 * arr[-2]:
            tau = 0.98

    if tau <= 0:
        tau = 0.01
    elif tau >= 1:
        tau = 0.99
    return tau

choice = input("The GNSS sensor measurement is missing. \nDo you want to use emergency plan (y/n)?")
if choice == 'y':
    print("When the y_gnss signal is lost, y_tof replaces gnss")
else:
    print("x_measurement drops because no y_gnss")

x_true = np.array([[10], [0.1], [-0.5], [1], [1], [1]])
x_pred = np.array([[9], [0.1], [-1], [2], [1], [1]])
sigma_pred = np.diag([1, 1, 2, 1, 1, 1])

x_true_arr, x_pred_arr, x_meas_arr = [], [], []
height = [10]
N = 50
tau = 0.5
for t in range(25):
    y = output(x_true)
    x_meas, sigma_meas = measurement_update(x_pred, sigma_pred, y)
    x_pred, sigma_pred = time_update(x_meas, sigma_meas, tau)
    x_true = dynamics(x_true, tau)

    height.append(x_true[0])
    x_true_arr.append(x_true)
    x_pred_arr.append(x_pred)
    x_meas_arr.append(x_meas)

    tau = height_control2(height, tau)

    if x_true[0] <= 1:
        tau = 1


for t in range(25):
    y = output(x_true)
    if choice == 'y':
        y[0] = y[1]
    else:
        y[0] = 0
    x_meas, sigma_meas = measurement_update(x_pred, sigma_pred, y)
    x_pred, sigma_pred = time_update(x_meas, sigma_meas, tau)
    x_true = dynamics(x_true, tau)

    height.append(x_true[0])
    x_true_arr.append(x_true)
    x_pred_arr.append(x_pred)
    x_meas_arr.append(x_meas)

    tau = height_control2(height, tau)

    if x_true[0] <= 1:
        tau = 1


# plot
x_plot = np.arange(N)
y_plot, y_plot2, y_plot3 = [], [], []

for i in range(len(x_true_arr)):
    y_plot.append(x_true_arr[i][0])
    y_plot2.append(x_meas_arr[i][0])
    # y_plot3.append(x_pred_arr[i][0])

plt.plot(x_plot * 0.1, y_plot, label="x_true")
plt.plot(x_plot * 0.1, y_plot2, label="x_meas")
# plt.plot(x_plot, y_plot3, label="x_pred")
plt.legend()
plt.ylim(-10, 30)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.show()

mse = np.mean((np.array(y_plot) - np.array(y_plot2)) ** 2)
print("MSE =", mse)
