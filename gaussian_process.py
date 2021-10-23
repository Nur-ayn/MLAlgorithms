import numpy as np
import matplotlib.pyplot as plt
from helpers.plotting import adjust_yerr

L_2 = 10
SIGMA_2 = 0.05


def exponential_cov(x_test, x):
    return np.exp(-0.5*L_2*np.subtract.outer(x_test, x)**2)

def predict(x_test, x_data, y_data, kernel, k):
    k_star = [ kernel(x_test, x_d) for x_d in x_data ]
    k_inv = np.linalg.inv(k)
    y_pred = np.dot(k_star, k_inv).dot(y_data)
    sigma_new = kernel(x_test, x_test) - np.dot(k_star, k_inv).dot(k_star)
    return y_pred, sigma_new
  

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, facecolor='lightpink')

xpts = np.linspace(-3, 3, 1000)
ypts = np.zeros(len(xpts))
sigma_0 = exponential_cov(0, 0)

errorbar = ax.errorbar(xpts, ypts, yerr=sigma_0, mfc='indianred', mec='darkred', ecolor='lightcoral', marker='o', markersize=3, linestyle='None', capsize=0.5)
ln, cap, bar = errorbar
ax.set_title('Nonlinear Regression with a Gaussian Process')
ax.set_xlabel('x')
ax.set_ylabel('distribution over f(x)')

x, y = [], []

def on_press(event):
    global x, y
    if event.button == 1:   
        x_new, y_new = event.xdata, event.ydata
        x.append(x_new)
        y.append(y_new)
        print('x and y = ', x, y)
        k_x = exponential_cov(x, x)
        predictions = [ predict(pt, x, y, exponential_cov, k_x + SIGMA_2*np.eye(k_x.shape[0])) for pt in xpts]
        y_pred, sigmas = np.transpose(predictions)
        ln.set_data(xpts, y_pred)
        adjust_yerr(errorbar, sigmas)
        ax.plot(x,y, 'P', color='yellow', markersize=5, zorder=5)
        plt.pause(0.5)

pressID = fig.canvas.mpl_connect('button_press_event', on_press)

plt.show()