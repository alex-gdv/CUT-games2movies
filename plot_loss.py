import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_mean_std(x, y, std=None, label=None, log_scale=False, rolling_avg=False):
    if log_scale:
        plt.yscale("log")
    # plt.plot(x, y, label=label)
    if rolling_avg:
        y_rolling_avg = y.rolling(10).mean()
        plt.plot(x, y_rolling_avg, label=label)
    if std is not None:
        plt.fill_between(x, y-std, y+std, alpha=0.2)

def save_graph(xlabel, ylabel, path, name):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    # plt.savefig(path + name)
    # plt.clf()

f = open("./data/loss_log.txt", "r")
f.readline()
iters = 0
loss_g_total = []
loss_g_nce_y = []
loss_g_nce_x = []
loss_g_gan = []
for line in f:
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
    if len(nums) == 10:
        loss_g_total.append(float(nums[-3]))
        loss_g_nce_y.append(float(nums[-1]))
        loss_g_nce_x.append(float(nums[-2]))
        loss_g_gan.append(float(nums[-6]))
        iters += 100

iters = np.arange(0, iters, 100)
iters = pd.DataFrame(iters)
plot_mean_std(iters, pd.DataFrame(loss_g_total), rolling_avg=True, label="G total")
plot_mean_std(iters, pd.DataFrame(loss_g_nce_x), rolling_avg=True, label="G NCE_X")
plot_mean_std(iters, pd.DataFrame(loss_g_nce_y), rolling_avg=True, label="G NCE_Y")
plot_mean_std(iters, pd.DataFrame(loss_g_gan), rolling_avg=True, label="G GAN")
save_graph("Iterations", "Loss", "", "")
