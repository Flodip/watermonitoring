from statistics import mean, pstdev

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# search window
t0_min = 3
t0_max = 30
t_min = t0_min
t_max = t0_max

n = t_max*2

# 28s pour data1


def __get_x_y(data):
    df = pd.read_csv(data)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f').values.astype(np.int64)
    df['time'] = list(map(lambda x: x / 10**9, df['time']))
    return df['time'].values, df['value']


def __reset_t():
    global t_min, t_max, n
    t_min = t0_min
    t_max = t0_max
    n = t_max*2


def autocorrelation(data, threshold=0.55, std_thresh=0.02, window_size=50):
    i = 0
    results = []
    stds = []
    std1s = []
    std2s = []
    max_norm_auto_corrs = []
    while True:
        try:
            a = data.values[i:i+n]
            #print("len a " + str(len(a)))
            max_norm_auto_corr, t, std, std1, std2 = __v(a)
            __update_t(t, window_size=window_size)
            #print("update: topt " + str(t) + " tmin " + str(t_min) + " tmax " + str(t_max))
            stds.append(std)
            std1s.append(std1)
            std2s.append(std2)
            max_norm_auto_corrs.append(max_norm_auto_corr)
            if max_norm_auto_corr > threshold and std > std_thresh:
                results.append(i)
                print(str(i//50) + " t_opt: " + str(t) + " t_max: " + str(t_max) +  ", a: " + str(len(a)) + " max. norm. auto corr.: " + str(max_norm_auto_corr), " std: " + str(std))
            else:
                __reset_t()
            i += 25
        # end of file
        except Exception as e:
            print(e)
            break
    return results, max_norm_auto_corrs, stds, std1s, std2s


def __update_t(t_opt, window_size):
    """ Shift search window around the t_opt

    :param t_opt: optimal lag
    :param window_size: size of the search window
    :return:
    """
    global t_min
    t_min = max(t0_min, round(t_opt - window_size//2))
    global t_max
    t_max = round(t_opt + window_size//2)
    global n
    n = t_max*2


def __x(a, m, t):
    """Normalized Auto-correlation

    :param a: data array
    :param m: starting sample
    :param t: lag
    :return: Normalized Auto-correlation, standard deviation
    """

    sum = 0
    # Sum(k=0, k=lag t-) of data
    for k in range(0, t-1):
        sum += (a[m+k] - mean(a[m:m+t-1])) * (a[m+k+t] - mean(a[m+t:m+t+t-1]))

    std1 = pstdev(a[m:m+t-1])
    std2 = pstdev(a[m+t:m+t+t-1])
    std = t * std1 * std2
    return sum/std, std, std1, std2


def __v(a, m=0):
    """Maximized Normalized Auto-correlation

    :param a: data array
    :param m: starting sample
    :return: Max Normalized Auto-correlation, corresponding lag, standard deviation
    """

    max_x = float("-inf")
    std = 0
    std1 = 0
    std2 = 0
    max_t = 0

    # looking for lag maximizing auto-correlation
    for t in range(t_min, t_max):
        x_tmp, std, std1, std2 = __x(a, m, t)
        t_tmp = t

        if x_tmp > max_x:
            max_x = x_tmp
            max_t = t_tmp

    return max_x, max_t, std, std1, std2


def main():
    time1, data1 = __get_x_y("../data/water/1625600640-1625601540.csv")
    time1 = time1 - time1[0]
    #time1, data1 = time1[:15000], data1[:15000]

    plt.plot(time1, data1)
    plt.xlabel("time (s)")
    plt.ylabel("magnetic field (ut)")
    plt.show()

    autocorr_res, max_autocorr_res, std_res, std1_res, std2_res = autocorrelation(data1)

    time1, data1 = time1.tolist(), data1.tolist()
    time2, data2 = [], []
    for i in autocorr_res:
        time2 = time2 + time1[i:i+25]
        data2 = data2 + data1[i:i+25]

    data3, data4, data5, data6 = [], [], [], []

    for i in range(0, len(max_autocorr_res)):
        data3 = data3 + [max_autocorr_res[i]]*25
        data4 = data4 + [std_res[i]]*25
        data5 = data5 + [std1_res[i]]*25
        data6 = data6 + [std2_res[i]]*25

    time34 = time1[:len(data3)]

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(5, 1, 1)
    ax2 = plt.subplot(5, 1, 2)
    ax3 = plt.subplot(5, 1, 3)
    ax4 = plt.subplot(5, 1, 4)
    ax5 = plt.subplot(5, 1, 5)

    ax1.plot(time1, data1)
    ax1.plot(time2, data2)
    ax1.legend(["OFF", "ON"])
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(r"magnetic field ($\mu$t)")

    ax2.plot(time34, data3)
    ax2.legend(["maximum normalized auto-correlation"])
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(r"\chi")

    ax3.plot(time34, data4)
    ax3.legend([r"$\tau\sigma(m, \tau)\sigma(m + \sigma, \sigma)$"])
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel(r"$\tau\sigma(m, \tau)\sigma(m + \sigma, \sigma)$")

    ax4.plot(time34, data5)
    ax4.legend([r"$\sigma(m, \tau)$"])
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel(r"$\sigma(m, \tau)$")

    ax5.plot(time34, data6)
    ax5.legend([r"$\sigma(m + \sigma, \sigma)$"])
    ax5.set_xlabel("time (s)")
    ax5.set_ylabel(r"$\sigma(m + \sigma, \sigma)$")

    plt.show()


if __name__ == "__main__":
    num_runs = 1
    import timeit
    print(str(timeit.timeit("main()", setup="from __main__ import main", number=num_runs)/num_runs))
    #plt.figure(figsize=(6,6))
    #ax1 = plt.subplot(1, 2, 1)
    #x2 = plt.subplot(1, 2, 2)
    #ax1.plot([1,2,3], [1,2,3])
    #ax1.scatter([4,5,6], [4,5,6])
    #ax1.legend(["test", r"$\tau$"])
    #ax1.set_xlabel("test")
    #ax1.set_ylabel(r"$\mu$test")
    #plt.show()

