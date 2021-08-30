import matplotlib
import numpy as np
import sys
sys.path.append("../recording")
sys.path.append("../")
from SineFit import guess_fft, sin_fit
from math import sqrt

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize as opt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from dtw import *
from data_dict import data_dict

second_size = 50

def get_data(filename):
    df = pd.read_csv(filename)
    #for i in range(0, df['time'].size):
    #    df.loc[i, 'time'] = pd.to_datetime(df.loc[i, 'time'], format='%Y-%m-%d %H:%M:%S.%f').timestamp()
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f').values.astype(float)
    return df['time'], df['value']


def fft_filter_func(a):
    T = 1.0 / 50.0
    N = len(a)

    f = np.linspace(0, 1.0/(2.0*T), int(N/2))
    res = np.fft.rfft(a - np.mean(a))

    my_filter = np.bitwise_or(f < 0.1, f > 25)
    power_filter = np.abs(res[:int(N/2)]) < 3
    ind = np.where(my_filter)
    res[ind] = 0
    ind2 = np.where(power_filter)
    res[ind2] = 0

    ires = np.fft.irfft(res)
    return ires


def fit_sine_curve_freq(x, y, guess_f=guess_fft):
    """
    1st method to get period from data
    :param data: path of file to read data from
    :param guess_f: function to estimate the points
    :return:
    """

    x = np.array(x.astype(float).values)
    y = np.array(y.astype(float).values)

    x0 = x[0]
    for i in range(0, x.size):
        x[i] -= x0

    guess = guess_f(x, y)

    try:
        params, params_covariance = opt.curve_fit(f=sin_fit, xdata=x, ydata=y, p0=guess)
        amplitude = params[0]
        freq = params[1]
        phase = params[2]
        offset = params[3]

        return freq

    except:
        return -1


def fit_sine_curve_knn(x, y, freqs=[1, 10, 15, 25]):
    n_features = len(freqs) - 1
    res = np.zeros(n_features)
    for i in range(0,len(x) - second_size, second_size):
        f = fit_sine_curve_freq(x[i:i+second_size], y[i:i+second_size]) * 1e9
        if f > 0:
            for i in range(n_features):
                if f > freqs[i] and f < freqs[i+1]:
                    res[i] = res[i] + 1

    res = res / (len(x) / second_size)

    return res





"""
Return a list of N tuples corresponding to (frequency - amplitude) for
frequencies between 1 and 25Hz
x is considered to be at 50Hz
"""
def fft_knn(x, y, freqs=[1, 10, 15, 25]):

    T = 1.0 / 50.0
    N = len(y)

    f = np.linspace(0, 1.0/(2.0*T), int(N/2))
    res_fft = np.fft.rfft(y - np.mean(y))

    highpass_filter = f < 0.9
    ind = np.where(highpass_filter)
    res_fft[ind] = 0

    n_features = len(freqs) - 1
    res = np.zeros(n_features)
    for idx in range(n_features):
        filt = np.bitwise_and(f > freqs[i], f < freqs[i+1])
        ind = np.where(filt)

        filtered_values = res_fft[ind]
        res[idx] = np.mean(filtered_values)

    return res

def knn_view(knn_func=fft_knn, freqs=[0, 8.33, 16.66, 25]):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for entry in data_dict:
        if len(entry["wc"]) == 0 and len(entry["tap water"]) == 0:
            continue
        x_signal, signal = get_data('../data/' + entry['filename'])
        for wc_entry in entry["wc"]:
            x_wc, wc = x_signal[int(wc_entry["start"]*50):int(wc_entry["end"]*50)], signal[int(wc_entry["start"]*50):int(wc_entry["end"]*50)]
            xyz = knn_func(x_wc, wc, freqs=freqs)
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker='o', color='r')
        for tap_entry in entry["tap water"]:
            x_tap, tap = x_signal[int(tap_entry["start"]*50):int(tap_entry["end"]*50)], signal[int(tap_entry["start"]*50):int(tap_entry["end"]*50)]
            xyz = knn_func(x_tap, tap, freqs=freqs)
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker='^', color='g')
    ax.set_xlabel('f1 mean amplitude')
    ax.set_ylabel('f2 mean amplitude')
    ax.set_zlabel('f3 mean amplitude')
    plt.show()


def read_data_dict(folderPath, knn_func, freqs, data_dict):
    x, y = [], []
    for data in data_dict:
        entry = data["data"]
        clas = data["class"]
        x_signal, signal = get_data(folderPath + data['filename'])

        tmp_x, tmp_y = x_signal[int(entry["start"]*50):int(entry["end"]*50)], signal[int(entry["start"]*50):int(entry["end"]*50)]
        xyz = knn_func(tmp_x, tmp_y, freqs)

        x.append(xyz)
        y.append(clas)

    return x, y

def parse_dict(data_dict):
    entries = []
    for entry in data_dict:
        for wc_entry in entry["wc"]:
            entries.append({"data": wc_entry,"class": "wc","filename":entry["filename"]})
        for tap_entry in entry["tap water"]:
            entries.append({"data": tap_entry,"class": "tap","filename":entry["filename"]})
    return entries

def knn_getdata(knn_func=fft_knn, freqs=[0,8.33, 16.66, 25], test_size=0.5):
    entries = parse_dict(data_dict)
    dict_train, dict_test = train_test_split(entries, test_size=test_size, random_state=2)
    X_train, y_train = read_data_dict('../data/dataFull/', knn_func, freqs, dict_train)
    X_test, y_test = read_data_dict('../data/dataFull/', knn_func, freqs, dict_test)

    return X_train, y_train, X_test, y_test

def evaluate_knn(neigh, X_test, y_test):
    wc_ok, wc_nok, tap_ok, tap_nok = 0, 0, 0, 0
    for X,y in zip(X_test, y_test):
        pred = neigh.predict([X])
        if (str(pred[0]) == str(y)):
            if y == 'wc':
                wc_ok = wc_ok + 1
            elif y == 'tap':
                tap_ok = tap_ok + 1
        else:
            if y == 'wc':
                wc_nok = wc_nok + 1
            elif y == 'tap':
                tap_nok = tap_nok + 1

    print("wc correct: " + str(wc_ok) + " / (" + str(wc_ok) + " + " + str(wc_nok) + ") = " + str(wc_ok / (wc_ok + wc_nok)))
    print("tap correct: " + str(tap_ok) + " / (" + str(tap_ok) + " + " + str(tap_nok) + ") = " + str(tap_ok / (tap_ok + tap_nok)))

    return (wc_ok + tap_ok) / (wc_ok + tap_ok + wc_nok + tap_nok)


def full_knn(pre_process=fit_sine_curve_knn, metric=None, freqs=[0,8.33,16.66,25], test_size=0.5, K=5):
    X_train, y_train, X_test, y_test = knn_getdata(pre_process, freqs=freqs, test_size=test_size)
    if metric is None:
        metric = 'minkowski'
    neigh = KNeighborsClassifier(n_neighbors=K, metric=metric)
    neigh.fit(X_train, y_train)

    return evaluate_knn(neigh, X_test, y_test)

def my_euclidean(X, Y):
    total = 0
    for x,y in zip(X, Y):
        total += (x-y)**2
    return sqrt(total)

def my_dtw(X, Y):
    al = dtw(X, Y)
    return al.distance

def tune_parameters():
    n_dim = [20,25,50]
    K = [1,3,5]
    metrics = [my_euclidean, my_dtw]

    max_success_rate = 0
    best_K = 0
    best_dim = 0
    best_metric = "euclidean"
    for dim in n_dim:
        for k in K:
            for metric in metrics:
                success_rate = full_knn(metric=metric, freqs=np.linspace(0, 25, dim), test_size=0.3, K=k)
                if success_rate > max_success_rate:
                    max_success_rate = success_rate
                    best_K = k
                    best_dim = dim
                    if metric == my_euclidean:
                        best_metric = "euclidean"
                    else:
                        best_metric = "dtw"
    retval = {
        "K": best_K,
        "dim": dim,
        "success_rate": max_success_rate,
        "metric": best_metric,
    }
    return retval


if __name__ == '__main__':
    params = tune_parameters()
    print(params)

