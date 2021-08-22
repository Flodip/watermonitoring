import json
import numpy as np
from dtw import *
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("../")
from data_dict import data_dict


def get_data(filename):
    df = pd.read_csv(filename)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')

    return df['time'].to_numpy(), df['value'].to_numpy()


def get_hits(x_signal, signal, x_template, template):

    len_signal = signal.shape[0]
    len_template = template.shape[0]

    x = []
    dist = []
    for i in range(0, len_signal-len_template, int(len(template)/3)):
        print(i / (len_signal - len_template))
        al = dtw(signal[i:i+len_template], template)
        di = al.distance
        xi = x_signal[i+len_template]

        dist.append(di)
        x.append(xi)

    return np.array(x), np.array(dist)

def get_similarity(signal, template):
    if len(template) > len(signal):
        return float('inf')
    al = dtw(signal[-len(template):], template)
    return al.normalizedDistance


def default_filter_func(a):
    return a.to_numpy()


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


def analysis(x_wc, wc, x_tap, tap, filter_func=default_filter_func):
    print("Performing DTW analysis")
    wc = filter_func(wc)
    x_wc = x_wc[:len(wc)]
    tap = filter_func(tap)
    x_tap = x_tap[:len(tap)]

    for entry in data_dict:
        if len(entry['wc']) == 0 and len(entry['tap water']) == 0:
            continue
        x_signal, signal = get_data('../data/dataFull/' + entry['filename'])
        signal = filter_func(signal)
        x_signal = x_signal[:len(signal)]
        x1, dist_wc = get_hits(x_signal, signal, x_wc, wc)
        x2, dist_tap = get_hits(x_signal, signal, x_tap, tap)

        fig, axs = plt.subplots(3)
        axs[0].plot(x_signal, signal, color='b')
        for wc_entry in entry['wc']:
            x_wc_entry, y_wc_entry = x_signal[wc_entry["start"]*50:wc_entry["end"]*50], signal[wc_entry["start"]*50:wc_entry["end"]*50]
            axs[0].plot(x_wc_entry, y_wc_entry, color='r', label='wc entry')
        for tap_entry in entry['tap water']:
            x_tap_entry, y_tap_entry = x_signal[tap_entry["start"]*50:tap_entry["end"]*50], signal[tap_entry["start"]*50:tap_entry["end"]*50]
            axs[0].plot(x_tap_entry, y_tap_entry, color='g', label='tap water entry')

        dist_wc = dist_wc / len(wc)
        dist_tap = dist_tap / len(tap)
        my_max = max(max(dist_wc), max(dist_tap))
        axs[1].plot(x1, dist_wc, color='r', label='normalized dist to wc')
        axs[1].plot(x2, dist_tap, color='g', label='normalized dist to tap water')
        axs[1].set_ylim(0, my_max)
        #fig.legend()

        ind = np.where(dist_tap < 0.017)
        is_tap_signal = np.zeros(len(dist_tap))
        is_tap_signal[ind] = 1
        axs[2].plot(x2, is_tap_signal)
        fig.savefig(entry['filename'] + ".png")


def fft_analysis(x_signal, signal):
    print("Performing FFT analysis")

    T = 1.0 / 50.0
    N = len(signal)

    f = np.linspace(0, 1.0/(2.0*T), int(N/2))
    res = np.fft.rfft(signal - np.mean(signal))

    fig_noisy, axs_noisy = plt.subplots(2)
    axs_noisy[0].plot(x_signal, signal)
    axs_noisy[1].plot(f, 2.0 / N * np.abs(res[:int(N/2)]))

    my_filter = np.bitwise_or(f < 0.1, f > 15)
    power_filter = np.abs(res[:int(N/2)]) < 10
    ind = np.where(my_filter)
    res[ind] = 0
    ind2 = np.where(power_filter)
    res[ind2] = 0

    ires = np.fft.irfft(res)

    fig_filt, axs_filt = plt.subplots(2)
    axs_filt[0].plot(x_signal, ires)
    axs_filt[1].plot(f, 2.0 / N * np.abs(res[:int(N/2)]))

    plt.legend()

    plt.show()

def old_dtw():
    x_source_wc, source_wc = get_data('data2.csv')
    wc_start, wc_end = 492, 573
    x_wc, wc = x_source_wc.iloc[wc_start*50:wc_end*50], source_wc.iloc[wc_start*50:wc_end*50]

    x_source_tap, source_tap = get_data('data3.csv')
    tap_start, tap_end = 216, 236
    x_tap, tap = x_source_tap.iloc[tap_start*50:tap_end*50], source_tap.iloc[tap_start*50:tap_end*50]

    analysis(x_wc, wc, x_tap, tap)

def show_dtw():
    sub_len = 20
    sub_signal = np.zeros(sub_len)
    for i in range(sub_len):
        sub_signal[i] = 1/10 * 2.73**(-(i-(sub_len/2))**2 / 10)
    x_source_tap, source_tap = get_data('data3.csv')
    my_len = 100
    template_start, template_end = 216, (216 + my_len)
    tap_start, tap_end = 500, (500+my_len)

    template = source_tap[template_start:template_end]
    tap = source_tap[tap_start:tap_end]

    template[10:10+sub_len] = template[10:10+sub_len] + sub_signal
    template[40:40+sub_len] = template[40:40+sub_len] + sub_signal
    tap[30:30+sub_len] = tap[30:30+sub_len] + sub_signal
    tap[60:60+sub_len] = tap[60:60+sub_len] + sub_signal

    alignment = dtw(tap, template, keep_internals=True)
    #alignment.plot(type="twoway", offset=-1)

    alignment.plot(type="twoway", offset=-1, xlab="Sample index", ylab=r"magnetic intensity ($\mu$T)")
    print(np.linalg.norm(tap-template))
    print(alignment.normalizedDistance)

    #plt.plot(x_source_tap, source_tap, color='b')
    #plt.plot(x_source_tap[template_start:template_end], source_tap[template_start:template_end], color='r')
    #plt.plot(x_source_tap[tap_start:tap_end], source_tap[tap_start:tap_end], color='r')
    #plt.show()
    #x_tap, tap = x_source_tap.iloc[tap_start*50:tap_end*50], source_tap.iloc[tap_start*50:tap_end*50]

def get_templates():
    folder = '../data/dataFull/'
    data_file_list = ['1625807700000000000-1625808600000000000.csv', '1625631240-1625632140.csv', '1625770800000000000-1625771700000000000.csv']
    taps, wcs = [], []
    for data_file in data_file_list:
        _, source_data = get_data(folder + data_file)
        for dd in data_dict:
            if dd['filename'] == data_file:
                for tap_water in dd['tap water']:
                    taps.append(source_data[int(tap_water['start']*50):int(tap_water['end']*50)])
                for wc in dd['wc']:
                    wcs.append(source_data[int(wc['start']*50):int(wc['end']*50)])
    res = []
    for tap in taps:
        res.append({
            "class": "tap water",
            "template": tap,
        })
    for wc in wcs:
        res.append({
            "class": "wc",
            "template": wc,
        })

    return res


def create_similarity_measures(templates):
    similarities = []
    i = 0
    for data in data_dict:
        print(str(i) + " / " + str(len(data_dict)))
        i += 1
        filepath = "../data/dataFull/" + data["filename"]
        _, signal = get_data(filepath)
        for entry_class in ["wc", "tap water"]:
            for entry in data[entry_class]:
                min_dists = {"wc": float('inf'), "tap water": float('inf')}
                for template in templates:
                    dist = get_similarity(signal[:int(entry["end"]*50)], template["template"])
                    if template["class"] == "wc" and dist < min_dists["wc"]:
                        min_dists["wc"] = dist
                    elif template["class"] == "tap water" and dist < min_dists["tap water"]:
                        min_dists["tap water"] = dist
                similarities.append({
                    "filepath": filepath,
                    "min_dists": min_dists,
                    "class": entry_class,
                })
    with open("simil.txt", "w+") as f:
        for sim in similarities:
            f.write(str(sim) + "\n")

def parse_similarity_measures(filepath="simil.txt"):
    similarities = []
    with open(filepath, "r") as f:
        for line in f.read().splitlines():
            s = json.loads(line)
            similarities.append(s)
    return similarities

min_diff = 1000
max_diff = -1000
def predict(similarity, thresh = 0):
    global min_diff
    global max_diff
    diff = similarity["min_dists"]["wc"] - similarity["min_dists"]["tap water"]
    if diff < min_diff:
        min_diff = diff
    if  diff > max_diff and diff < 10:
        max_diff = diff

    if diff > thresh:
        return "wc"
    else:
        return "tap water"


def test_similarity_measures(similarities, thresh=0):
    wc_ok, wc_nok, tap_ok, tap_nok = 0, 0, 0, 0
    for simil in similarities:
        prediction = predict(simil, thresh)
        if prediction == "wc" and simil["class"] == "wc":
            wc_ok += 1
        elif prediction == "tap water" and simil["class"] == "wc":
            wc_nok += 1
        elif prediction == "tap water" and simil["class"] == "tap water":
            tap_ok += 1
        elif prediction == "wc" and simil["class"] == "tap water":
            tap_nok += 1
        else:
            print("ERRROR")
    res = {
        "wc_ok": wc_ok,
        "wc_nok": wc_nok,
        "tap_ok": tap_ok,
        "tap_nok": tap_nok,
        "accuracy": (wc_ok + tap_ok) / (wc_ok + tap_ok + wc_nok + tap_nok),
    }
    return res

def test_templates(templates):
    wc_ok, wc_nok, tap_ok, tap_nok = 0, 0, 0, 0
    i = 0
    for data in data_dict:
        print(str(i) + " / " + str(len(data_dict)))
        i += 1
        _, signal = get_data("../data/dataFull/" + data["filename"])
        for entry_class in ["wc", "tap water"]:
            for entry in data[entry_class]:
                min_dist = float('inf')
                best_class = ""
                for template in templates:
                    dist = get_similarity(signal[:int(entry["end"]*50)], template["template"])
                    if dist < min_dist:
                        min_dist = dist
                        best_class = template["class"]
                if entry_class == "wc" and entry_class == best_class:
                    wc_ok += 1
                elif entry_class == "wc" and entry_class != best_class:
                    wc_nok += 1
                elif entry_class == "tap water" and entry_class == best_class:
                    tap_ok += 1
                else:
                    tap_nok += 1
    print("wc correct: " + str(wc_ok) + " / (" + str(wc_ok) + " + " + str(wc_nok) + ") = " + str(wc_ok / (wc_ok + wc_nok)))
    print("tap correct: " + str(tap_ok) + " / (" + str(tap_ok) + " + " + str(tap_nok) + ") = " + str(tap_ok / (tap_ok + tap_nok)))

    return (wc_ok + tap_ok) / (wc_ok + tap_ok + wc_nok + tap_nok)


def dtw_classifier(similarities):
    best = None
    best_thresh = 0
    for thresh in np.linspace(-0.00863405061896113*10,0.010766850570994357*10, 10):
        acc = test_similarity_measures(similarities, thresh = thresh)
        if best is None or acc["accuracy"] > best["accuracy"]:
            best = acc
            best_thresh = thresh
    print(best)
    print("Best thresh: " + str(thresh))

def clean_simil():
    with open("simil.txt", "rt") as file:
        data = file.read()
        data = data.replace("'", "\"")
        data = data.replace("inf", str(10000))
    with open("simil.txt", "wt") as file:
        file.write(data)


if __name__ == '__main__':
    print("Performing DTW test")
    #show_dtw()
    generate_similarities = False
    if generate_similarities:
        templates = get_templates()
        create_similarity_measures(templates)
        clean_simil()
    else:
        similarities = parse_similarity_measures()
        dtw_classifier(similarities)
