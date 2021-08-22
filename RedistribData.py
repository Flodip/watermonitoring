import numpy as np
import pandas as pd


# rearranging data to have constant delta t
def redistribute_data(df):
    df['dt'] = df['time'].diff().fillna(0)
    dt = np.mean(df['dt'])  # mean of diff between times

    new_df = pd.DataFrame(columns=['time', 'value'])
    i = 0
    t = df['time'].iloc[i]
    while t < df['time'].iloc[-1]:
        t1 = df['time'].iloc[i + 1]
        # t0 + dt > t1 ?
        while t > t1:
            i = i + 1
            t1 = df['time'].iloc[i + 1]
        t0 = df['time'].iloc[i]
        v0 = df['value'].iloc[i]
        v1 = df['value'].iloc[i + 1]

        v = v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        new_df = new_df.append({'time': t, 'value': v}, ignore_index=True)
        t = t + dt

    return new_df, dt


def redistribute_data(x, y):
    dt = x.diff().fillna(0)
    dt = np.mean(dt)  # mean of diff between times

    new_df = pd.DataFrame(columns=['time', 'value'])
    i = 0
    t = x.iloc[i]
    while t < x.iloc[-1]:
        t1 = x.iloc[i + 1]
        # t0 + dt > t1 ?
        while t > t1:
            i = i + 1
            t1 = x.iloc[i + 1]
        t0 = x.iloc[i]
        v0 = y.iloc[i]
        v1 = y.iloc[i + 1]

        v = v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        new_df = new_df.append({'time': t, 'value': v}, ignore_index=True)
        t = t + dt

    return new_df, dt


def do_fft(df):
    sp = np.fft.fft(df['value'] - np.mean(df['value']))
    freq = np.fft.fftfreq(df['time'].shape[-1])

    sp.real[np.abs(sp.real) < threshold] = 0

    dp = np.fft.ifft(sp)
    plt.scatter(df['time'], dp.real, s=3)

    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()


def do_ifft(df, sp):
    df["smoothed"] = np.fft.irfft(sp)

    plt.plot(df["time"], df["smoothed"])
    plt.show()
    return df


def clean():
    # ADJUST SAMPLES TO GET EQUALLY DISTRIBUTED POINTS OVER TIME
    df = pd.read_csv(output_path)
    print(df.iloc[2]["time"])
    for i in range(0, len(df) - 1):
        print(df.iloc[i + 1]["time"] - df.iloc[i]["time"])
    if not path.isfile(output_path) and False:
        df1 = pd.read_csv(data_path)
        df1 = df1.iloc[10:54500]
        redistribute_data(df1)

    # plot_data(df, df2)
    if path.isfile(output_path) and False:
        df2 = pd.read_csv(output_path)
        do_fft(df2, 54289)
    # plt.show()
