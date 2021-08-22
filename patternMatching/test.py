import matplotlib
matplotlib.use('tkAgg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_data(filename):
    print("Gettings data")
    df = pd.read_csv(filename)
    #for i in range(0, df['time'].size):
    #    df.loc[i, 'time'] = pd.to_datetime(df.loc[i, 'time'], format='%Y-%m-%d %H:%M:%S.%f').timestamp()
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f').values.astype(float)
    return df['time'], df['value']

def myfft(y):
    T = 1.0 / 50.0
    N = len(y)
    f = np.linspace(0, 1.0/(2.0*T), int(N/2))

    res = np.fft.rfft(y - np.mean(y))

    return f, np.abs(res[:int(N/2)])

fig = plt.figure()
axis = plt.axes(xlim=(0, 25),
                ylim=(0, 1))
line, = axis.plot([], [], lw=3)

t, signal = get_data("data2.csv")
myfft(signal.iloc[0:50])

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x, y = myfft(signal.iloc[i*50:(i+1)*50])
    line.set_data(x,y)

    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
        frames=200, interval=20, blit=True)

anim.save('test.mp4', writer = 'ffmpeg', fps = 30)
