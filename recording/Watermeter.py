import matplotlib
import os

from recording.CleanData import get_x_y
from recording.SaveData import get_data_as_csv

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # sd.save_data()

    # get_data_as_csv(1625961600000, 1625965200000, interval=900)
    directory_path = "../data/dataPool/09-07/"
    directory = os.listdir(directory_path)
    directory.sort()
    for filename in directory:
        if filename.endswith(".csv"):
            x, y = get_x_y(directory_path + filename)
            x = x - x[0]
            print(filename)
            plt.plot(x, y)
            plt.show()
