import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from os import path
#import sympy as sym
from scipy import optimize as opt
import csv

file = "../data/data1/data1_0449"
data_path = file + ".csv"
output_path = "../data/data1_extract_on-clean.csv"
sampling_rate = 50
threshold = 500
window = 100


def get_x_y(data=data_path):
	df = pd.read_csv(data)
	df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f').values.astype(np.int64)
	df['time'] = list(map(lambda x: x / 10**9, df['time']))
	#for i in range(0, df['time'].size):
	#	df.loc[i, 'time'] = pd.to_datetime(df.loc[i, 'time'], format='%Y-%m-%d %H:%M:%S.%f').timestamp()
	return df['time'], df['value']


# https://stackoverflow.com/questions/61168646/scipy-optimize-curvefit-calculates-wrong-values
def guess_fft(x, y):
	ff = np.fft.fftfreq(len(x), (x[1] - x[0]))
	Fyy = abs(np.fft.fft(y))
	guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
	guess_amp = np.std(y) * 2. ** 0.5
	guess_offset = np.mean(y)
	guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])
	return guess


def sin_fit(x, amplitude, freq, phase, offset):
	return amplitude * np.sin(x * freq + phase) + offset


def sin_fit_mono(x, v):
	amplitude = v[0]
	freq = v[1]
	phase = v[2]
	offset = v[3]
	return amplitude * np.sin(x * freq + phase) + offset


def guess_simple(x, y):
	guess_amp = 0.05
	guess_freq = 2*np.pi/0.33
	guess_offset = 0.38
	guess = np.array([guess_amp, guess_freq, 0., guess_offset])
	return guess


def fit_sine_curve(x, y, guess_f=guess_fft):
	"""
	1st method to get period from data
	:param x: x param
	:param y: y param
	:param guess_f: function to estimate the points
	:return:
	"""
	x = np.array(x.astype(np.float).values)
	y = np.array(y.astype(np.float).values)

	guess = guess_f(x, y)
	try:
		# params: [amplitude, freq, phase, offset]
		params, params_covariance = opt.curve_fit(f=sin_fit, xdata=x, ydata=y, p0=guess)
		is_on = False
		minv, maxv = min(y), max(y)
		if (3 < params[1] < 100 and abs(minv - maxv) > 0.08) or (5 < params[1] < 100 and 0.03 < abs(params[0]) < 0.08):
			is_on = True
		return is_on, params, params_covariance, minv, maxv
	except Exception as e:
		#print(e)
		return False, guess, [math.inf, math.inf, math.inf, math.inf], min(y), max(y)


def extract_file_seconds(data=data_path):
	import csv
	df = pd.read_csv(data)

	for i in range(0, len(df)//sampling_rate):
		with open(file + "_{:04d}".format(i) + ".csv", 'w') as output:
			writer = csv.writer(output, delimiter=',')
			writer.writerow(['time', 'value'])
			for j in range(0, 50):
				writer.writerow([df.iloc[i*sampling_rate+j]['time'], df.iloc[i*sampling_rate+j]['value']])


def sine_fit_file(filepath, guess_f=guess_fft, window=100, do_write=False):
	x, y = get_x_y(filepath)
	i = 0
	filepath_split = filepath.split("/")
	filepath_out = filepath_split[:-1]
	filepath_out = "/".join(filepath_out) + "_processed/" + "p_" + filepath_split[len(filepath_split)-1]
	res, res_params, res_params_cov, res_min, res_max = [], [], [], [], []
	if do_write:
		with open(filepath_out, 'w') as output:
			writer = csv.writer(output, delimiter=',')
			writer.writerow(['time', 'period'])
			while x.size-i >= window:
				j = i+window
				is_on, params, params_cov, min, max = fit_sine_curve(x[i:j], y[i:j], guess_f)
				res_params, res_params_cov = res_params + [params], res_params_cov + [params_cov]
				res_min, res_max = res_min + [min], res_max + [max]
				it1, it2 = i/50, j/50
				# 0: amplitude, 1: freq
				if is_on:
					res = res + [i]
					writer.writerow([int(it1), 2 * math.pi / params[1]])
				i = j
	else:
		while x.size-i >= window:
			j = i+window
			is_on, params, params_cov, min, max = fit_sine_curve(x[i:j], y[i:j], guess_f)
			res_params, res_params_cov = res_params + [params], res_params_cov + [params_cov]
			res_min, res_max = res_min + [min], res_max + [max]
			it1, it2 = i/50, j/50
			# 0: amplitude, 1: freq
			if is_on:
				res = res + [i]
			i = j

	res = _clean_res(res)
	return res, res_params, res_params_cov, res_min, res_max


def _clean_res(res):
	""" Clean results from sine fit by deleting intervals of 1s, and filling up voids in intervals

	:param res: ON data from watermeter
	:return: cleaned ON data
	"""
	if len(res) == 1:  # just one element
		res.pop(0)
	elif len(res) > 1:
		if res[0] + window * 2 == res[1]:  # missing match between 1st and 2nd element
			res.insert(0, res[0] - window)
		if res[0] + window * 2 < res[1]:  # 1st match alone
			res.pop(0)
			return _clean_res(res)

		res_size = len(res)
		if res_size == 1:  # just one element left
			res.pop(0)
		else:
			i = 1
			while i < res_size - 1:
				if res[i] == res[i-1] + window * 2:  # missing match between 2 matches
					res.insert(i, res[i] - window)
					res_size = res_size + 1
				if res[i] != res[i-1] + window and res[i] != res[i+1] - window:  # match alone
					res.pop(i)
					i = i - 1
					res_size = res_size - 1

				i = i + 1

			if res[i] == res[i-1] + window * 2:  # last match is missing previous element
				res.insert(i, res[i] - window)
			if res[i] > res[i-1] + window * 2:  # last match alone
				res.pop(i)

	return res


def main():
	directory = "../data/dataPool/10-07/"
	directory = "../data/water/"
	l_dir = os.listdir(directory)
	#l_dir = ["1625591640-1625592540.csv"]
	l_dir = ["1625600640-1625601540.csv"]
	for filename in l_dir:
		if filename.endswith(".csv"):
			#print(filename)
			x, y = get_x_y(directory + filename)
			x = x - x[0]
			#print(len(x))
			#plt.plot(x,y)
			#plt.title(filename)
			#mng = plt.get_current_fig_manager()
			#mng.resize(*mng.window.maxsize())
			#plt.show()
			res, res_params, res_params_cov, res_min, res_max = sine_fit_file(directory + filename, window=window)
			#print(len(res))
			#print(len(res_params))
			do_plot = True
			if do_plot:
				x2, y2 = [], []
				x, y = x.tolist(), y.tolist()

				k = 0
				for i in res:
					if k < i:
						x2 = x2 + x[k:i]
						y2 = y2 + [None] * (i-k)
					k = i
					x2 = x2 + x[i:i+window]
					y2 = y2 + y[i:i+window]
				y_p1, y_p2, y_p3, y_p4 = [], [], [], []
				y_min_max = []
				for i in range(0, len(res_params)):
					y_p1 = y_p1 + [abs(res_params[i][0])]*window  # amplitude
					y_p2 = y_p2 + [res_params[i][1]]*window
					y_p3 = y_p3 + [res_params[i][2]]*window
					y_p4 = y_p4 + [res_params[i][3]] * window
					y_min_max = y_min_max + [abs(res_max[i]-res_min[i])] * window
				x_params = x[:len(y_p1)]
				rows = 6
				plt.figure(figsize=(6, 6))
				ax0 = plt.subplot(rows, 1, 1)
				#ax0.set_title(filename)
				ax0.plot(x,y)
				ax0.set_ylabel(r"magnetic field ($\mu$t)")

				ax1 = plt.subplot(rows, 1, 2)
				ax1.plot(x, y)
				ax1.plot(x2, y2)
				ax1.legend(["OFF", "ON"])
				ax1.set_ylabel(r"magnetic field ($\mu$t)")

				ax2 = plt.subplot(rows, 1, 3, sharex=ax1)
				ax2.plot(x_params, y_p1)
				ax2.set_ylabel("abs(amplitude)")

				ax3 = plt.subplot(rows, 1, 4, sharex=ax1)
				ax3.plot(x_params, y_p2)
				ax3.set_ylabel("freq (Hz)")

				ax5 = plt.subplot(rows, 1, 5, sharex=ax1)
				ax5.plot(x_params, y_p4)
				ax5.set_ylabel("offset")

				ax8 = plt.subplot(rows, 1, 6, sharex=ax1)
				ax8.plot(x_params, y_min_max)
				ax8.set_ylabel("abs(min-max)")
				ax8.set_xlabel("time (s)")

				plt.show()

	#sine_fit_curve(get_x_y("../data/data1/data1_0058.csv"))
	#fit_sine_curve(data="../patternMatching/data/1625561040-1625561940.csv")
	# minimize_sine(data="../data/data1/data1_0058.csv")


if __name__ == "__main__":
	num_runs = 1
	import timeit
	print(str(timeit.timeit("main()", setup="from __main__ import main", number=num_runs)/num_runs))


# print(df.head())

# plt.plot(df['value'])

# plt.plot(df['dt'])
# df['dt'] = df['time'].diff().shift(-1).fillna(0)

# DT = np.mean(df['dt'])
# print(DT)
# plt.plot(df['dt'])