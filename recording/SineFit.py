import math
import os

import numpy as np
import pandas as pd
import matplotlib

import config
from Utils import get_x_y

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from os import path
# import sympy as sym
from scipy import optimize as opt
import csv

file = "../data/data1/data1_0449"
data_path = file + ".csv"
output_path = "../data/data1_extract_on-clean.csv"
sampling_rate = 50
threshold = 500
window = 100


def guess_fft(x, y):
	"""
	https://stackoverflow.com/questions/61168646/scipy-optimize-curvefit-calculates-wrong-values
	First guess parameters of curve fit using fft
	:param x: time
	:param y: signal value
	:return: array containing guesses for [amplitude frequency phase offset]
	"""
	ff = np.fft.fftfreq(len(x), (x[1] - x[0]))
	Fyy = abs(np.fft.fft(y))
	guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
	guess_amp = np.std(y) * 2. ** 0.5
	guess_offset = np.mean(y)
	guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])
	return guess


def sin_fit(x, amplitude, freq, phase, offset):
	return amplitude * np.sin(x * freq + phase) + offset


def do_sine_fit(x, y, guess_f=guess_fft):
	"""
	Estimates the parameters of a sine function with the curve fit method from scipy.opt.curve_fit
	:param x: x axis data
	:param y: y axisdata
	:param guess_f: function to estimate the starting values of the parameters of the fitting function
	:return: boolean True if water is deemed used or false otherwise, params of sin_fit, covariance of params
	"""
	x = np.array(x.astype(np.float).values)
	y = np.array(y.astype(np.float).values)

	guess = guess_f(x, y)
	ON = True
	try:
		# params: [amplitude, freq, phase, offset]
		params, params_covariance = opt.curve_fit(f=sin_fit, xdata=x, ydata=y, p0=guess)
		cfg = config.sine_fit
		if cfg['min_freq'] * 2 * np.pi < params[1] < cfg['max_freq'] * 2 * np.pi \
				and (abs(min(y) - max(y)) > cfg['min_diff'] or cfg['min_amplitude'] < abs(params[0]) < cfg['max_amplitude']):
			return ON, params, params_covariance
		else:
			return not ON, params, params_covariance
	except Exception as e:
		print(e)
		return not ON, guess, [math.inf, math.inf, math.inf, math.inf]


def sine_fit(x, y, guess_f=guess_fft):
	"""
	Calls sine fit over large data to differentiate usable data from noise
	:param x: x axis data
	:param y: y axis data
	:param guess_f: function to estimate the starting values of the parameters of the fitting function
	:return: positions in x where ON, params, params_cov, min, max from sine fit for each 2s interval of x,y
	"""
	i = 0
	res, res_params, res_params_cov, res_min, res_max = [], [], [], [], []
	while x.size - i >= window:
		j = i + window
		is_on, params, params_cov = do_sine_fit(x[i:j], y[i:j], guess_f)
		res_params, res_params_cov = res_params + [params], res_params_cov + [params_cov]
		res_min, res_max = res_min + [min(y[i:j])], res_max + [max(y[i:j])]
		if is_on:
			res = res + [i]
		i = j

	res = _clean_res(res)
	return res, res_params, res_params_cov, res_min, res_max


def _clean_res(res):
	"""
	Cleans results from sine fit by deleting intervals of 1s, and filling up voids in intervals
	1s interval use of water is extremely rare, and it is often noise wrongly detected as ON
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
				if res[i] == res[i - 1] + window * 2:  # missing match between 2 matches
					res.insert(i, res[i] - window)
					res_size = res_size + 1
				if res[i] != res[i - 1] + window and res[i] != res[i + 1] - window:  # match alone
					res.pop(i)
					i = i - 1
					res_size = res_size - 1

				i = i + 1

			if res[i] == res[i - 1] + window * 2:  # last match is missing previous element
				res.insert(i, res[i] - window)
			if res[i] > res[i - 1] + window * 2:  # last match alone
				res.pop(i)

	return res


def main1():
	"""
	Short example to show the use of do_sine_fit
	"""
	x, y = get_x_y("../data/data1.csv")
	x = x - x[0]

	plt.plot(x, y)
	plt.show()
	start, end = 228, 230
	start, end = start * 50, end * 50
	x, y = x[start:end], y[start:end]
	is_on, params, params_cov = do_sine_fit(x, y)
	print(params)
	plt.plot(x, y)
	plt.plot(x, sin_fit(x, params[0], params[1], params[2], params[3]))

	x, y = np.array(x.astype(np.float).values), np.array(y.astype(np.float).values)
	guess = guess_fft(x, y)
	print(guess)
	print(params)
	plt.plot(x, sin_fit(x, guess[0], guess[1], guess[2], guess[3]))
	plt.legend(['signal wave', 'curve fit', 'fft'])
	plt.show()

def main2():
	"""
	Example to show the use of sine_fit, and displaying results
	"""
	directory = "../data/water/"
	l_dir = ["1625600640-1625601540.csv"]
	for filename in l_dir:
		if filename.endswith(".csv"):
			x, y = get_x_y(directory + filename)
			x = x - x[0]
			res, res_params, res_params_cov, res_min, res_max = sine_fit(x, y)
			do_plot = True
			if do_plot:
				x2, y2 = [], []
				x, y = x.tolist(), y.tolist()

				k = 0
				for i in res:
					if k < i:
						x2 = x2 + x[k:i]
						y2 = y2 + [None] * (i - k)
					k = i
					x2 = x2 + x[i:i + window]
					y2 = y2 + y[i:i + window]
				y_p1, y_p2, y_p3, y_p4 = [], [], [], []
				y_min_max = []
				for i in range(0, len(res_params)):
					y_p1 = y_p1 + [abs(res_params[i][0])] * window  # amplitude
					y_p2 = y_p2 + [res_params[i][1]] * window
					y_p3 = y_p3 + [res_params[i][2]] * window
					y_p4 = y_p4 + [res_params[i][3]] * window
					y_min_max = y_min_max + [abs(res_max[i] - res_min[i])] * window
				x_params = x[:len(y_p1)]
				rows = 6
				plt.figure(figsize=(6, 6))
				ax0 = plt.subplot(rows, 1, 1)
				# ax0.set_title(filename)
				ax0.plot(x, y)
				ax0.set_ylabel(r"magnetic field ($\mu$t)")

				ax1 = plt.subplot(rows, 1, 2, sharex=ax0)
				ax1.plot(x, y)
				ax1.plot(x2, y2)
				ax1.legend(["OFF", "ON"])
				ax1.set_ylabel(r"magnetic field ($\mu$t)")

				ax2 = plt.subplot(rows, 1, 3, sharex=ax0)
				ax2.plot(x_params, y_p1)
				ax2.set_ylabel("abs(amplitude)")

				ax3 = plt.subplot(rows, 1, 4, sharex=ax0)
				ax3.plot(x_params, y_p2)
				ax3.set_ylabel("freq (Hz)")

				ax5 = plt.subplot(rows, 1, 5, sharex=ax0)
				ax5.plot(x_params, y_p4)
				ax5.set_ylabel("offset")

				ax8 = plt.subplot(rows, 1, 6, sharex=ax0)
				ax8.plot(x_params, y_min_max)
				ax8.set_ylabel("abs(min-max)")
				ax8.set_xlabel("time (s)")

				plt.show()


if __name__ == "__main__":
	main2()
