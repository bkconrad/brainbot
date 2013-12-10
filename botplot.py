#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time

# calculates the running mean of x using N samples
def running_mean_fast(x, N):
	return np.convolve(x, np.ones(N)/N)[N-1:]

# argument parsing
parser = argparse.ArgumentParser(description='Visualize BrainBot data')
parser.add_argument('--file', '-f', help="the output file to operate on")
parser.add_argument('--no-update', '-n', action='store_true', help="do not update or redraw the graph")

args = parser.parse_args()

# some globals to hold our data
x = []
y = []
i = 0

# open our record file
filename = args.file or '/home/kaen/code/bitfighter-hg/exe/screenshots/record'
with open(filename, 'r+', 1) as record_file:

	# read initial data
	for line in record_file:
		y.append(float(line))
		x.append(i)
		i += 1

	# set up our graphs
	graph1 = plt.plot(x, y, 'k.', markersize=1)[0]
	graph2 = plt.plot(x[:-100], running_mean_fast(y, 100)[:-100],    'b', linewidth=2)[0]
	graph3 = plt.plot(x[:-1000], running_mean_fast(y, 1000)[:-1000], 'r', linewidth=2)[0]

	# when --no-update is specified we just display and exit
	if args.no_update:
		plt.show()
		exit()

	# turn on 'interactive' mode and loop until exit
	plt.ion()
	while True:

		# seek to the current position to clear the file's readahead buffer
		record_file.seek(0, os.SEEK_CUR)

		# read all new data from the file
		for line in record_file:
			y.append(float(line))
			x.append(i)
			i += 1

		# update the graph data
		graph1.set_xdata(x)
		graph1.set_ydata(y)
		graph2.set_xdata(x[:-100])
		graph2.set_ydata(running_mean_fast(y, 100)[:-100])
		graph3.set_xdata(x[:-1000])
		graph3.set_ydata(running_mean_fast(y, 1000)[:-1000])

		# set the axis to inclue the last 2000 data points
		plt.axis([i - 2000, i, 0.0, 1.5])

		# draw the graph and pause for three seconds
		plt.draw()
		plt.pause(3)
