#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time
import re

# calculates the running mean of x using N samples
def running_sum_fast(x, N):
	return np.convolve(x, np.ones(N))[N-1:]

# argument parsing
parser = argparse.ArgumentParser(description='Visualize BrainBot data')
parser.add_argument('--file', '-f', help="the output file to operate on")
parser.add_argument('--no-fitting', '-n', action='store_true', help="do not refit axis to latest data")

args = parser.parse_args()

# some globals to hold our data
x = []
y = []
i = 0

streams = { }

def parseChunk(chunk):
	
	for match in re.finditer('\w+:\n(?:-[^\n]*\n)*', chunk):

		subchunk = match.group(0)

		channels = subchunk.split('- ')

		# get the stream name and consume it
		streamName = channels[0].replace(':', '')
		del channels[0]

		if not streams.get(streamName):
			streams[streamName] = { }
			streams[streamName]['axes'] = plt.subplot(4, 3, len(streams.keys()))
			streams[streamName]['axes'].set_title(streamName, {'fontsize': 'small'})

		stream = streams[streamName]

		for channel in channels:
			name, value = channel.split(': ')

			if not stream.get(name):
				stream[name] = { }
				stream[name]['line'] = plt.plot([], [], linewidth=2)[0]
				stream[name]['data'] = []
				stream[name]['line'].set_label(name)
				plt.legend(loc='lower left', fontsize='small')

			stream[name]['data'].append(float(value))
			stream[name]['line'].set_xdata(range(len(stream[name]['data'])))
			stream[name]['line'].set_ydata(stream[name]['data'])
			xmax = len(stream[name]['data'])
			stream['axes'].set_xlim(xmax - 100, xmax)

# open our record file
reporting_filename = args.file or '/home/kaen/code/bitfighter/exe/screenshots/reporting'
record_filename = args.file or '/home/kaen/code/bitfighter/exe/screenshots/record'
with open(reporting_filename, 'r+', 1) as reporting_file, open(record_filename, 'r+', 1) as record_file:

	plt.plot()

	# parse initial data
	parseChunk(reporting_file.read())

	for line in record_file:
		y.append(float(line))
		x.append(i)
		i += 1

	# set up our record graphs
	record_axes = plt.subplot(4, 3, 12)
	graph2 = plt.plot(x[:-100], running_sum_fast(y, 100)[:-100],    'b', linewidth=1)[0]
	graph3 = plt.plot(x[:-1000], running_sum_fast(y, 1000)[:-1000], 'r', linewidth=1)[0]

	# turn on 'interactive' mode and loop until exit
	plt.ion()
	while True:

		# seek to the current position to clear the file's readahead buffer
		reporting_file.seek(0, os.SEEK_CUR)

		parseChunk(reporting_file.read())


		# seek to the current position to clear the file's readahead buffer
		record_file.seek(0, os.SEEK_CUR)

		# read all new data from the file
		for line in record_file:
			y.append(float(line))
			x.append(i)
			i += 1

		# update the graph data
		graph2.set_xdata(x[:-100])
		graph2.set_ydata(running_sum_fast(y, 100)[:-100])
		graph3.set_xdata(x[:-1000])
		graph3.set_ydata(running_sum_fast(y, 1000)[:-1000])

		if not args.no_fitting:
			record_axes.set_xlim(i - 2000, i)
			record_axes.set_ylim(-100, 100)

		# set the axis to inclue the last 2000 data points
		# 	plt.axis([i - 2000, i, -100, 100])

		# draw the graph and pause for three seconds
		plt.draw()
		plt.pause(3)
