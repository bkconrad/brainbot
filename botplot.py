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

		print('***')
		print(subchunk)

		channels = subchunk.split('- ')

		# get the stream name and consume it
		streamName = channels[0].replace(':', '')
		del channels[0]

		print(streamName)

		if not streams.get(streamName):
			streams[streamName] = { }
			streams[streamName]['axes'] = plt.subplot(3, 2, len(streams.keys()))
			streams[streamName]['axes'].set_title(streamName)

		stream = streams[streamName]

		for channel in channels:
			name, value = channel.split(': ')

			if not stream.get(name):
				stream[name] = { }
				stream[name]['line'] = plt.plot([], [])[0]
				stream[name]['data'] = []
				stream[name]['line'].set_label(name)
				plt.legend(loc='lower left', fontsize='small')

			stream[name]['data'].append(float(value))
			stream[name]['line'].set_xdata(range(len(stream[name]['data'])))
			stream[name]['line'].set_ydata(stream[name]['data'])
			xmax = len(stream[name]['data'])
			stream['axes'].set_xlim(xmax - 100, xmax)

# open our record file
filename = args.file or '/home/kaen/code/bitfighter/exe/screenshots/reporting'
with open(filename, 'r+', 1) as record_file:

	plt.plot()

	# parse initial data
	parseChunk(record_file.read())

	# turn on 'interactive' mode and loop until exit
	plt.ion()
	while True:

		# seek to the current position to clear the file's readahead buffer
		record_file.seek(0, os.SEEK_CUR)

		parseChunk(record_file.read())

		# set the axis to inclue the last 2000 data points
		# if not args.no_fitting:
		# 	plt.axis([i - 2000, i, -100, 100])

		# draw the graph and pause for three seconds
		plt.draw()
		plt.pause(3)
