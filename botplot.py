#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
import sys
import select
import time

x = []
y = []

i = 0

def runningMeanFast(x, N):
	return np.convolve(x, np.ones((N,))/N)[(N-1):]

while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
  line = sys.stdin.readline()
  if line:
  	print(line)
	y.append(float(line))
	x.append(i)

	i += 1

graph1 = plt.plot(x, y, 'k.', markersize=1)[0]
graph2 = plt.plot(x, runningMeanFast(y, 100), 'b', linewidth=2)[0]
graph3 = plt.plot(x, runningMeanFast(y, 1000), 'r', linewidth=2)[0]

plt.ion()


while True:
	# If there's input ready, do something, else do something
	# else. Note timeout is zero so select won't block at all.
	while sys.stdin in select.select([sys.stdin], [], [], .2)[0]:
	  line = sys.stdin.readline()
	  if line:
	  	print(line)
		y.append(float(line))
		x.append(i)

		i += 1
	  else: # an empty line means stdin has been closed
		print('eof')
		exit(0)
	else:
	  	print('draw')
		graph1.set_xdata(x)
		graph1.set_ydata(y)
		graph2.set_xdata(x)
		graph2.set_ydata(runningMeanFast(y, 100))
		graph3.set_xdata(x)
		graph3.set_ydata(runningMeanFast(y, 1000))
		plt.show()
		plt.draw()
		plt.pause(3)
