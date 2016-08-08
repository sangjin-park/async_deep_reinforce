#!/home/itsukara/anaconda3/envs/tensorflow/bin/python

import numpy as np
import argparse
import matplotlib.pyplot as plt
from operator import itemgetter

parser = argparse.ArgumentParser(description="plot data in csv file")
parser.add_argument('filename')
parser.add_argument('-x', '--x-column', type=int, default=1,
                    help="column index of x-axis (0 origin)")
parser.add_argument('-y', '--y-column', type=int, default=2,
                    help="column index of y-axis (0 origin)")
parser.add_argument('-a', '--average-number-of-samples', dest="ans", type=int, default=100,
                    help="average number of samples")
parser.add_argument('-s', '--scale', type=float, default=1e6,
                    help="scale factor: data in x-column is divided by SCALE")
parser.add_argument('-xl', '--xlabel', default="M steps",
                    help="label of x-axis")
parser.add_argument('-yl', '--ylabel', default="Score",
                    help="label of y-axis")

args = parser.parse_args()

data = np.genfromtxt(args.filename, delimiter=",", dtype=np.float)

# sort data along args.x_column and make it np.array again
data = sorted(data, key=itemgetter(args.x_column))
data = np.array(data)

x = data[:, args.x_column]
y = data[:, args.y_column]
x = x / args.scale
plt.plot(x, y, ',')

weight = np.ones(args.ans, dtype=np.float)/args.ans
y_average = np.convolve(y, weight, 'valid')
header = np.ones(args.ans - 1) * y_average[0]
y_average = np.hstack((header, y_average))
plt.plot(x, y_average)

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.grid()

plt.show()
