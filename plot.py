import numpy as np
import argparse
import time
import re
import sys
from operator import itemgetter

parser = argparse.ArgumentParser(description="plot data in A3C log file and update it periodically")
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
parser.add_argument('-t', '--title', default=None,
                    help="title of figure")
parser.add_argument('-n', '--interval', type=int, default=10,
                    help="interval of refresh (0 means no refresh)")
parser.add_argument('-e', '--endmark', default="END",
                    help="End Mark of in reward line")
parser.add_argument('--save', action='store_true',
                    help="save graph to file 'filename.png' and don't display it")

def read_data(f):
  data = []
  line = f.readline()
  while line != "":
    match = prog.match(line)
    if match:
      t = float(match.group(1))
      s = float(match.group(2))
      r = float(match.group(3))
      data.append([t, s, r])
    line = f.readline()
  return data

def draw_graph(ax, data):
  ans = args.ans
  if len(data) < 5:
    return
  elif len(data) < args.ans:
    ans = len(data) - 1

# sort data along args.x_column and make it np.array again
  data = sorted(data, key=itemgetter(args.x_column))
  data = np.array(data)

  x = data[:, args.x_column]
  y = data[:, args.y_column]
  y_max = np.max(y)
  ax.set_ylim(ymax = y_max * 1.05)

  x = x / args.scale
  ax.plot(x, y, ',')

  weight = np.ones(ans, dtype=np.float)/ans
  y_average = np.convolve(y, weight, 'valid')
  rim = ans - 1
  rim_l = rim // 2
  rim_r = rim - rim_l
  ax.plot(x[rim_l:-rim_r], y_average)

  ax.set_xlabel(args.xlabel)
  ax.set_ylabel(args.ylabel)

  ax.grid(linewidth=1, linestyle="-", alpha=0.1)

args = parser.parse_args()
if args.title is None:
  args.title = args.filename

# trick for headless environment 
if args.save:
  import matplotlib as mpl
  mpl.use('Agg')
import matplotlib.pyplot as plt

f = open(args.filename, "r")
prog = re.compile('t=\s*(\d+),s=\s*(\d+).*r=\s*(\d+)@' + args.endmark)

data = []
fig = plt.figure(args.title)
ax = fig.add_subplot(111)
ax.set_title(args.title)
while True:
  new_data = read_data(f)
  print(len(new_data), "data added.")
  if (len(new_data) > 0):
      data.extend(new_data)
      # ax.clear()
      draw_graph(ax, data)
  if args.save:
    savefilename = args.title + ".png"
    plt.savefig(savefilename)
    print("Graph saved to ", savefilename)
    sys.exit(0)
  elif args.interval == 0:
    plt.show()
    sys.exit(0)
  else:
    plt.pause(args.interval)

