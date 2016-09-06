import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2

parser = argparse.ArgumentParser(description="view psc_info")
parser.add_argument('filename')
parser.add_argument('-t', '--title', default=None,
                    help="title of figure")
parser.add_argument('--save', action='store_true',
                    help="save graph to file 'filename.png' and don't display it")
args = parser.parse_args()
if args.title is None:
  args.title = args.filename

f = open(args.filename, "rb")
psc_info = pickle.load(f)
psc_n = psc_info["psc_n"]
psc_vcount = psc_info["psc_vcount"]

print("psc_n = ", psc_n)

vcount = np.array(psc_vcount)
vp = vcount / psc_n

v = np.arange(128)
vexp = np.sum(v * vp, axis=1)
image = np.reshape(vexp, (42, 42))
image = cv2.resize(image, (42*8, 42*8))

if args.save:
  savefilename = args.title + "-exp.png"
  plt.imsave(savefilename, image, cmap="gray")
  print("Graph saved to ", savefilename)
else:
  plt.imshow(image, cmap="gray")
  plt.show()
