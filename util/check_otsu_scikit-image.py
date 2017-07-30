#!/usr/bin/python

#Main source: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html

import matplotlib.pyplot as plt
from skimage import io
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import exposure

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Image path.")

args = vars(ap.parse_args())

image_path = args["image"]

original = io.imread(image_path)
val = filters.threshold_otsu(original)

hist, bins_center = exposure.histogram(original)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(original, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(original < val, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()
