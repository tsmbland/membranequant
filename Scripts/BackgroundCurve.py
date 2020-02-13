import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

from IA import load_image, def_ROI, af_subtraction, cytbg

"""
Generate curve from single embryo

"""

# Load images
gfp = load_image(filename=...)
af = load_image(filename=...)

# AF subtract
img = af_subtraction(gfp, af, m=..., c=...)
plt.imshow(img, cmap='gray', vmin=0)
plt.show()

# Specify ROI
coors = def_ROI(img, spline=True)

# Generate background curve
bg = cytbg(img, coors, thickness=100)

# Save
np.savetxt(fname=..., X=bg)

"""
Generate average curve from multiple embryos
- perform above operations on multiple embryos and average the curves

"""
