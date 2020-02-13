import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

from IA import load_image, def_ROI, af_correlation_pbyp, make_mask, offset_coordinates

"""
Generate AF relationship from single embryo

"""

# Load images
gfp = load_image(filename=...)
af = load_image(filename=...)

# Specify ROI
coors = def_ROI(gfp, spline=True)

# Create mask
mask = make_mask([512, 512], offset_coordinates(coors, 25))

# Get correlation
a = af_correlation_pbyp(img1=gfp, img2=af, mask=mask, sigma=1, plot='scatter')
print('m=%s, c=%s' % (a[0], a[1]))

"""
Generate AF relationship from multiple embryos
- create 3D gfp, af and mask stacks shape [512, 512, n], then same procedure

"""
