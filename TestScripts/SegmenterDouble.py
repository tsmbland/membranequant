import sys

sys.path.append('..')
from IA import *
import os

# Load image(s)
ch1 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w2488 SP 535-50.TIF')
ch2 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w3488 SP 630-75.TIF')
ch3 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w4561 SP 630-75.TIF')
img1 = af_subtraction(ch1, ch2, m=2, c=0)
img2 = ch3 - 3545.6087

# Load background curves
CytBg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/CytBgGFPaf.txt')
MemBg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/MemBgGFPaf.txt')

# Set up segmenter
seg = Segmenter1Double(img_g=img1, img_r=3 * img2, cytbg_g=CytBg, cytbg_r=CytBg, membg_g=MemBg, membg_r=MemBg, mag=1,
                       resolution=1, cytbg_offset=4, thickness=50)

# Specify ROI
seg.def_ROI()
seg.coors = seg.fit_spline(seg.coors)

# Run
seg.run(iterations=3, parallel=True)

# Plot
seg.plot()
seg.plot_straight()
