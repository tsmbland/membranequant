from IA import *
import os

"""
Segmentation using Segment0 method
Single channel segmentation using cytoplasmic background curve to find edge

"""

# Load image(s)
ch1 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w2488 SP 535-50.TIF')
ch2 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w3488 SP 630-75.TIF')
ch3 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w4561 SP 630-75.TIF')
img1 = af_subtraction(ch1, ch2, m=2, c=0)
img2 = ch3 - 3545.6087

# Load background curve
bg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/CytBgGFPaf.txt')

# Set up segmenter
seg = Segmenter2(img_g=img1, img_r=5 * img2, bg_g=bg, bg_r=bg, mag=1, iterations=3, parallel=True)

# Specify ROI
seg.def_ROI()

# Run
seg.run()

# Plot
seg.plot()
seg.plot_straight()
