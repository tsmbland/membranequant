from IA import *
import os

"""
Segmentation using Segment0 method
Single channel segmentation using cytoplasmic background curve to find edge

"""

# Load image(s)
ch1 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w2488 SP 535-50.TIF')
ch2 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w3488 SP 630-75.TIF')
img = af_subtraction(ch1, ch2, m=2, c=0)

# Load background curve
bg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/CytBgGFPaf.txt')

# Set up segmenter
seg = Segmenter1(img=img, bgcurve=bg, mag=1, iterations=3, parallel=False)

# Specify ROI
seg.def_ROI()

# Run
seg.run()

# Plot
seg.plot()

# Straighten
plt.imshow(straighten(img, seg.newcoors, 50), cmap='gray', vmin=0)
plt.show()
