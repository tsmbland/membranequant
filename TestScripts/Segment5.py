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

coors = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/ROI_fitted.txt')

# Load background curves
CytBg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/CytBgGFPaf.txt')
MemBg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/MemBgGFPaf.txt')

# Set up segmenter
seg = Segmenter5(img=img, coors=coors, cytbg=CytBg, membg=MemBg, mag=1, parallel=False, iterations=3, resolution=5)

# Specify ROI
seg.def_ROI()

# Run
seg.run()

# Plot
seg.plot()

# Straighten
plt.imshow(straighten(img, seg.newcoors, 50), cmap='gray', vmin=0)
plt.show()
