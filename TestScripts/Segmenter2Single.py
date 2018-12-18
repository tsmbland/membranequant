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
seg = Segmenter2Single(img=img, coors=coors, cytbg=CytBg, membg=MemBg, mag=1, parallel=True, iterations=3, resolution=5)

# Specify ROI
seg.def_ROI()
seg.coors = seg.fit_spline(seg.coors)

# Run
seg.segment()

# Plot
seg.plot()
seg.plot_straight()

q = Quantifier2b(img=img, coors=seg.coors, mag=1, thickness=50, cytbg=CytBg, membg=MemBg)
q.run()
plt.plot(q.sigs)
plt.axhline(0)
plt.show()


