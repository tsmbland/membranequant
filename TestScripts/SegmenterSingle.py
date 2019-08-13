import sys

sys.path.append('..')
from IA import *
import os

# Load image(s)
ch1 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w2488 SP 535-50.TIF')
ch2 = load_image(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/_0_w3488 SP 630-75.TIF')
img = af_subtraction(gaussian_filter(ch1, 1), gaussian_filter(ch2, 1), m=2, c=0)

coors = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/ROI_fitted.txt')

# Load background curves
CytBg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/CytBgGFPaf.txt')
MemBg = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/../TestDataset/MemBgGFPaf.txt')

# Set up segmenter
seg = Segmenter1Single(img=img, coors=coors, cytbg=CytBg, membg=MemBg, mag=1, resolution=1,
                       cytbg_offset=4, thickness=50)

# Specify ROI
seg.def_ROI()
seg.coors = seg.fit_spline(seg.coors)

seg.save('ROI.txt')
# seg.coors = seg.fit_spline(seg.coors)
# 
# # Run
# seg.run(parallel=True, iterations=3)
# seg.save('coors3.txt')
#
# # Plot
# seg.plot()
# seg.plot_straight()

# # Quantification
# coors = np.loadtxt('coors.txt')
# # coors = np.roll(coors, 200, 0)
#
# q = Quantifier(img=img, coors=coors, mag=1, thickness=50, cytbg=CytBg, membg=MemBg, cytbg_offset=4)
# q.run()
#
# plt.plot(q.sigs)
# plt.show()
#
# plt.plot(q.cyts)
# plt.show()
#
# # plt.plot(q.bg)
# # plt.show()
#
# # plt.plot(q.sigs / q.cyts)
# # plt.axhline(0)
# # plt.show()
#
# plt.imshow(q.straight_fit)
# plt.show()
#
# plt.imshow(q.straight_cyt)
# plt.show()
#
# plt.imshow(q.straight_mem)
# plt.show()
#
# # plt.imshow(q.straight_bg)
# # plt.show()
#
# plt.imshow(q.straight_resids)
# plt.show()
# #
# # # saveimg(q.straight, 'straight4.tif')
# # # saveimg(q.straight_fit, 'straight_fit4.tif')
# # # saveimg(q.straight_resids, 'straight_resids4.tif')
# # # saveimg(q.straight_mem, 'straight_mem4.tif')
# # # saveimg(q.straight_cyt, 'straight_cyt4.tif')
