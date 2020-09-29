import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

from IA import *

"""
Paths

"""

direcs = ...

"""
2 channel calibration

"""


def n2_analysis_2channel(direcs, plot=None, c=None):
    # Import data
    img1 = np.zeros([512, 512, len(direcs)])
    img2 = np.zeros([512, 512, len(direcs)])
    mask = np.zeros([512, 512, len(direcs)])
    for i, e in enumerate(direcs):
        img1[:, :, i] = load_image(sorted(glob.glob('%s/*488 SP 535-50*' % e), key=len)[0])
        img2[:, :, i] = load_image(sorted(glob.glob('%s/*488 SP 630-75*' % e), key=len)[0])
        coors = offset_coordinates(np.loadtxt(e + '/ROI.txt'), 5)
        mask[:, :, i] = make_mask([512, 512], coors)

    # Get correlation
    params, xdata2, ydata2 = af_correlation_pbyp(img1, img2, mask, plot=plot, c=c, sigma=2, intercept0=False)
    print(params)

    # Plot prediction
    plt.scatter(params[0] * xdata2 + params[1], ydata2, s=0.001)
    plt.plot([0, max(ydata2)], [0, max(ydata2)], c='k', linestyle='--')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

    return params


n2_analysis_2channel(direcs)

"""
3 channel calibration

"""


def n2_analysis_3channel(direcs, plot=None, c=None):
    # Import data
    img1 = np.zeros([512, 512, len(direcs)])
    img2 = np.zeros([512, 512, len(direcs)])
    img3 = np.zeros([512, 512, len(direcs)])
    mask = np.zeros([512, 512, len(direcs)])
    for i, e in enumerate(direcs):
        img1[:, :, i] = load_image(sorted(glob.glob('%s/*488 SP 535-50*' % e), key=len)[0])
        img2[:, :, i] = load_image(sorted(glob.glob('%s/*488 SP 630-75*' % e), key=len)[0])
        img3[:, :, i] = load_image(sorted(glob.glob('%s/*561 SP 630-75*' % e), key=len)[0])
        coors = offset_coordinates(np.loadtxt(e + '/ROI.txt'), 5)
        mask[:, :, i] = make_mask([512, 512], coors)

    # Get correlation
    params, xdata3, ydata3, zdata3 = af_correlation_pbyp_3channel(img1, img2, img3, mask, sigma=2, plot=plot, c=c,
                                                                  intercept0=False)
    print(params)

    # Plot prediction
    plt.scatter(params[0] * xdata3 + params[1] * ydata3 + params[2], zdata3, s=0.001)
    plt.plot([0, max(zdata3)], [0, max(zdata3)], c='k', linestyle='--')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

    return params


n2_analysis_3channel(direcs)
