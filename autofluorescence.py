import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import cv2
import random
from IA import polycrop


def make_mask(shape, roi):
    return cv2.fillPoly(np.zeros(shape) * np.nan, [np.int32(roi)], 1)


def af_correlation(img1, img2, mask, sigma=0, plot=None, c=None, intercept0=False):
    """

    Calculates pixel-by-pixel correlation between two channels
    Takes 3d image stacks shape [512, 512, n]

    :param img1: gfp channel
    :param img2: af channel
    :param mask: from make_mask function
    :param sigma: gaussian filter width
    :param plot: type of plot to show
    :param c: colour on plot
    :return:
    """

    # Gaussian filter
    if len(img1.shape) == 3:
        img1 = gaussian_filter(img1, sigma=[sigma, sigma, 0])
        img2 = gaussian_filter(img2, sigma=[sigma, sigma, 0])
    else:
        img1 = gaussian_filter(img1, sigma=sigma)
        img2 = gaussian_filter(img2, sigma=sigma)

    # Mask
    img1 *= mask
    img2 *= mask

    # Flatten
    xdata = img2.flatten()
    ydata = img1.flatten()

    # Remove nans
    xdata = xdata[~np.isnan(xdata)]
    ydata = ydata[~np.isnan(ydata)]

    # Fit to line
    if not intercept0:
        popt, pcov = curve_fit(lambda x, slope, intercept: slope * x + intercept, xdata, ydata)
        a = popt
    else:
        popt, pcov = curve_fit(lambda x, slope: slope * x, xdata, ydata)
        a = [popt[0], 0]

    # Scatter plot
    if plot == 'scatter':
        plt.scatter(xdata, ydata, s=0.001, c=c)
        xline = np.linspace(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99), 20)
        yline = a[0] * xline + a[1]
        plt.plot(xline, yline, c='r')
        plt.xlim(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99))
        plt.ylim(np.percentile(ydata, 0.01), np.percentile(ydata, 99.99))
        plt.xlabel('AF channel')
        plt.ylabel('GFP channel')

    # Heatmap
    elif plot == 'heatmap':
        xline = np.linspace(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99), 20)
        yline = a[0] * xline + a[1]
        plt.plot(xline, yline, c='r')
        heatmap, xedges, yedges = np.histogram2d(xdata, ydata, bins=500)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.xlim(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99))
        plt.ylim(np.percentile(ydata, 0.01), np.percentile(ydata, 99.99))
        plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='Greys')
    else:
        pass

    return a, xdata, ydata


def af_correlation_3channel(img1, img2, img3, mask, sigma=0, plot=None, ax=None, c=None, intercept0=False):
    """
    AF correlation taking into account red channel

    :param img1: GFP channel
    :param img2: AF channel
    :param img3: RFP channel
    :param mask:
    :param plot:
    :return:
    """

    # Gaussian filter
    if len(img1.shape) == 3:
        img1 = gaussian_filter(img1, sigma=[sigma, sigma, 0])
        img2 = gaussian_filter(img2, sigma=[sigma, sigma, 0])
        img3 = gaussian_filter(img3, sigma=[sigma, sigma, 0])
    else:
        img1 = gaussian_filter(img1, sigma=sigma)
        img2 = gaussian_filter(img2, sigma=sigma)
        img3 = gaussian_filter(img3, sigma=sigma)

    # Mask
    img1 *= mask
    img2 *= mask
    img3 *= mask

    # Flatten
    xdata = img2.flatten()
    ydata = img3.flatten()
    zdata = img1.flatten()

    # Remove nans
    xdata = xdata[~np.isnan(xdata)]
    ydata = ydata[~np.isnan(ydata)]
    zdata = zdata[~np.isnan(zdata)]

    # Fit to surface
    if not intercept0:
        popt, pcov = curve_fit(lambda x, slope1, slope2, intercept: slope1 * x[0] + slope2 * x[1] + intercept,
                               np.vstack((xdata, ydata)), zdata)
        p = popt
    else:
        popt, pcov = curve_fit(lambda x, slope1, slope2: slope1 * x[0] + slope2 * x[1], np.vstack((xdata, ydata)),
                               zdata)
        p = [popt[0], popt[1], 0]

    # Scatter plot
    if plot == 'scatter':
        # Set up figure
        if not ax:
            ax = plt.figure().add_subplot(111, projection='3d')

        # Plot surface
        xx, yy = np.meshgrid([np.percentile(xdata, 0.01), np.percentile(xdata, 99.99)],
                             [np.percentile(ydata, 0.01), np.percentile(ydata, 99.99)])
        zz = p[0] * xx + p[1] * yy + p[2]
        ax.plot_surface(xx, yy, zz, alpha=0.2, color=c)

        # Scatter plot
        set = random.sample(range(len(xdata)), 10000)
        ax.scatter(xdata[set], ydata[set], zdata[set], s=1, c=c)

        # Tidy plot
        ax.set_xlim(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99))
        ax.set_ylim(np.percentile(ydata, 0.01), np.percentile(ydata, 99.99))
        ax.set_zlim(np.percentile(zdata, 0.01), np.percentile(zdata, 99.99))
        ax.set_xlabel('AF')
        ax.set_ylabel('RFP')
        ax.set_zlabel('GFP')

    return p, xdata, ydata, zdata


def af_subtraction(ch1, ch2, m, c):
    """
    Subtract ch2 from ch1
    ch2 is first adjusted to m * ch2 + c

    :param ch1:
    :param ch2:
    :param m:
    :param c:
    :return:
    """

    af = m * ch2 + c
    signal = ch1 - af
    return signal


def af_subtraction_3channel(ch1, ch2, ch3, m1, m2, c):
    """

    """

    af = m1 * ch2 + m2 * ch3 + c
    signal = ch1 - af
    return signal


def bg_subtraction(img, roi, band=(25, 75)):
    a = polycrop(img, roi, band[1]) - polycrop(img, roi, band[0])
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a
