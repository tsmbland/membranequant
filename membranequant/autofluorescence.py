import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import random
import glob
import scipy.odr as odr
from .funcs import load_image, offset_coordinates, make_mask

"""
Test
ODR functions for 3channel method

"""


class AfCorrelation:
    def __init__(self, paths, ch1_regex, ch2_regex, ch3_regex=None, roi_regex=None, sigma=0, intercept0=False,
                 expand=0):
        # Import images
        self.ch1 = np.array([load_image(sorted(glob.glob('%s/%s' % (p, ch1_regex)))[0]) for p in paths])
        self.ch2 = np.array([load_image(sorted(glob.glob('%s/%s' % (p, ch2_regex)))[0]) for p in paths])
        if ch3_regex is not None:
            self.ch3 = np.array([load_image(sorted(glob.glob('%s/%s' % (p, ch3_regex)))[0]) for p in paths])
        else:
            self.ch3 = None

        # Import rois, make mask
        self.mask = np.array([make_mask([512, 512],
                                        offset_coordinates(np.loadtxt(sorted(glob.glob('%s/%s' % (p, roi_regex)))[0]),
                                                           expand)) for p in paths])

        # Get correlation
        if self.ch3 is None:
            self.params, self.xdata, self.ydata = af_correlation(self.ch1, self.ch2, self.mask, sigma=sigma,
                                                                 intercept0=intercept0)
        else:
            self.params, self.xdata, self.ydata, self.zdata = af_correlation_3channel(self.ch1, self.ch2, self.ch3,
                                                                                      self.mask, sigma=sigma,
                                                                                      intercept0=intercept0)

    def plot_correlation(self, s=None):
        if self.ch3 is None:
            s = 0.01 if s is None else s
            self._plot_correlation_2channel(s=s)
        else:
            s = 0.1 if s is None else s
            self._plot_correlation_3channel(s=s)

    def _plot_correlation_2channel(self, s=0.01):
        fig, ax = plt.subplots()
        ax.scatter(self.xdata, self.ydata, s=s)
        xline = np.linspace(np.percentile(self.xdata, 0.01), np.percentile(self.xdata, 99.99), 20)
        yline = self.params[0] * xline + self.params[1]
        ax.plot(xline, yline, c='r')
        ax.set_xlim(np.percentile(self.xdata, 0.01), np.percentile(self.xdata, 99.99))
        ax.set_ylim(np.percentile(self.ydata, 0.01), np.percentile(self.ydata, 99.99))
        ax.set_xlabel('Channel 2')
        ax.set_ylabel('Channel 1')

    def _plot_correlation_3channel(self, s=1):
        # Set up figure
        ax = plt.figure().add_subplot(111, projection='3d')

        # Plot surface
        xx, yy = np.meshgrid([np.percentile(self.xdata, 0.01), np.percentile(self.xdata, 99.99)],
                             [np.percentile(self.ydata, 0.01), np.percentile(self.ydata, 99.99)])
        zz = self.params[0] * xx + self.params[1] * yy + self.params[2]
        ax.plot_surface(xx, yy, zz, alpha=0.2)

        # Scatter plot
        sample = random.sample(range(len(self.xdata)), min(10000, len(self.xdata)))
        ax.scatter(self.xdata[sample], self.ydata[sample], self.zdata[sample], s=s)

        # Tidy plot
        ax.set_xlim(np.percentile(self.xdata, 0.01), np.percentile(self.xdata, 99.99))
        ax.set_ylim(np.percentile(self.ydata, 0.01), np.percentile(self.ydata, 99.99))
        ax.set_zlim(np.percentile(self.zdata, 0.01), np.percentile(self.zdata, 99.99))
        ax.set_xlabel('Channel 2')
        ax.set_ylabel('Channel 3')
        ax.set_zlabel('Channel 1')

    # def plot_prediction(self, s=0.001):
    #     if self.ch3 is None:
    #         self._plot_prediction_2channel(s=s)
    #     else:
    #         self._plot_prediction_3channel(s=s)
    #
    # def _plot_prediction_2channel(self, s=0.001):
    #     plt.scatter(self.params[0] * self.xdata + self.params[1], self.ydata, s=s)
    #     plt.plot([0, max(self.ydata)], [0, max(self.ydata)], c='k', linestyle='--')
    #     plt.xlim(left=0)
    #     plt.ylim(bottom=0)
    #
    # def _plot_prediction_3channel(self, s=0.001):
    #     plt.scatter(self.params[0] * self.xdata + self.params[1] * self.ydata + self.params[2], self.zdata, s=s)
    #     plt.plot([0, max(self.zdata)], [0, max(self.zdata)], c='k', linestyle='--')
    #     plt.xlim(left=0)
    #     plt.ylim(bottom=0)


def af_correlation(img1, img2, mask, sigma=0, intercept0=False):
    """
    Calculates pixel-by-pixel correlation between two channels
    Takes 3d image stacks shape [n, 512, 512]

    :param img1: gfp channel
    :param img2: af channel
    :param mask: from make_mask function
    :param sigma: gaussian filter width
    :return:
    """

    # Gaussian filter
    if len(img1.shape) == 3:
        img1 = gaussian_filter(img1, sigma=[0, sigma, sigma])
        img2 = gaussian_filter(img2, sigma=[0, sigma, sigma])
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

    # Perform orthogonal distance regression
    if not intercept0:
        odr_mod = odr.Model(lambda b, x: b[0] * x + b[1])
        odr_data = odr.Data(xdata, ydata)
        odr_odr = odr.ODR(odr_data, odr_mod, beta0=[1, 0])
        output = odr_odr.run()
        params = output.beta

    else:
        odr_mod = odr.Model(lambda b, x: b[0] * x)
        odr_data = odr.Data(xdata, ydata)
        odr_odr = odr.ODR(odr_data, odr_mod, beta0=[1])
        output = odr_odr.run()
        params = [output.beta[0], 0]

    return params, xdata, ydata


def af_correlation_3channel(img1, img2, img3, mask, sigma=0, intercept0=False):
    """
    AF correlation taking into account red channel

    :param img1: GFP channel
    :param img2: AF channel
    :param img3: RFP channel
    :param mask:
    :return:
    """

    # Gaussian filter
    if len(img1.shape) == 3:
        img1 = gaussian_filter(img1, sigma=[0, sigma, sigma])
        img2 = gaussian_filter(img2, sigma=[0, sigma, sigma])
        img3 = gaussian_filter(img3, sigma=[0, sigma, sigma])
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
        params = popt
    else:
        popt, pcov = curve_fit(lambda x, slope1, slope2: slope1 * x[0] + slope2 * x[1], np.vstack((xdata, ydata)),
                               zdata)
        params = [popt[0], popt[1], 0]

    return params, xdata, ydata, zdata


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
