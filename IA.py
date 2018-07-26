from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import sys
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import cv2
from matplotlib import animation
from scipy.ndimage.interpolation import map_coordinates
from joblib import Parallel, delayed
import multiprocessing
import pickle
import copy
import pandas as pd

sns.set()
sns.set_style("ticks")
sns.despine()

os.chdir(os.path.expanduser('~/Desktop/Data'))


######################## FILE HANDLING #######################


def embryofiles(direc, num):
    """
    Returns a list of files containing the prefix corresponding to the embryo number specifies

    :param direc:
    :param num:
    :return:
    """

    embryofileslist = [os.path.basename(x) for x in glob.glob('%s/*_%s_*' % (direc, int(num)))]
    embryofileslist.extend(os.path.basename(x) for x in glob.glob('%s/*_%s.nd' % (direc, int(num))))
    return embryofileslist


def direcslist(direc):
    """

    Gives a list of directories in a given directory (full path)
    Excludes directories that contain !

    :param direc:
    :return:
    """

    dlist = glob.glob('%s/*/' % direc)
    dlist = [x[:-1] for x in dlist if '!' not in x]
    return dlist


def direcslist2(direc):
    """
    Like direcslist but goes down an extra level

    :param direc:
    :return:
    """
    dlist = []
    clist = glob.glob('%s/*/' % direc)
    clist = [x for x in clist if '!' not in x]
    for i in clist:
        dlist.extend(glob.glob('%s/*/' % i))
    dlist = [x for x in dlist if '!' not in x]
    return dlist


def embryos_direcslist(conds):
    dlist = []
    for i in conds:
        dlist.extend(glob.glob('%s/*/' % i))
    dlist = [x[:-1] for x in dlist if '!' not in x]
    return dlist


def organise(direc, start=0):
    """
    Better way of creating embryo folders
    Works for >10 embryos, but doesn't work if there are gaps in the numbering

    :param direc:
    :return:
    """

    embryo = start
    while len(embryofiles(direc, embryo)) != 0:
        os.makedirs('%s/%s' % (direc, embryo))
        embyrofileslist = embryofiles(direc, embryo)

        for file in embyrofileslist:
            os.rename('%s/%s' % (direc, file), '%s/%s/%s' % (direc, embryo, file))
        embryo += 1


######################## DATA IMPORT #######################


def loadimage(filename):
    """
    Given the filename of a TIFF, creates numpy array with pixel intensities

    :param filename:
    :return:
    """

    img = np.array(Image.open(filename), dtype=np.float64)
    img[img == 0] = np.nan
    return img


def readnd(direc):
    """

    :param direc: directory to embryo folder containing nd file
    :return: dictionary containing data from nd file
    """

    nd = {}
    nddirec = glob.glob('%s/_*.nd' % (direc))[0]
    f = open(nddirec, 'r').readlines()
    for line in f[:-1]:
        nd[line.split(', ')[0].replace('"', '')] = line.split(', ')[1].strip().replace('"', '')
    return nd


def read_conditions(direc):
    cond = {}
    a = os.path.basename(os.path.dirname(direc))
    cond['date'] = a.split('_')[0]
    cond['strain'] = a.split('_')[1]
    cond['exp'] = a.split('_')[2]
    cond['img'] = a.split('_')[3]
    return cond


class Data:
    """
    Structure to hold all imported data for an embryo
    Compatible with most experiments

    """

    def __init__(self, direc):

        # Directory
        self.direc = direc
        self.cond_direc = os.path.dirname(direc)

        # nd file
        try:
            self.nd = readnd(direc)
        except IndexError:
            self.nd = None

        # Conditions
        try:
            self.conds = read_conditions(direc)
        except:
            self.conds = None

        # EmbryoID
        self.emID = os.path.basename(direc)

        # DIC
        try:
            self.DIC = loadimage(sorted(glob.glob('%s/*DIC SP Camera*' % direc), key=len)[0])
        except IndexError:
            self.DIC = None

        # GFP
        try:
            self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 535-50*' % direc), key=len)[0])
        except IndexError:
            try:
                self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 525-50*' % direc), key=len)[0])
            except IndexError:
                try:
                    self.GFP = loadimage(sorted(glob.glob('%s/*GFP*' % direc), key=len)[0])
                except IndexError:
                    self.GFP = None

        # AF
        try:
            self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
        except IndexError:
            try:
                self.AF = loadimage(sorted(glob.glob('%s/*AF*' % direc), key=len)[0])
            except IndexError:
                self.AF = None

        # RFP
        try:
            self.RFP = loadimage(sorted(glob.glob('%s/*561 SP 630-75*' % direc), key=len)[0])
        except IndexError:
            self.RFP = None

        # ROI orig
        try:
            self.ROI_orig = np.loadtxt('%s/ROI_orig.txt' % direc)
        except FileNotFoundError:
            self.ROI_orig = None

        # ROI fitted
        try:
            self.ROI_fitted = np.loadtxt('%s/ROI_fitted.txt' % direc)
        except FileNotFoundError:
            try:
                self.ROI_fitted = np.loadtxt('%s/ROI_orig.txt' % direc)
            except FileNotFoundError:
                self.ROI_fitted = None

        # Surface area / Volume
        try:
            [self.sa, self.vol] = geometry(self.ROI_fitted)
        except:
            self.sa = None
            self.vol = None


class Data2:
    def __init__(self, direc):
        self.direc = direc
        self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 525-50*' % direc), key=len)[0])
        self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
        self.RFP = loadimage(sorted(glob.glob('%s/*561 SP 630-75*' % direc), key=len)[0])
        self.ROI_orig = np.loadtxt('%s/ROI_orig.txt' % direc)

        try:
            self.ROI_fitted = np.loadtxt('%s/ROI_fitted.txt' % direc)
        except FileNotFoundError:
            self.ROI_fitted = self.ROI_orig

        try:
            [self.sa, self.vol] = geometry(self.ROI_fitted)
        except:
            self.sa = None
            self.vol = None


class Data3:
    def __init__(self, direc):
        self.direc = direc
        self.GFP = loadimage(sorted(glob.glob('%s/*GFP*' % direc), key=len)[0])
        self.AF = loadimage(sorted(glob.glob('%s/*AF*' % direc), key=len)[0])
        self.RFP = loadimage(sorted(glob.glob('%s/*PAR2*' % direc), key=len)[0])
        self.ROI_orig = np.loadtxt('%s/ROI_orig.txt' % direc)

        try:
            self.ROI_fitted = np.loadtxt('%s/ROI_fitted.txt' % direc)
        except FileNotFoundError:
            self.ROI_fitted = self.ROI_orig

        try:
            [self.sa, self.vol] = geometry(self.ROI_fitted)
        except:
            self.sa = None
            self.vol = None

######################## DATA EXPORT #######################


def saveimg(img, direc):
    im = Image.fromarray(img)
    im.save(direc)


def pklsave(direc, object, name):
    file = open('%s/%s.pkl' % (direc, name), 'wb')
    pickle.dump(object, file)


def pklload(direc, name):
    file = open('%s/%s.pkl' % (direc, name), 'rb')
    res = pickle.load(file)
    return res


class Res:
    def __init__(self, cyt=None, cort=None, total=None):
        self.cyt = cyt
        self.cort = cort
        self.total = total


class Results:
    """
    Class that holds all results for embryos in given directories

    """

    def __init__(self, direcs):
        self.direcs = []

        self.cyts_GFP = np.array([])
        self.corts_GFP = np.array([])
        self.totals_GFP = np.array([])

        self.cyts_RFP = np.array([])
        self.corts_RFP = np.array([])
        self.totals_RFP = np.array([])

        self.gfp_spatial = []

        self.rfp_spatial = []

        self.gfp_csection = []

        self.rfp_csection = []

        for d in direcs:
            for e in direcslist(d):

                # GFP Quantification
                try:
                    res1 = pklload(e, 'res1')
                    self.cyts_GFP = np.append(self.cyts_GFP, [res1.cyt], axis=0)
                    self.corts_GFP = np.append(self.corts_GFP, [res1.cort], axis=0)
                    self.totals_GFP = np.append(self.totals_GFP, [res1.total], axis=0)
                except:
                    pass

                # RFP Quantification
                try:
                    res2 = pklload(e, 'res2')
                    self.cyts_RFP = np.append(self.cyts_RFP, [res2.cyt], axis=0)
                    self.corts_RFP = np.append(self.corts_RFP, [res2.cort], axis=0)
                    self.totals_RFP = np.append(self.totals_RFP, [res2.total], axis=0)
                except:
                    pass

                # GFP Spatial Quantification
                try:
                    res1_spatial = pklload(e, 'res1_spatial')
                    self.gfp_spatial.extend([res1_spatial])
                except:
                    pass

                # RFP Spatial Quantification
                try:
                    res2_spatial = pklload(e, 'res2_spatial')
                    self.rfp_spatial.extend([res2_spatial])
                except:
                    pass

                # GFP Cross Section
                try:
                    res1_csection = pklload(e, 'res1_csection')
                    self.gfp_csection.extend([res1_csection])
                except:
                    pass

                # RFP Cross Section
                try:
                    res2_csection = pklload(e, 'res2_csection')
                    self.rfp_csection.extend([res2_csection])
                except:
                    pass

                # Direcs
                self.direcs.extend([e])

        self.gfp_spatial = np.array(self.gfp_spatial)
        self.rfp_spatial = np.array(self.rfp_spatial)
        self.gfp_csection = np.array(self.gfp_csection)
        self.rfp_csection = np.array(self.rfp_csection)


####################### AF CORRECTION ######################

class Settings:
    """
    Structure to hold acquisition-specific settings (e.g. AF correction settings)

    """

    def __init__(self, m=0., c=0., x=0., y=0.):
        self.m = m
        self.c = c
        self.x = x
        self.y = y


def n2_analysis(direcs, plot=0):
    """
    Cytoplasmic mean correlation between the two channels for N2 embryos

    :param direcs:
    :param plot:
    :return:
    """

    xdata = []
    ydata = []

    for direc in direcs:
        embryos = direcslist(direc)

        # Cytoplasmic means
        for embryo in range(len(embryos)):
            data = Data(embryos[embryo])
            xdata.extend([cytoconc(data.AF, data.ROI_fitted)])
            ydata.extend([cytoconc(data.GFP, data.ROI_fitted)])

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    a = np.polyfit(xdata, ydata, 1)
    print(a)

    if plot == 1:
        x = np.array([0.9 * min(xdata.flatten()), 1.1 * max(xdata.flatten())])
        y = (a[0] * x) + a[1]
        plt.plot(x, y, c='b')
        plt.scatter(xdata, ydata)

    print(np.mean(xdata, 0))
    print(np.mean(ydata, 0))


def af_subtraction(ch1, ch2, settings):
    af = settings.m * ch2 + settings.c
    signal = ch1 - af
    return signal


#################### IMAGE PROCESSING ####################


def rolling_ave(img, window, periodic=1):
    """
    Returns rolling average across the x axis of an image (used for straightened profiles)

    :param img: image data
    :param window: number of pixels to average over. Odd number is best
    :param periodic: if 1, rolls over at ends
    :return: ave
    """

    if periodic == 0:
        ave = np.zeros([len(img[:, 0]), len(img[0, :])])
        for y in range(len(img[:, 0])):
            ave[y, int(window / 2):-int(window / 2)] = np.convolve(img[y, :], np.ones(window) / window, mode='valid')
        return ave

    elif periodic == 1:
        ave = np.zeros([len(img[:, 0]), len(img[0, :])])
        starts = np.append(range(len(img[0, :]) - int(window / 2), len(img[0, :])),
                           range(len(img[0, :]) - int(window / 2)))
        ends = np.append(np.array(range(int(np.ceil(window / 2)), len(img[0, :]))),
                         np.array(range(int(np.ceil(window / 2)))))

        for x in range(len(img[0, :])):
            if starts[x] < x < ends[x]:
                ave[:, x] = np.nanmean(img[:, starts[x]:ends[x]], axis=1)
            else:
                ave[:, x] = np.nanmean(np.append(img[:, starts[x]:], img[:, :ends[x]], axis=1), axis=1)
        return ave


def savitsky_golay(img, window=9, order=2):
    """
    Smoothens profile across the y dimension for each x
    Intended for rolling averaged profiles
    Not fully tested (need to refine parameters)

    :param img:
    :return:
    """

    savgos = np.zeros([len(img[:, 0]), len(img[0, :])])
    for x in range(len(img[0, :])):
        savgos[:, x] = savgol_filter(img[:, x], window, order, mode='mirror')
    return savgos


def interp(img, n):
    """
    Interpolates values along y axis for each x value
    :param img:
    :param n:
    :return:
    """

    interped = np.zeros([n, len(img[0, :])])
    for x in range(len(img[0, :])):
        interped[:, x] = np.interp(np.linspace(0, len(img[:, x]), n), range(len(img[:, x])), img[:, x])

    return interped


################# COORDINATE HANDLING ################


def offset_line(line, offset):
    """

    :param line: in the form [[x,y],[x,y]]
    :param offset:
    :return:
    """

    xcoors = line[:, 0]
    ycoors = line[:, 1]

    # Create coordinates
    rise = ycoors[1] - ycoors[0]
    run = xcoors[1] - xcoors[0]
    bisectorgrad = rise / run
    tangentgrad = -1 / bisectorgrad

    xchange = ((offset ** 2) / (1 + tangentgrad ** 2)) ** 0.5
    ychange = xchange / abs(bisectorgrad)
    newxs = xcoors + np.sign(rise) * np.sign(offset) * xchange
    newys = ycoors - np.sign(run) * np.sign(offset) * ychange

    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)
    return newcoors


def extend_line(line, extend):
    """


    :param line: in the form [[x,y],[x,y]]
    :param extend: e.g. 1.1 = 10% longer
    :return:
    """

    xcoors = line[:, 0]
    ycoors = line[:, 1]

    len = np.hypot((xcoors[0] - xcoors[1]), (ycoors[0] - ycoors[1]))
    extension = (extend - 1) * len * 0.5

    rise = ycoors[1] - ycoors[0]
    run = xcoors[1] - xcoors[0]
    bisectorgrad = rise / run
    tangentgrad = -1 / bisectorgrad

    xchange = ((extension ** 2) / (1 + bisectorgrad ** 2)) ** 0.5
    ychange = xchange / abs(tangentgrad)
    newxs = xcoors - np.sign(rise) * np.sign(tangentgrad) * xchange * np.array([-1, 1])
    newys = ycoors - np.sign(run) * np.sign(tangentgrad) * ychange * np.array([-1, 1])
    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)
    return newcoors


def rotate_coors(coors):
    """
    Rotates coordinate array so that most posterior point is at the beginning

    :param coors:
    :return:
    """

    # PCA to find long axis
    M = (coors - np.mean(coors.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M)

    # Find most extreme points
    a = np.argmin(np.minimum(score[0, :], score[1, :]))
    b = np.argmax(np.maximum(score[0, :], score[1, :]))

    # Find the one closest to user defined posterior
    dista = np.hypot((coors[0, 0] - coors[a, 0]), (coors[0, 1] - coors[a, 1]))
    distb = np.hypot((coors[0, 0] - coors[b, 0]), (coors[0, 1] - coors[b, 1]))

    # Rotate coordinates
    if dista < distb:
        newcoors = np.roll(coors, len(coors[:, 0]) - a, 0)
    else:
        newcoors = np.roll(coors, len(coors[:, 0]) - b, 0)

    return newcoors


def offset_coordinates(coors, offsets):
    """
    Reads in coordinates, adjusts according to offsets

    :param coors: two column array containing x and y coordinates. e.g. coors = np.loadtxt(filename)
    :param offsets: array the same length as coors. Direction?
    :return: array in same format as coors containing new coordinates

    To save this in a fiji readable format run:
    np.savetxt(filename, newcoors, fmt='%.4f', delimiter='\t')

    """

    xcoors = coors[:, 0]
    ycoors = coors[:, 1]

    if not hasattr(offsets, '__len__'):
        offsets = np.ones([len(xcoors)]) * offsets

    # Create new spline coordinates
    newxs = np.zeros([len(offsets)])
    newys = np.zeros([len(offsets)])
    forward = np.append(np.array(range(1, len(offsets))), [0])  # periodic boundaries
    back = np.append([len(offsets) - 1], np.array(range(len(offsets) - 1)))

    for i in range(len(offsets)):
        rise = ycoors[forward[i]] - ycoors[back[i]]
        run = xcoors[forward[i]] - xcoors[back[i]]
        bisectorgrad = rise / run
        tangentgrad = -1 / bisectorgrad

        xchange = ((offsets[i] ** 2) / (1 + tangentgrad ** 2)) ** 0.5
        ychange = xchange / abs(bisectorgrad)
        newxs[i] = xcoors[i] + np.sign(rise) * np.sign(offsets[i]) * xchange
        newys[i] = ycoors[i] - np.sign(run) * np.sign(offsets[i]) * ychange

    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)

    return newcoors


############### SEGMENTATION ################


def fit_background_v2(curve, bgcurve):
    """
    Used in segmentation. Takes interpolated curves, returns optimal fitting parameters.
    (popt[2] then used to adjust coordinates)

    :param curve:
    :param bgcurve:
    :return:
    """

    # Fix ends: use as initial guess for fitting
    line = np.polyfit(
        [np.nanmean(bgcurve[:int(len(bgcurve) * 0.2)]), np.nanmean(bgcurve[int(len(bgcurve) * 0.8):])],
        [np.nanmean(curve[:int(len(curve) * 0.2)]), np.nanmean(curve[int(len(curve) * 0.8):])], 1)

    # Fit
    popt, pcov = curve_fit(gaussian_plus, bgcurve, curve,
                           bounds=([line[0] - 1000, line[1] - 1000, 250, 0, 50],
                                   [line[0] + 1000, line[1] + 1000, 750, np.inf, 80]),
                           p0=[line[0], line[1], 500, 0, 65])

    return popt


def calc_offsets3(img, bgcurve):
    """
    Calculates coordinate offset required, by fitting straightened image to background curve

    What if img has zeros? (i.e. too close to the edge, or straightening error)
    Do I need to interpolate this much?

    :param img: straightened image
    :param bgcurve: background curve
    :return: offsets
    """

    img2 = interp(img, 1000)
    img3 = rolling_ave(img2, 50)
    img4 = savitsky_golay(img3, 251, 5)

    bgcurve2 = np.interp(np.linspace(0, len(bgcurve), 2000), range(len(bgcurve)), bgcurve)

    offsets = np.zeros(len(img[0, :]) // 5)

    for x in range(len(offsets)):
        try:
            a = fit_background_v2(img4[:, x * 5], bgcurve2)
            offsets[x] = (a[2] - 500) / 20
        except RuntimeError:
            offsets[x] = np.nan

    # Interpolate nans
    nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
    offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])

    # Interpolate
    offsets = np.interp(np.linspace(0, len(offsets), len(img[0, :])), range(len(offsets)), offsets)

    return offsets


def fit_coordinates_alg3(img, coors, bgcurve, iterations, mag=1):
    """
    Segmentation algorithm. Segments by fitting rolling average profiles to background curve, and offsetting original
    coordinates. Followed by smoothing

    :param img: af corrected embryo image
    :param coors: initial coordinates
    :param bgcurve: background curve
    :param iterations:
    :return: new coordinates

    Coors can be saved to txt file by e.g.
    np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')

    """

    for i in range(iterations):
        straight = straighten(img, coors, int(50 * mag))
        straight = interp(straight, 50)
        offsets = calc_offsets3(straight, bgcurve)
        coors = offset_coordinates(coors, offsets)
        coors = np.vstack(
            (savgol_filter(coors[:, 0], 19, 1, mode='wrap'), savgol_filter(coors[:, 1], 19, 1, mode='wrap'))).T

    coors = rotate_coors(coors)
    return coors


# def calc_offsets4(img, bgcurve):
#     """
#     Calculates coordinate offset required, by fitting straightened image to background curve
#     + throws mse of fitting
#
#     :param img: straightened image
#     :param bgcurve: background curve
#     :return: offsets
#     """
#
#     img2 = interp(img, 1000)
#     img3 = rolling_ave(img2, 50)
#     img4 = savitsky_golay(img3, 251, 5)
#
#     bgcurve2 = np.interp(np.linspace(0, len(bgcurve), 2000), range(len(bgcurve)), bgcurve)
#     bgcurve2 = savgol_filter(bgcurve2, 251, 5)
#
#     offsets = np.zeros(len(img[0, :]) // 10)
#     mse = np.zeros(len(img[0, :]) // 10)
#
#     for x in range(len(offsets)):
#         try:
#             a = fit_background_v2(img4[:, x * 10], bgcurve2)
#             offsets[x] = (a[2] - 500) / 20
#             mse[x] = ((img4[:, x * 10] - gaussian_plus(bgcurve2, *a)) ** 2).mean(axis=None)
#         except RuntimeError:
#             offsets[x] = np.nan
#             mse[x] = np.nan
#
#     # Interpolate nans
#     nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
#     offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])
#
#     nans, x = np.isnan(mse), lambda z: z.nonzero()[0]
#     mse[nans] = np.interp(x(nans), x(~nans), mse[~nans])
#
#     # Interpolate
#     offsets = np.interp(np.linspace(0, len(offsets), len(img[0, :])), range(len(offsets)), offsets)
#     mse = np.interp(np.linspace(0, len(mse), len(img[0, :])), range(len(mse)), mse)
#
#     return offsets, mse
#
#
# def fit_coordinates_alg4(data, settings, bgcurve, iterations, mag=1):
#     """
#     Segmentation algorithm that uses both channels
#     For each location, fits to both channels, and picks offset from channel with best fit
#
#     :param data:
#     :param settings:
#     :param bgcurve:
#     :param iterations:
#     :param mag:
#     :return:
#     """
#
#     coors = data.ROI_orig
#     for i in range(iterations):
#
#         # GFP
#         straight = straighten(af_subtraction(data.GFP, data.AF, settings), coors, int(50 * mag))
#         straight = interp(straight, 50)
#         offsets1, mse1 = calc_offsets4(straight, bgcurve)
#
#         # RFP
#         straight = straighten(data.RFP, coors, int(50 * mag))
#         straight = interp(straight, 50)
#         offsets2, mse2 = calc_offsets4(straight, bgcurve)
#
#         # plt.plot(offsets1)
#         # plt.plot(offsets2)
#         # plt.plot((offsets1 + offsets2) / 2)
#         # plt.show()
#
#         # offsets = np.zeros([len(offsets1)])
#         # for pos in range(len(offsets)):
#         #     if mse1[pos] < mse2[pos]:
#         #         offsets[pos] = offsets1[pos]
#         #     elif mse2[pos] < mse1[pos]:
#         #         offsets[pos] = offsets2[pos]
#
#         coors = offset_coordinates(coors, (offsets1 + offsets2) / 2)
#         coors = np.vstack(
#             (savgol_filter(coors[:, 0], 19, 1, mode='wrap'), savgol_filter(coors[:, 1], 19, 1, mode='wrap'))).T
#
#     try:
#         coors = rotate_coors(coors)
#     except:
#         print('segmentation error: %s' % data.direc)
#
#     return coors


def gaussian_plus(x, j, k, l, a, c):
    """
    For fitting signal curve to background curve + gaussian. Interpolated curves, sliding permitted
    (Used for segmentation)

    :param x: background curve
    :param j: bg curve fit
    :param k: bg curve fit
    :param l: bg curve fit (offset)
    :param a: gaussian param
    :param c: gaussian param
    :return: y: bgcurve + gaussian
    """

    y = (j * x[int(l):int(l) + 1000] + k) + (
        a * np.e ** (-((np.array(range(1000)) - (1000 - l)) ** 2) / (2 * (c ** 2))))

    return y


############## CORTICAL SIGNAL ###############


def fit_background_v2_2(curve, bgcurve):
    """
    Used for background subtraction. Returns fitted bgcurve which can then be subtracted from the signal curve

    :param curve:
    :param bgcurve:
    :return:
    """

    # Fix ends: use as initial guess for fitting
    line = np.polyfit(
        [np.nanmean(bgcurve[:int(len(bgcurve) * 0.2)]), np.nanmean(bgcurve[int(len(bgcurve) * 0.8):])],
        [np.nanmean(curve[:int(len(curve) * 0.2)]), np.nanmean(curve[int(len(curve) * 0.8):])], 1)

    # Fit
    popt, pcov = curve_fit(gaussian_plus2, bgcurve, curve, bounds=(
        [line[0] - 1000, line[1] - 1000, 0, 2.5], [line[0] + 1000, line[1] + 1000, np.inf, 4]),
                           p0=[line[0], line[1], 0, 3.25])

    # Create new bgcurve
    bgcurve = bgcurve * popt[0] + popt[1]

    return popt, bgcurve


def gaussian_plus2(x, j, k, a, c):
    """
    For fitting signal curve to background curve + gaussian. Non-interpolated curves, no sliding permitted
    For fitting background curve prior to subtraction

    :param x: background curve
    :param j: bg curve fit
    :param k: bg curve fit
    :param l: bg curve fit (offset)
    :param a: gaussian param
    :param b: gaussian param
    :param c: gaussian param
    :return: y: bgcurve + gaussian
    """

    y = (j * x + k) + (a * np.e ** (-((np.array(range(50)) - 25) ** 2) / (2 * (c ** 2))))

    return y


def cortical_signal_GFP(data, bg, settings, bounds, mag=1):
    # Correct autofluorescence
    img = af_subtraction(data.GFP, data.AF, settings=settings)

    # Straighten
    img = straighten(img, data.ROI_fitted, int(50 * mag))

    # Average
    if bounds[0] < bounds[1]:
        profile = np.nanmean(img[:, int(len(img[0, :]) * bounds[0]): int(len(img[0, :]) * bounds[1] + 1)], 1)
    else:
        profile = np.nanmean(
            np.hstack((img[:, :int(len(img[0, :]) * bounds[1] + 1)], img[:, int(len(img[0, :]) * bounds[0]):])), 1)

    # Adjust for magnification (e.g. if 2x multiplier is used)
    profile = np.interp(np.linspace(0, len(profile), 50), range(len(profile)), profile)

    # Get cortical signal
    a, bg = fit_background_v2_2(profile, bg[25:75])
    cort = np.trapz(profile - bg)

    return cort


def cortical_signal_RFP(data, bg, bounds, mag=1):
    # Straighten
    img = straighten(data.RFP, data.ROI_fitted, int(50 * mag))

    # Average
    if bounds[0] < bounds[1]:
        profile = np.nanmean(img[:, int(len(img[0, :]) * bounds[0]): int(len(img[0, :]) * bounds[1] + 1)], 1)
    else:
        profile = np.nanmean(
            np.hstack((img[:, :int(len(img[0, :]) * bounds[1] + 1)], img[:, int(len(img[0, :]) * bounds[0]):])), 1)

    # Adjust for magnification (e.g. if 2x multiplier is used)
    profile = np.interp(np.linspace(0, len(profile), 50), range(len(profile)), profile)

    # Get cortical signal
    a, bg = fit_background_v2_2(profile, bg[25:75])
    cort = np.trapz(profile - bg)

    return cort


def spatial_signal_GFP(data, bg, settings, mag=1):
    # Correct autofluorescence
    img = af_subtraction(data.GFP, data.AF, settings)

    # Straighten
    img_straight = straighten(img, data.ROI_fitted, int(50 * mag))

    # Smoothen
    img_straight = rolling_ave(img_straight, int(20 * mag))

    # Get cortical signals
    sigs = np.zeros(len(img_straight[0, :]) // 5)
    for x in range(len(sigs)):
        try:
            profile = img_straight[:, x * 5]
            profile = np.interp(np.linspace(0, len(profile), 50), range(len(profile)), profile)
            a, bg2 = fit_background_v2_2(profile, bg[25:75])
            sigs[x] = np.trapz(profile - bg2)

        except RuntimeError:
            sigs[x] = np.nan

    # Interpolate nans
    nans, x = np.isnan(sigs), lambda z: z.nonzero()[0]
    sigs[nans] = np.interp(x(nans), x(~nans), sigs[~nans])

    # Interpolate
    sigs = np.interp(np.linspace(0, len(sigs), 1000), range(len(sigs)), sigs)

    return sigs


def spatial_signal_RFP(data, bg, mag=1):
    # Straighten
    img_straight = straighten(data.RFP, data.ROI_fitted, int(50 * mag))

    # Smoothen
    img_straight = rolling_ave(img_straight, int(20 * mag))

    # Get cortical signals
    sigs = np.zeros(len(img_straight[0, :]) // 5)
    for x in range(len(sigs)):
        try:
            profile = img_straight[:, x * 5]
            profile = np.interp(np.linspace(0, len(profile), 50), range(len(profile)), profile)
            a, bg2 = fit_background_v2_2(profile, bg[25:75])
            sigs[x] = np.trapz(profile - bg2)

        except RuntimeError:
            sigs[x] = np.nan

    # Interpolate nans
    nans, x = np.isnan(sigs), lambda z: z.nonzero()[0]
    sigs[nans] = np.interp(x(nans), x(~nans), sigs[~nans])

    # Interpolate
    sigs = np.interp(np.linspace(0, len(sigs), 1000), range(len(sigs)), sigs)

    return sigs


def bounded_mean(array, bounds):
    if bounds[0] < bounds[1]:
        mean = np.nanmean(array[int(len(array) * bounds[0]): int(len(array) * bounds[1] + 1)])
    else:
        mean = np.nanmean(
            np.hstack((array[:int(len(array) * bounds[1] + 1)], array[int(len(array) * bounds[0]):])))
    return mean



################### CYTOPLASMIC SIGNAL #####################


def polycrop(img, polyline, enlarge=-10):
    """
    Crops image according to polyline coordinates

    :param img:
    :param polyline:
    :param enlarge:
    :return:
    """

    newcoors = np.int32(offset_coordinates(polyline, enlarge * np.ones([len(polyline[:, 0])])))
    mask = np.zeros(img.shape)
    mask = cv2.fillPoly(mask, [newcoors], 1)
    newimg = img * mask
    return newimg


def reverse_polycrop(img, polyline, enlarge=10):
    """
    Crops the area outside the polyline

    :param img:
    :param polyline:
    :param enlarge:
    :return:
    """

    newcoors = np.int32(offset_coordinates(polyline, enlarge * np.ones([len(polyline[:, 0])])))
    mask = np.ones(img.shape)
    mask = cv2.fillPoly(mask, [newcoors], 0)
    newimg = img * mask
    return newimg


def cytoplasmic_signal_GFP(data, settings, mag=1):
    # Correct autofluorescence
    img = af_subtraction(data.GFP, data.AF, settings=settings)

    # Get cytoplasmic signal
    img2 = polycrop(img, data.ROI_fitted, -20 * mag)
    mean = np.nanmean(img2[np.nonzero(img2)])

    return mean


def cytoplasmic_signal_RFP(data, mag=1):
    img = data.RFP

    # Subtract background
    bg = straighten(img, offset_coordinates(data.ROI_fitted, 50 * mag), int(50 * mag))
    mean1 = np.nanmean(bg[np.nonzero(bg)])

    # Get cytoplasmic signal
    img2 = polycrop(img, data.ROI_fitted, -20 * mag)
    mean2 = np.nanmean(img2[np.nonzero(img2)])  # mean, excluding zeros

    return mean2 - mean1


def cytoconc(img, coors):
    img2 = polycrop(img, coors, -20)
    mean = np.nanmean(img2[np.nonzero(img2)])
    return mean


########################### MISC ############################


def getsecs(time):
    """
    Converts time in hh:mm:ss into seconds (i.e. seconds from midnight)
    For comparison of times between acquisitions/events

    :param time: in string format (hh:mm:ss) e.g. nd['StartTime1'].split()[1]
    :return:
    """
    h, m, s = time.split(':')
    secs = int(h) * 3600 + int(m) * 60 + int(s)
    return secs


def geometry(coors):
    """
    Returns surface area and volume estimates given coordinates

    :param coors:
    :return:
    """

    # PCA
    M = (coors - np.mean(coors.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M)

    # Find long  axis
    len = max([max(score[0, :]) - min(score[0, :]), max(score[1, :]) - min(score[1, :])])
    width = max([max(score[0, :]) - min(score[0, :]), max(score[1, :]) - min(score[1, :])])

    volume = (4 / 3) * np.pi * (len / 2) * (width / 2) ** 2
    sa = 4 * np.pi * ((2 * (((len / 2) * (width / 2)) ** 1.6) + (((width / 2) * (width / 2)) ** 1.6)) / 3) ** (1 / 1.6)

    return sa, volume


def cross_section(img, coors, thickness, extend):
    """
    Returns cross section across the long axis of the embryo

    :param img:
    :param coors:
    :param thickness:
    :param extend:
    :return:
    """

    # PCA
    M = (coors - np.mean(coors.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M)

    # Find ends
    a = np.argmin(np.minimum(score[0, :], score[1, :]))
    b = np.argmax(np.maximum(score[0, :], score[1, :]))

    # Find the one closest to user defined posterior
    dista = np.hypot((coors[0, 0] - coors[a, 0]), (coors[0, 1] - coors[a, 1]))
    distb = np.hypot((coors[0, 0] - coors[b, 0]), (coors[0, 1] - coors[b, 1]))

    if dista < distb:
        line0 = np.array([coors[a, :], coors[b, :]])
    else:
        line0 = np.array([coors[b, :], coors[a, :]])

    # Extend line
    line0 = extend_line(line0, extend)

    # Thicken line
    line1 = offset_line(line0, thickness / 2)
    line2 = offset_line(line0, -thickness / 2)
    end1 = np.array(
        [np.linspace(line1[0, 0], line2[0, 0], thickness), np.linspace(line1[0, 1], line2[0, 1], thickness)]).T
    end2 = np.array(
        [np.linspace(line1[1, 0], line2[1, 0], thickness), np.linspace(line1[1, 1], line2[1, 1], thickness)]).T

    # Get cross section
    num_points = 100
    zvals = np.zeros([thickness, num_points])
    for section in range(thickness):
        xvalues = np.linspace(end1[section, 0], end2[section, 0], num_points)
        yvalues = np.linspace(end1[section, 1], end2[section, 1], num_points)
        zvals[section, :] = map_coordinates(img.T, [xvalues, yvalues])

    return np.flip(np.nanmean(zvals, 0), axis=0)


def straighten(img, coors, thickness):
    """
    Creates straightened image based on coordinates. Should be 1 pixel length apart in a loop

    :param img:
    :param coors:
    :param thickness:
    :return:
    """

    img2 = np.zeros([thickness, len(coors[:, 0])])
    offsets = np.linspace(thickness / 2, -thickness / 2, thickness)
    for section in range(thickness):
        sectioncoors = offset_coordinates(coors, offsets[section])
        img2[section, :] = map_coordinates(img.T, [sectioncoors[:, 0], sectioncoors[:, 1]])
    return img2


def normalise(instance1, instance2, objects):
    """
    Creates new class, by dividing objects in class instance 1 by corresponding objects in class instance 2

    :param instance1:
    :param instance2:
    :param objects:
    :return:
    """

    norm = copy.deepcopy(instance1)
    for o in objects:
        setattr(norm, o, getattr(instance1, o) / np.mean(getattr(instance2, o)))
    return norm


def rotated_embryo(img, coors, l):
    """
    Takes an image and rotates according to coordinates so that anterior is on left, posterior on right

    :param img:
    :param coors:
    :param l: length of each side in returned image
    :return:
    """

    # PCA
    M = (coors - np.mean(coors.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M)

    # Find ends
    a = np.argmin(np.minimum(score[0, :], score[1, :]))
    b = np.argmax(np.maximum(score[0, :], score[1, :]))

    # Find the one closest to user defined posterior
    dista = np.hypot((coors[0, 0] - coors[a, 0]), (coors[0, 1] - coors[a, 1]))
    distb = np.hypot((coors[0, 0] - coors[b, 0]), (coors[0, 1] - coors[b, 1]))

    if dista < distb:
        line0 = np.array([coors[a, :], coors[b, :]])
    else:
        line0 = np.array([coors[b, :], coors[a, :]])

    # Extend line
    length = np.hypot((line0[0, 0] - line0[1, 0]), (line0[0, 1] - line0[1, 1]))
    line0 = extend_line(line0, l / length)

    # Thicken line
    line1 = offset_line(line0, l / 2)
    line2 = offset_line(line0, -l / 2)
    end1 = np.array(
        [np.linspace(line1[0, 0], line2[0, 0], l), np.linspace(line1[0, 1], line2[0, 1], l)]).T
    end2 = np.array(
        [np.linspace(line1[1, 0], line2[1, 0], l), np.linspace(line1[1, 1], line2[1, 1], l)]).T

    # Get cross section
    num_points = l
    zvals = np.zeros([l, l])
    for section in range(l):
        xvalues = np.linspace(end1[section, 0], end2[section, 0], num_points)
        yvalues = np.linspace(end1[section, 1], end2[section, 1], num_points)
        zvals[section, :] = map_coordinates(img.T, [xvalues, yvalues])

    # Mirror
    zvals = np.flip(zvals, 1)

    return zvals


def experiment_database():
    df = pd.DataFrame(columns=['Direc', 'Date', 'Line', 'Experiment', 'Imaging', 'n'])

    for d in direcslist('.'):
        for c in direcslist(d):
            try:
                cond = read_conditions('%s/0' % c)
                n = len(direcslist(c))
                row = pd.DataFrame([[c[2:], cond['date'], cond['strain'], cond['exp'], cond['img'], n]],
                                   columns=['Direc', 'Date', 'Line', 'Experiment', 'Imaging', 'n'])
                df = df.append(row)

            except IndexError:
                pass

    df.to_csv('./db.csv')


def composite(data, settings, factor, mag=1):
    img1 = af_subtraction(data.GFP, data.AF, settings)
    bg = straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50 * mag), 50 * mag)
    img2 = data.RFP - np.nanmean(bg[np.nonzero(bg)])
    img3 = img1 + (factor * img2)
    return img3


experiment_database()

##############################################

# Get rid of redundant functions
# A coordinates/line data type?
# Slider template
# Fewer dependencies
# Organise code
# Need a better cherry bg curve
# Segmentation needs to be more resistant to errors
# Why is segmentation so much slower in parallel?
# Function to delete all pkl files from a directory (for tidying up)
# Also should delete old _straight and _cell files
# Segmentation so much quicker in optogenetics experiment. Why?
# Segmentation seems to fail a bit at the posterior pole
