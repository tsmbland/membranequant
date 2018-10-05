from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
# from matplotlib.widgets import Slider
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import cv2
from scipy.ndimage.interpolation import map_coordinates
from joblib import Parallel, delayed
import multiprocessing
import copy
import pandas as pd
import scipy.misc
import shutil

"""
From local to server: '../../../../../../../Volumes/lab-goehringn/working/Tom/ModelData'

From server to server: '../working/Tom/ModelData'

From local to local: '../../ImageAnalysis', '../../ImagingData/Tidy' 
"""

# # Server to server
# ddirec = '../working/Tom/ImagingData/Tidy'
# adirec = '../working/Tom/ImageAnalysis/Analysis'

# Local to local
# ddirec = '../../../../../Desktop/Data'
# adirec = '../../../../../Desktop/Analysis'
# fdirec = '../../../../../Desktop/Figures'
ddirec = '/Users/blandt/Desktop/Data'
adirec = '/Users/blandt/Desktop/Analysis'
fdirec = '/Users/blandt/Desktop/Figures'


######################## FILE HANDLING #######################


def embryofiles(direc, num):
    """
    Returns a list of files containing the prefix corresponding to the embryo number specified

    :param direc:
    :param num:
    :return:
    """

    embryofileslist = [os.path.basename(x) for x in glob.glob('%s/*_%s_*' % (direc, int(num)))]
    embryofileslist.extend(os.path.basename(x) for x in glob.glob('%s/*_%s.nd' % (direc, int(num))))
    return embryofileslist


def stagefiles(direc, num):
    """
    Returns a list of files containing the suffix corresponding to the stage number specified

    :param direc:
    :param num:
    :return:
    """

    stagefileslist = [os.path.basename(x) for x in glob.glob('%s/*_s%s_*' % (direc, int(num)))]
    stagefileslist.extend(os.path.basename(x) for x in glob.glob('%s/*_s%s.TIF' % (direc, int(num))))
    return stagefileslist


def timefiles(direc, num):
    """
    Returns a list of files containing the suffix corresponding to the timepoint specified

    :param direc:
    :param num:
    :return:
    """

    timefileslist = [os.path.basename(x) for x in glob.glob('%s/*_t%s.TIF' % (direc, int(num)))]
    return timefileslist


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


def split_stage_positions(direc, start=0):
    stage = start
    while len(stagefiles(direc, stage)) != 0:
        os.makedirs('%s/%s' % (direc, stage))
        stagefileslist = stagefiles(direc, stage)

        for file in stagefileslist:
            os.rename('%s/%s' % (direc, file), '%s/%s/%s' % (direc, stage, file))
        stage += 1


def split_timepoints(direc, start=0):
    time = start
    while len(timefiles(direc, time)) != 0:
        os.makedirs('%s/%s' % (direc, time))
        timefileslist = timefiles(direc, time)

        for file in timefileslist:
            os.rename('%s/%s' % (direc, file), '%s/%s/%s' % (direc, time, file))
        time += 1


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
    cond['misc'] = a.split('_')[4:]
    return cond


######################## DATA EXPORT #######################

def savedata(res, direc):
    d = vars(res)
    for key, value in d.items():
        np.savetxt('%s/%s.txt' % (direc, key), value, fmt='%.4f', delimiter='\t')


def saveimg(img, direc):
    im = Image.fromarray(img)
    im.save(direc)


def saveimg_jpeg(img, direc, min, max):
    a = scipy.misc.toimage(img, cmin=min, cmax=max)
    a.save(direc)


####################### AF CORRECTION ######################

class Settings:
    """
    Structure to hold acquisition-specific settings (e.g. AF correction settings)

    m
    c

    """

    def __init__(self, m=0., c=0., x=0., y=0.):
        self.m = m
        self.c = c


def n2_analysis(direcs, plot=0):
    """
    Cytoplasmic mean correlation between the two channels for N2 embryos

    :param direcs:
    :param plot:
    :return:
    """

    xdata = []
    ydata = []
    cdata = []
    bdata = []

    for direc in direcs:
        embryos = direcslist(direc)

        for embryo in range(len(embryos)):
            data = Data(embryos[embryo])

            # Cytoplasmic means
            xdata.extend([cytoconc(data.AF, data.ROI_fitted)])
            ydata.extend([cytoconc(data.GFP, data.ROI_fitted)])
            cdata.extend([cytoconc(data.RFP, data.ROI_fitted)])

            # Background
            bg = straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50 * 1), int(50 * 1))
            mean1 = np.nanmean(bg[np.nonzero(bg)])
            bdata.extend([mean1])

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    bdata = np.array(bdata)
    cdata = np.array(cdata)

    a = np.polyfit(xdata, ydata, 1)
    print(a)
    print(np.mean(xdata, 0))
    print(np.mean(ydata, 0))

    if plot == 1:
        x = np.array([0.9 * min(xdata.flatten()), 1.1 * max(xdata.flatten())])
        y = (a[0] * x) + a[1]
        plt.plot(x, y, c='b')
        plt.scatter(xdata, ydata)

    a2 = np.polyfit(bdata, cdata, 1)
    print(a2)
    print(np.mean(bdata, 0))
    print(np.mean(cdata, 0))

    if plot == 2:
        x = np.array([0.9 * min(bdata.flatten()), 1.1 * max(bdata.flatten())])
        y = (a[0] * x) + a[1]
        plt.plot(x, y, c='b')
        plt.scatter(bdata, cdata)


def N2Analysis(r, r_base=None, show=False):
    xdata = r.a.cyt
    ydata = r.g.cyt
    plt.scatter(xdata, ydata)
    a = np.polyfit(xdata, ydata, 1)
    print(a)
    x = np.array([0.9 * min(xdata.flatten()), 1.1 * max(xdata.flatten())])
    y = (a[0] * x) + a[1]
    plt.plot(x, y, c='b')
    if show:
        plt.show()

        # # Red shift
        # a = np.polyfit(r_base.a.cyt, r_base.g.cyt, 1)
        # offset = r.a.cyt - (r.g.cyt - a[1]) / a[0]
        # plt.scatter(r.r.cyt - r.r.ext, offset)
        # plt.show()


############### SEGMENTATION ################


class Segmenter:
    """

   Input data:
   img_g           image
   bgcurve_g       background curve. Must be 2* wider than the eventual profile for img
   coors           original coordinates

   Input parameters:
   mag             magnification (1 = 60x)
   iterations      number of times to run algorithm
   parallel        True: preform segmentation in parallel
   resolution      will perform fitting algorithm at gaps set by this, interpolating between
   freedom         0 = no freedom, 1 = max freedom

   Misc parameters:
   it              Interpolation of profiles/bgcurves
   thickness       thickness of straightened images
   rol_ave         sets the degree of image smoothening
   end_region      for end fitting
   n_end_fits      number of end fits to perform, will interpolate



   To to:
   - Removing outliers better than smoothening coordinates at end

   """

    def __init__(self, img, bgcurve, coors, mag=1, iterations=2, parallel=True, resolution=5, freedom=0.3, save=False,
                 direc=None, plot=False):

        # Inputs
        self.img = img
        self.bgcurve = bgcurve
        self.coors = coors
        self.mag = mag
        self.iterations = iterations
        self.parallel = parallel
        self.resolution = resolution
        self.freedom = freedom
        self.save = save
        self.direc = direc
        self.plot = plot
        self.itp = 1000
        self.thickness = 50
        self.rol_ave = 50
        self.end_region = 0.2
        self.n_end_fits = 50

        self.newcoors = coors
        self.resids = np.zeros(len(coors))

        # Warnings
        if self.save and self.direc is None:
            raise Exception('Must specify directory')

    def run(self):
        """
        Performs segmentation algorithm

        """

        for i in range(self.iterations):

            # Straighten
            straight = straighten(self.img, self.newcoors, int(self.thickness * self.mag))

            # Filter/smoothen/interpolate images
            straight = rolling_ave(interp(straight, self.itp), self.rol_ave)

            # Interpolate bgcurves
            bgcurve = interp_1d_array(self.bgcurve, 2 * self.itp)

            # Calculate offsets
            if self.parallel:
                offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(self.fit_background)(straight[:, x * self.resolution], bgcurve)
                    for x in range(len(straight[0, :]) // self.resolution)))
            else:
                offsets = np.zeros(len(straight[0, :]) // self.resolution)
                for x in range(len(straight[0, :]) // self.resolution):
                    offsets[x] = self.fit_background(straight[:, x * self.resolution], bgcurve)

            # Interpolate nans
            nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
            offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])

            # Interpolate
            offsets = interp_1d_array(offsets, len(self.newcoors))

            # Offset coordinates
            self.newcoors = offset_coordinates(self.newcoors, offsets)

            # Filter
            self.newcoors = np.vstack(
                (savgol_filter(self.newcoors[:, 0], 19, 1, mode='wrap'),
                 savgol_filter(self.newcoors[:, 1], 19, 1, mode='wrap'))).T

        # Rotate
        self.newcoors = rotate_coors(self.newcoors)

        # Save
        if self.save:
            np.savetxt('%s/ROI_fitted.txt' % self.direc, self.newcoors, fmt='%.4f', delimiter='\t')

        # Plot
        if self.plot:
            plt.imshow(straighten(self.img, self.newcoors, self.thickness * self.mag), cmap='gray')
            plt.show()

    def fit_background(self, curve, bgcurve):
        """
        Takes cross sections from images and finds optimal offset for alignment

        :param curve: signal curve from cross section of straightened img_g
        :param bgcurve:
        :return: o: offset value for this section
        """

        # Fix ends, interpolate: Green
        ms, cs = self.fix_ends(curve, bgcurve)
        msfull = np.zeros([2 * self.itp])
        msfull[int((self.itp / 2) * (1 - self.freedom)):int((self.itp / 2) * (1 + self.freedom))] = interp_1d_array(ms,
                                                                                                                    self.itp * self.freedom)
        csfull = np.zeros([2 * self.itp])
        csfull[int((self.itp / 2) * (1 - self.freedom)):int((self.itp / 2) * (1 + self.freedom))] = interp_1d_array(cs,
                                                                                                                    self.itp * self.freedom)

        # Input for curve fitter
        x = np.stack((bgcurve, msfull, csfull), axis=0)

        # Fit gaussian to find offset
        try:
            popt, pcov = curve_fit(self.gaussian_plus, x, curve,
                                   bounds=([(self.itp / 2) * (1 - self.freedom), 0, 20],
                                           [(self.itp / 2) * (1 + self.freedom), np.inf, 200]),
                                   p0=[self.itp / 2, 0, 100])

            o = (popt[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fix_ends(self, curve, bgcurve):
        """
        Calculates parameters to fit the ends of curve to bgcurve, across different alignments

        :param curve: signal curve
        :param bgcurve: background curve
        :return: ms, cs: end fitting parameters
        """

        x = (self.itp * self.freedom) / self.n_end_fits
        ms = np.zeros([self.n_end_fits])
        cs = np.zeros([self.n_end_fits])
        for l in range(self.n_end_fits):
            bgcurve_seg = bgcurve[
                          int(((self.itp / 2) * (1 - self.freedom)) + (x * l)): int(
                              (self.itp + (self.itp / 2) * (1 - self.freedom)) + (x * l))]
            line = np.polyfit(
                [np.mean(curve[:int(len(curve) * self.end_region)]),
                 np.mean(curve[int(len(curve) * (1 - self.end_region)):])],
                [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
                 np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])], 1)
            ms[l] = line[0]
            cs[l] = line[1]
        return ms, cs

    def gaussian_plus(self, x, l, a, c):
        """
        Function used for fitting algorithm

        :param x: input 5 column array containing bgcurve and end fit parameters
        :param l: offset parameter
        :param a: gaussian height
        :param c: gaussian width
        :return: y: curve
        """

        g0 = a * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / c)
        g1 = a * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / c))
        g = np.append(g0, g1)
        y = (x[0, int(l):int(l) + self.itp] + g - x[2, int(l)]) / x[1, int(l)]

        return y


class Segmenter2:
    """

    Input data:
    img_g           green channel image
    img_r           red channel image
    bgcurve_g       green background curve. Must be 2* wider than the eventual profile for img_r
    bgcurve_r       red background curve. Must be 2* wider than the eventual profile for img_r
    coors           original coordinates

    Input parameters:
    mag             magnification (1 = 60x)
    iterations      number of times to run algorithm
    parallel        True: preform segmentation in parallel
    resolution      will perform fitting algorithm at gaps set by this, interpolating between
    freedom         0 = no freedom, 1 = max freedom

    Misc parameters:
    it              Interpolation of profiles/bgcurves
    thickness       thickness of straightened images
    rol_ave         sets the degree of image smoothening
    end_region      for end fitting
    n_end_fits      number of end fits to perform, will interpolate

    Outputs:
    newcoors
    resids

    To to:
    - Removing outliers better than smoothening coordinates at end

    """

    def __init__(self, img_g, img_r, bg_g, bg_r, coors, mag=1, iterations=2, parallel=True, resolution=5,
                 freedom=0.3, save=False, direc=None, plot=False):

        self.img_g = img_g
        self.img_r = img_r
        self.bg_g = bg_g
        self.bg_r = bg_r
        self.coors = coors
        self.mag = mag
        self.iterations = iterations
        self.parallel = parallel
        self.resolution = resolution
        self.freedom = freedom
        self.save = save
        self.direc = direc
        self.plot = plot
        self.itp = 1000
        self.thickness = 50
        self.rol_ave = 50
        self.end_region = 0.2
        self.n_end_fits = 20

        self.newcoors = coors
        self.resids = np.zeros(len(coors))

        # Warnings
        if self.save and self.direc is None:
            raise Exception('Must specify directory')

    def run(self):
        """
        Performs segmentation algorithm

        """

        for i in range(self.iterations):

            # Straighten
            straight_g = straighten(self.img_g, self.newcoors, int(self.thickness * self.mag))
            straight_r = straighten(self.img_r, self.newcoors, int(self.thickness * self.mag))

            # Smoothen/interpolate images
            straight_g = rolling_ave(interp(straight_g, self.itp), self.rol_ave)
            straight_r = rolling_ave(interp(straight_r, self.itp), self.rol_ave)

            # Interpolate bgcurves
            bgcurve_g = interp_1d_array(self.bg_g, 2 * self.itp)
            bgcurve_r = interp_1d_array(self.bg_r, 2 * self.itp)

            # Calculate offsets
            if self.parallel:
                offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(self.fit_background_2col)(straight_g[:, x * self.resolution],
                                                      straight_r[:, x * self.resolution], bgcurve_g, bgcurve_r)
                    for x in range(len(straight_g[0, :]) // self.resolution)))
            else:
                offsets = np.zeros(len(straight_g[0, :]) // self.resolution)
                for x in range(len(straight_g[0, :]) // self.resolution):
                    offsets[x] = self.fit_background_2col(straight_g[:, x * self.resolution],
                                                          straight_r[:, x * self.resolution], bgcurve_g, bgcurve_r)

            # Interpolate nans
            nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
            offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])

            # Interpolate
            offsets = interp_1d_array(offsets, len(self.newcoors))

            # Offset coordinates
            self.newcoors = offset_coordinates(self.newcoors, offsets)

            # Filter
            self.newcoors = np.vstack(
                (savgol_filter(self.newcoors[:, 0], 19, 1, mode='wrap'),
                 savgol_filter(self.newcoors[:, 1], 19, 1, mode='wrap'))).T

            # Interpolate nans
            nans, x = np.isnan(self.newcoors), lambda z: z.nonzero()[0]
            self.newcoors[nans] = np.interp(x(nans), x(~nans), self.newcoors[~nans])

        # Rotate
        self.newcoors = rotate_coors(self.newcoors)

        # Save
        if self.save:
            np.savetxt('%s/ROI_fitted.txt' % self.direc, self.newcoors, fmt='%.4f', delimiter='\t')

        # Plot
        if self.plot:
            plt.imshow(straighten(self.img_g, self.newcoors, self.thickness * self.mag), cmap='gray')
            plt.show()
            plt.imshow(straighten(self.img_r, self.newcoors, self.thickness * self.mag), cmap='gray')
            plt.show()

    def fit_background_2col(self, curve_g, curve_r, bgcurve_g, bgcurve_r):
        """
        Takes cross sections from images and finds optimal offset for alignment

        :param curve_g: signal curve from cross section of straightened img_g
        :param curve_r: signal curve from cross section of straightened img_r
        :param bgcurve_g:
        :param bgcurve_r:
        :return: o: offset value for this section
        """

        # Fix ends, interpolate: Green
        ms_g, cs_g = self.fix_ends(curve_g, bgcurve_g)
        msfull_g = np.zeros([4 * self.itp])
        msfull_g[int((self.itp / 2) * (1 - self.freedom)):int((self.itp / 2) * (1 + self.freedom))] = interp_1d_array(
            ms_g,
            self.itp * self.freedom)
        csfull_g = np.zeros([4 * self.itp])
        csfull_g[int((self.itp / 2) * (1 - self.freedom)):int((self.itp / 2) * (1 + self.freedom))] = interp_1d_array(
            cs_g,
            self.itp * self.freedom)

        # Fix ends, interpolate: Red
        ms_r, cs_r = self.fix_ends(curve_r, bgcurve_r)
        msfull_r = np.zeros([4 * self.itp])
        msfull_r[int((self.itp / 2) * (1 - self.freedom)):int((self.itp / 2) * (1 + self.freedom))] = interp_1d_array(
            ms_r,
            self.itp * self.freedom)
        csfull_r = np.zeros([4 * self.itp])
        csfull_r[int((self.itp / 2) * (1 - self.freedom)):int((self.itp / 2) * (1 + self.freedom))] = interp_1d_array(
            cs_r,
            self.itp * self.freedom)

        # Input for curve fitter
        x = np.stack((np.append(bgcurve_g, bgcurve_r), msfull_g, csfull_g, msfull_r, csfull_r), axis=0)

        # Fit gaussian to find offset
        try:
            popt, pcov = curve_fit(self.gaussian_plus_2col, x, np.append(curve_g, curve_r),
                                   bounds=([(self.itp / 2) * (1 - self.freedom), 0, 20, 0, 20],
                                           [(self.itp / 2) * (1 + self.freedom), np.inf, 200, np.inf, 200]),
                                   p0=[self.itp / 2, 0, 100, 0, 100])

            o = (popt[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fix_ends(self, curve, bgcurve):
        """
        Calculates parameters to fit the ends of curve to bgcurve, across different alignments

        :param curve: signal curve
        :param bgcurve: background curve
        :return: ms, cs: end fitting parameters
        """

        x = (self.itp * self.freedom) / self.n_end_fits
        ms = np.zeros([self.n_end_fits])
        cs = np.zeros([self.n_end_fits])
        for l in range(self.n_end_fits):
            bgcurve_seg = bgcurve[
                          int(((self.itp / 2) * (1 - self.freedom)) + (x * l)): int(
                              (self.itp + (self.itp / 2) * (1 - self.freedom)) + (x * l))]
            line = np.polyfit(
                [np.mean(curve[:int(len(curve) * self.end_region)]),
                 np.mean(curve[int(len(curve) * (1 - self.end_region)):])],
                [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
                 np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])], 1)
            ms[l] = line[0]
            cs[l] = line[1]
        return ms, cs

    def gaussian_plus_2col(self, x, l, a_g, c_g, a_r, c_r):
        """
        Function used for fitting algorithm

        :param x: input 5 column array containing bgcurves (end on end) and end fit parameters
        :param l: offset parameter
        :param a_g: green gaussian height
        :param c_g: green gaussian width
        :param a_r: red gaussian height
        :param c_r: red gaussian width
        :return: y: curves (end on end)
        """

        y = np.zeros([2 * self.itp])

        # Green region
        g0 = a_g * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / c_g)
        g1 = a_g * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / c_g))
        g = np.append(g0, g1)
        y[:self.itp] = (x[0, int(l):int(l) + self.itp] + g - x[2, int(l)]) / x[1, int(l)]

        # Red region
        g0 = a_r * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / c_r)
        g1 = a_r * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / c_r))
        g = np.append(g0, g1)
        y[self.itp:2 * self.itp] = (x[0, (2 * self.itp) + int(l):int(l) + 3 * self.itp] + g - x[
            4, int(l)]) / x[3, int(l)]

        return y


class Segmenter3:
    def __init__(self, img, coors, mag=1, iterations=2, parallel=False, resolution=5, save=False,
                 direc=None, plot=False):

        # Inputs
        self.img = img
        self.coors = coors
        self.mag = mag
        self.iterations = iterations
        self.parallel = parallel
        self.resolution = resolution
        self.save = save
        self.direc = direc
        self.plot = plot
        self.itp = 1000
        self.thickness = 50
        self.rol_ave = 50
        self.end_region = 0.2
        self.n_end_fits = 50

        self.newcoors = coors

        # Warnings
        if self.save and self.direc is None:
            raise Exception('Must specify directory')

    def run(self):
        """
        Performs segmentation algorithm

        """

        for i in range(self.iterations):

            # Straighten
            straight = straighten(self.img, self.newcoors, int(self.thickness * self.mag))

            # Filter/smoothen/interpolate images
            straight = rolling_ave(interp(straight, self.itp), self.rol_ave)

            # Calculate offsets
            if self.parallel:
                offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(self.func)(straight, x) for x in range(len(straight[0, :]))))
            else:
                offsets = np.zeros(len(straight[0, :]))
                for x in range(len(straight[0, :])):
                    offsets[x] = self.func(straight, x)

            # Interpolate
            offsets = interp_1d_array(offsets, len(self.newcoors))

            # Offset coordinates
            self.newcoors = offset_coordinates(self.newcoors, offsets)

            # Filter
            self.newcoors = np.vstack(
                (savgol_filter(self.newcoors[:, 0], 19, 1, mode='wrap'),
                 savgol_filter(self.newcoors[:, 1], 19, 1, mode='wrap'))).T

        # Rotate
        self.newcoors = rotate_coors(self.newcoors)

        # Save
        if self.save:
            np.savetxt('%s/ROI_fitted.txt' % self.direc, self.newcoors, fmt='%.4f', delimiter='\t')

        # Plot
        if self.plot:
            plt.imshow(straighten(self.img, self.newcoors, self.thickness * self.mag), cmap='gray')
            plt.show()

    def func(self, straight, x):
        return ((self.itp / 2) - np.argmin(
            np.absolute(straight[:, x] - np.mean(
                [np.mean(straight[int(0.9 * self.itp):, x]), np.mean(straight[:int(0.1 * self.itp), x])]))) * (
                    len(straight[:, 0]) / self.itp)) / (self.itp / self.thickness)


################## ANALYSIS #################


class Analyser:
    """

    """

    def __init__(self, img, coors, direc, name, bg=None, mag=1, funcs='all', bounds=(0, 1), thickness=50):
        self.img = img
        self.coors = coors
        self.bg = bg
        self.direc = direc
        self.name = name
        self.mag = mag
        self.funcs = funcs
        self.bounds = bounds
        self.thickness = thickness
        self.img_straight = None
        self.res = self.Res()

    class Res:
        """

        """

        def __init__(self):
            self.mem = None
            self.spa = None
            self.cyt = None
            self.tot = None
            self.cse = None
            self.asi = None
            self.pro = None
            self.fbc = None
            self.ext = None

    def run(self):
        """

        """
        # Straighten image
        self.img_straight = straighten(self.img, self.coors, int(50 * self.mag))

        # Perform analysis
        if self.funcs is None:
            pass
        else:
            if self.funcs == 'all':
                self.funcs = ['mem', 'spa', 'cyt', 'tot', 'cse', 'asi', 'pro', 'fbc', 'ext']
            for func in self.funcs:
                getattr(self, func)()

        # Save results
        self.savedata()

    def mem(self):
        """

        """

        # Average
        profile = bounded_mean_2d(self.img_straight, self.bounds)
        profile = interp_1d_array(profile, 50)

        # Get cortical signal
        bg = fix_ends(profile, self.bg[25:75])
        self.res.mem = [np.trapz(profile - bg)]

    def spa(self):
        """
        Should add interpolation

        """

        # Smoothen
        img_straight = rolling_ave(self.img_straight, int(20 * self.mag))

        # Get cortical signals
        sigs = np.zeros([100])
        for x in range(100):
            profile = img_straight[:, int(np.linspace(0, len(img_straight[0, :]), 100)[x] - 1)]
            profile = interp_1d_array(profile, 50)
            bg2 = fix_ends(profile, self.bg[25:75])
            sigs[x] = np.trapz(profile - bg2)
        self.res.spa = sigs

    def cyt(self):
        """

        """
        img2 = polycrop(self.img, self.coors, -20 * self.mag)
        self.res.cyt = [np.nanmean(img2[np.nonzero(img2)])]

    def tot(self):
        """

        """

        img2 = polycrop(self.img, self.coors, 5 * self.mag)
        self.res.tot = [np.nanmean(img2[np.nonzero(img2)])]

    def cse(self, thickness=10, extend=1.5):
        """
        Returns cross section across the long axis of the embryo

        :param thickness:
        :param extend:
        """

        # PCA
        M = (self.coors - np.mean(self.coors.T, axis=1)).T
        [latent, coeff] = np.linalg.eig(np.cov(M))
        score = np.dot(coeff.T, M)

        # Find ends
        a = np.argmin(np.minimum(score[0, :], score[1, :]))
        b = np.argmax(np.maximum(score[0, :], score[1, :]))

        # Find posterior end
        dista = np.hypot((self.coors[0, 0] - self.coors[a, 0]), (self.coors[0, 1] - self.coors[a, 1]))
        distb = np.hypot((self.coors[0, 0] - self.coors[b, 0]), (self.coors[0, 1] - self.coors[b, 1]))

        if dista < distb:
            line0 = np.array([self.coors[a, :], self.coors[b, :]])
        else:
            line0 = np.array([self.coors[b, :], self.coors[a, :]])

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
            zvals[section, :] = map_coordinates(self.img.T, [xvalues, yvalues])

        self.res.cse = np.flipud(np.nanmean(zvals, 0))

    def asi(self):
        """

        """
        ant = bounded_mean(self.res.spa, (0.25, 0.75))
        post = bounded_mean(self.res.spa, (0.75, 0.25))
        self.res.asi = [(ant - post) / (2 * (ant + post))]

    def pro(self):
        """

        """
        if self.thickness != 50:
            img_straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
            self.res.pro = bounded_mean_2d(img_straight, self.bounds)
        else:
            self.res.pro = bounded_mean_2d(self.img_straight, self.bounds)

    def fbc(self):
        """

        """
        profile = bounded_mean_2d(self.img_straight, self.bounds)
        profile = interp_1d_array(profile, 50)
        self.res.fbc = fix_ends(profile, self.bg[25:75])

    def ext(self):
        """

        """
        img = polycrop(self.img, offset_coordinates(self.coors, 60 * self.mag)) - polycrop(self.img,
                                                                                           offset_coordinates(
                                                                                               self.coors,
                                                                                               10 * self.mag))
        self.res.ext = [np.nanmean(img[np.nonzero(img)])]

    def savedata(self):
        """

        """
        d = vars(self.res)
        for key, value in d.items():
            if value is not None:
                np.savetxt('%s/%s_%s.txt' % (self.direc, self.name, key), value, fmt='%.4f', delimiter='\t')


class ImportAll:
    """
    g_      green channel
    ga      green channel, af corrected
    gb      green channel, bg subtracted

    a_      af channel
    ab      af channel, bg subtracted

    r_      red channel
    rb      red channel, bg subtracted

    """

    def __init__(self, direc):
        self.g = Analyser.Res()
        self.a = Analyser.Res()
        self.r = Analyser.Res()
        self.c = Analyser.Res()
        self.b = Analyser.Res()

        a = glob.glob('%s/*.txt' % direc)
        for b in a:
            c = os.path.basename(os.path.normpath(b))[:-4]
            c = c.split('_')
            if hasattr(self, c[0]):
                setattr(getattr(self, c[0]), c[1], np.loadtxt(b))


class ImportAllBatch:
    """
    Imports all the data for a given group of embryos

    """

    def __init__(self, direcs):
        self.g = Analyser.Res()
        self.a = Analyser.Res()
        self.r = Analyser.Res()
        self.c = Analyser.Res()
        self.b = Analyser.Res()

        a = glob.glob('%s/*.txt' % direcs[0])
        for b in a:
            c = os.path.basename(os.path.normpath(b))[:-4]
            c = c.split('_')
            if hasattr(self, c[0]):
                setattr(getattr(self, c[0]), c[1], np.array([np.loadtxt(b)]))

        for d in direcs[1:]:
            a = glob.glob('%s/*.txt' % d)
            for b in a:
                c = os.path.basename(os.path.normpath(b))[:-4]
                c = c.split('_')
                if hasattr(self, c[0]):
                    setattr(getattr(self, c[0]), c[1],
                            np.append(getattr(getattr(self, c[0]), c[1]), np.array([np.loadtxt(b)]), axis=0))

                    # self.mean = ImportAll(None)
                    # for channel in vars(self.mean):
                    #     for analysis in vars(Analyser.Res()):
                    #         if getattr(getattr(self, channel), analysis) is not None:
                    #             setattr(getattr(self.mean, channel), analysis,
                    #                     np.mean(getattr(getattr(self, channel), analysis), axis=0))


def ImportAllBatch2(dest):
    """
    Creates dictionary of different groups of embryos

    """
    dic = {}
    for d in direcslist(dest):
        dic[os.path.basename(os.path.normpath(d))] = ImportAllBatch(func2(d))
    return dic


################### IMPORTERS ###################

class Importers:
    class Data0:
        def __init__(self, direc):
            self.direc = direc
            self.DIC = loadimage(sorted(glob.glob('%s/*DIC SP Camera*' % direc), key=len)[0])
            self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 535-50*' % direc), key=len)[0])
            self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
            self.RFP = loadimage(sorted(glob.glob('%s/*561 SP 630-75*' % direc), key=len)[0])
            self.ROI = np.loadtxt('%s/ROI.txt' % direc)

    class Data1:
        def __init__(self, direc):
            self.direc = direc
            self.DIC = None
            self.GFP = loadimage(sorted(glob.glob('%s/*GFP*' % direc), key=len)[0])
            self.AF = loadimage(sorted(glob.glob('%s/*AF*' % direc), key=len)[0])
            self.RFP = loadimage(sorted(glob.glob('%s/*PAR2*' % direc), key=len)[0])
            self.ROI = np.loadtxt('%s/ROI.txt' % direc)

    class Data2:
        def __init__(self, direc):
            self.direc = direc
            self.DIC = None
            self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 525-50*' % direc), key=len)[0])
            self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
            self.RFP = loadimage(sorted(glob.glob('%s/*561 SP 630-75*' % direc), key=len)[0])
            self.ROI = np.loadtxt('%s/ROI.txt' % direc)

    class Data3:
        def __init__(self, direc):
            self.direc = direc
            self.DIC = loadimage(sorted(glob.glob('%s/*DIC SP Camera*' % direc), key=len)[0])
            self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 535-50*' % direc), key=len)[0])
            self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
            self.RFP = None
            self.ROI = np.loadtxt('%s/ROI.txt' % direc)

    class Data4:
        def __init__(self, direc):
            self.direc = direc
            self.DIC = None
            self.GFP = loadimage(sorted(glob.glob('%s/*GFP*' % direc), key=len)[0])
            self.AF = loadimage(sorted(glob.glob('%s/*AF*' % direc), key=len)[0])
            self.RFP = None
            self.ROI = np.loadtxt('%s/ROI.txt' % direc)

    class Data5:
        def __init__(self, direc):
            self.direc = direc
            self.DIC = None
            self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 525-50*' % direc), key=len)[0])
            self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
            self.RFP = None
            self.ROI = np.loadtxt('%s/ROI.txt' % direc)


########################### MISC ############################


def fix_ends(curve, bgcurve):
    """
    Used for background subtraction. Returns fitted bgcurve which can then be subtracted from the signal curve
    Bg fitted by fixing ends

    :param curve:
    :param bgcurve:
    :return:
    """

    # Fix ends
    line = np.polyfit(
        [np.mean(bgcurve[:int(len(bgcurve) * 0.2)]), np.mean(bgcurve[int(len(bgcurve) * 0.8):])],
        [np.mean(curve[:int(len(curve) * 0.2)]), np.mean(curve[int(len(curve) * 0.8):])], 1)

    # Create new bgcurve
    curve2 = bgcurve * line[0] + line[1]

    return curve2


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


def interp_1d_array(arr, n):
    return np.interp(np.linspace(0, len(arr), n), np.array(range(len(arr))), arr)


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
        a = map_coordinates(img.T, [sectioncoors[:, 0], sectioncoors[:, 1]])
        a[a == 0] = np.mean(a)
        img2[section, :] = a
    return img2


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


def interp(img, n):
    """
    Interpolates values along y axis for each x value
    :param img:
    :param n:
    :return:
    """

    interped = np.zeros([n, len(img[0, :])])
    for x in range(len(img[0, :])):
        interped[:, x] = interp_1d_array(img[:, x], n)

    return interped


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
                ave[:, x] = np.mean(img[:, starts[x]:ends[x]], axis=1)
            else:
                ave[:, x] = np.mean(np.append(img[:, starts[x]:], img[:, :ends[x]], axis=1), axis=1)
        return ave


def bounded_mean(array, bounds):
    if bounds[0] < bounds[1]:
        mean = np.mean(array[int(len(array) * bounds[0]): int(len(array) * bounds[1] + 1)])
    else:
        mean = np.mean(np.hstack((array[:int(len(array) * bounds[1] + 1)], array[int(len(array) * bounds[0]):])))
    return mean


def bounded_mean_2d(array, bounds):
    if bounds[0] < bounds[1]:
        mean = np.mean(array[:, int(len(array[0, :]) * bounds[0]): int(len(array[0, :]) * bounds[1])], 1)
    else:
        mean = np.mean(
            np.hstack((array[:, :int(len(array[0, :]) * bounds[1])], array[:, int(len(array[0, :]) * bounds[0]):])), 1)
    return mean


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


#################### Used in scripts only#######################


def af_subtraction(ch1, ch2, settings):
    af = settings.m * ch2 + settings.c
    signal = ch1 - af
    return signal


def bg_subtraction(img, coors, mag):
    bg = np.mean(straighten(img, offset_coordinates(coors, 50 * mag), int(50 * mag)))
    return img - bg


def bg(img, coors, mag):
    a = polycrop(img, offset_coordinates(coors, 60 * mag)) - polycrop(img, offset_coordinates(coors, 10 * mag))
    return np.nanmean(a[np.nonzero(a)])


def normalise(instance1, instance2):
    """
    Creates new class, by dividing objects in class instance 1 by objects in class instance 2

    :param instance1:
    :param instance2:
    :return:
    """

    norm = copy.deepcopy(instance1)
    for o in vars(instance1):
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
    zvals = np.fliplr(zvals)

    return zvals


def copy_data(list, dest):
    """

    """

    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    for v in list:
        shutil.copytree('%s/%s' % (ddirec, v), '%s/%s' % (dest, v))


def append_batch(prefix, l):
    a = [None] * len(l)
    for i, c in enumerate(l):
        a[i] = '%s%s' % (prefix, c)
    return a


def norm_to_bounds(array, bounds=(0, 1), percentile=10):
    line = np.polyfit([np.percentile(array, percentile), np.percentile(array, 100 - percentile)],
                      [bounds[0], bounds[1]],
                      1)
    return array * line[0] + line[1]


def setup(dict, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    for key, value in dict.items():
        os.mkdir('%s/%s' % (dest, key))
        for v in value:
            shutil.copytree('%s/%s' % (ddirec, v), '%s/%s/%s' % (dest, key, v))


def func(dest):
    dlist = []
    for i in direcslist(dest):
        for j in direcslist(i):
            dlist.extend(direcslist(j))
    return dlist


def func2(dest):
    dlist = []
    for j in direcslist(dest):
        dlist.extend(direcslist(j))
    return dlist


##############################################

# Analysis is v wasteful because straightening images for every function
# Polyfit: must be quicker way
# Need to refine parameters for straightening alg: may be causing problems
# A way of specifying in the script which embryos to exclude
# Adapt code to allow different bgcurves for green and cherry
# Change structure so you first define the options for all jobs (maybe save as a huge spreadsheet), then loop through this