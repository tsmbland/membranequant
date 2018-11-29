from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, differential_evolution
from scipy.misc import toimage
import cv2
from scipy.ndimage.interpolation import map_coordinates
from joblib import Parallel, delayed
import multiprocessing
import math
from scipy.interpolate import splprep, splev

"""
Functions for segmentation and quantification of images

To do:
- ipynbs for segmentation example
- change mag to 60
- make more use of norm_coors function
- commit
- tidy test datasets (include af-corrected image?), write info file
- change segmenter 1 and 2 to differential_evolution?
- write readme

"""


############### SEGMENTATION ################


class SegmenterGeneric:
    """
    Parent class for segmentation subclasses
    Subclasses add calc_offsets method

    Input data:
    img             image to display if plot = TRUE
    coors           original coordinates

    Input parameters:
    mag             magnification (1 = 60x)
    iterations      number of times to run algorithm
    periodic        True if input coordinates form a closed polygon
    parallel        True: preform segmentation in parallel
    resolution      will perform fitting algorithm at gaps set by this, interpolating between
    save            True: will save output coordinates as .txt file
    direc           Directory to save coordinates to
    plot            True: will display output coordinates overlaid onto img

    Output:
    newcoors        New coordinates resulting from segmentation

    """

    def __init__(self, img, coors=None, mag=1, iterations=3, periodic=True, parallel=False, resolution=5):

        # Inputs
        self.img = img
        self.coors = coors
        self.mag = mag
        self.iterations = iterations
        self.parallel = parallel
        self.resolution = resolution
        self.periodic = periodic
        self.method = None

        # Outputs
        self.newcoors = None

    def run(self):
        """
        Performs segmentation algorithm

        """

        self.newcoors = self.coors

        # Interpolate coors to one pixel distance between points
        self.newcoors = self.interp_coors(self.newcoors)

        for i in range(self.iterations):

            # Calculate offsets
            offsets = self.calc_offsets()

            # Interpolate nans
            nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
            offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])

            # Interpolate
            offsets = interp_1d_array(offsets, len(self.newcoors))

            # Offset coordinates
            self.newcoors = offset_coordinates(self.newcoors, offsets)

            # Filter
            if self.periodic:
                self.newcoors = np.vstack(
                    (savgol_filter(self.newcoors[:, 0], 19, 1, mode='wrap'),
                     savgol_filter(self.newcoors[:, 1], 19, 1, mode='wrap'))).T
            elif not self.periodic:
                self.newcoors = np.vstack(
                    (savgol_filter(self.newcoors[:, 0], 19, 1, mode='nearest'),
                     savgol_filter(self.newcoors[:, 1], 19, 1, mode='nearest'))).T

            # Interpolate to one px distance between points
            self.newcoors = self.interp_coors(self.newcoors)

        # Rotate
        if self.periodic:
            self.newcoors = self.rotate_coors(self.newcoors)

    def save(self, direc):
        np.savetxt(direc, self.newcoors, fmt='%.4f', delimiter='\t')

    def plot(self):
        """

        """
        plt.imshow(self.img, cmap='gray', vmin=0)
        plt.plot(self.newcoors[:, 0], self.newcoors[:, 1], c='r')
        plt.scatter(self.newcoors[0, 0], self.newcoors[0, 1], c='r')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    @staticmethod
    def rotate_coors(coors):
        """
        Rotates coordinate array so that most posterior point is at the beginning

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

    @staticmethod
    def interp_coors(coors):
        """
        Interpolates coordinates to one pixel distances (or as close as possible to one pixel)
        Linear interpolation

        :param coors:
        :return:
        """

        c = np.append(coors, [coors[0, :]], axis=0)
        distances = ((np.diff(c[:, 0]) ** 2) + (np.diff(c[:, 1]) ** 2)) ** 0.5
        distances_cumsum = np.append([0], np.cumsum(distances))
        px = sum(distances) / round(sum(distances))  # effective pixel size
        newpoints = np.zeros((int(round(sum(distances))), 2))
        newcoors_distances_cumsum = 0

        for d in range(int(round(sum(distances)))):
            index = sum(distances_cumsum - newcoors_distances_cumsum <= 0) - 1
            newpoints[d, :] = (
                coors[index, :] + ((newcoors_distances_cumsum - distances_cumsum[index]) / distances[index]) * (
                    c[index + 1, :] - c[index, :]))
            newcoors_distances_cumsum += px

        return newpoints

    def fit_spline(self, coors):
        """
        Fits a spline to points specifying the coordinates of the cortex, then interpolates to pixel distances

        :param coors:
        :return:
        """

        # Append the starting x,y coordinates
        x = np.r_[coors[:, 0], coors[0, 0]]
        y = np.r_[coors[:, 1], coors[0, 1]]

        # Fit spline
        tck, u = splprep([x, y], s=0, per=True)

        # Evaluate spline
        xi, yi = splev(np.linspace(0, 1, 1000), tck)

        # Interpolate
        return self.interp_coors(np.vstack((xi, yi)).T)

    class ROI:
        def __init__(self):
            self.xpoints = []
            self.ypoints = []
            self.fig = plt.gcf()
            self.ax = plt.gca()
            self.point0 = None
            self.points = None

            self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
            self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
            plt.show()

        def button_press_callback(self, event):
            self.xpoints.extend([event.xdata])
            self.ypoints.extend([event.ydata])
            self.display_points()
            self.fig.canvas.draw()

        def key_press_callback(self, event):
            if event.key == 'backspace':
                if len(self.xpoints) != 0:
                    self.xpoints = self.xpoints[:-1]
                    self.ypoints = self.ypoints[:-1]
                self.display_points()
                self.fig.canvas.draw()
            if event.key == 'enter':
                plt.close()

        def display_points(self):
            try:
                self.point0.remove()
                self.points.remove()
            except (ValueError, AttributeError) as error:
                pass
            if len(self.xpoints) != 0:
                self.points = self.ax.scatter(self.xpoints, self.ypoints, c='b')
                self.point0 = self.ax.scatter(self.xpoints[0], self.ypoints[0], c='r')

    def def_ROI(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.img, cmap='gray', vmin=0)
        ax.text(0.03, 0.88,
                'Specify ROI clockwise from the posterior (4 points minimum)'
                '\nBACKSPACE to remove last point'
                '\nPress ENTER when done',
                color='white',
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        roi = self.ROI()
        self.coors = self.fit_spline(np.vstack((roi.xpoints, roi.ypoints)).T)


class Segmenter1(SegmenterGeneric):
    """

    Single channel segmentation, based on background cytoplasmic curves


    Input data:
    img             image
    bgcurve         background curve. Must be 2* wider than the eventual profile for img
    coors           original coordinates

    Parameters:
    freedom         0 = no freedom, 1 = max freedom
    it              Interpolation of profiles/bgcurves
    thickness       thickness of straightened images
    rol_ave         sets the degree of image smoothening
    end_region      for end fitting
    n_end_fits      number of end fits to perform, will interpolate

    """

    def __init__(self, img, bgcurve, coors=None, mag=1, iterations=3, parallel=False, resolution=5, freedom=0.3,
                 periodic=True):

        super().__init__(img, coors, mag, iterations, periodic, parallel, resolution)

        # Inputs
        self.bgcurve = bgcurve
        self.freedom = freedom
        self.itp = 1000
        self.thickness = 50
        self.rol_ave = 50
        self.end_region = 0.2
        self.n_end_fits = 50

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.newcoors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        bgcurve = interp_1d_array(self.bgcurve, 2 * self.itp)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_background)(straight[:, x * int(self.mag * self.resolution)], bgcurve)
                for x in range(len(straight[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.fit_background(straight[:, x * int(self.mag * self.resolution)], bgcurve)
        return offsets

    def fit_background(self, curve, bgcurve):
        """
        Takes cross sections from images and finds optimal offset for alignment

        :param curve: signal curve from cross section of straightened img_g
        :param bgcurve: background curve
        :return: o: offset value for this section
        """

        # Input for curve fitter
        x = np.stack((bgcurve, (np.hstack((curve, np.zeros([self.itp]))))), axis=0)

        # Fit gaussian to find offset
        try:
            popt, pcov = curve_fit(self.func, x, curve,
                                   bounds=([(self.itp / 2) * (1 - self.freedom), 0, 20],
                                           [(self.itp / 2) * (1 + self.freedom), np.inf, 200]),
                                   p0=[self.itp / 2, 0, 100])

            if math.isclose(popt[0], (self.itp / 2) * (1 - self.freedom)) or math.isclose(popt[0], (self.itp / 2) * (
                        1 + self.freedom)):
                o = np.nan
            else:
                o = (popt[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def func(self, x, l, a, c):
        """
        Fits profile to bgcurve + gaussian

        """

        profile = x[1, :self.itp]
        bgcurve_seg = x[0, int(l):int(l) + self.itp]

        line = np.polyfit(
            [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
             np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])],
            [np.mean(profile[:int(len(profile) * self.end_region)]),
             np.mean(profile[int(len(profile) * (1 - self.end_region)):])], 1)

        p0 = line[0]
        p1 = line[1]

        g0 = a * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / c)
        g1 = a * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / c))
        g = np.append(g0, g1)
        y = p0 * (x[0, int(l):int(l) + self.itp] + g) + p1

        return y


class Segmenter2(SegmenterGeneric):
    """

    Two-channel segmentation, using two background curves

    Input data:
    img_g           green channel image
    img_r           red channel image
    bgcurve_g       green background curve. Must be 2* wider than the eventual profile for img_r
    bgcurve_r       red background curve. Must be 2* wider than the eventual profile for img_r
    coors           original coordinates

    Input parameters:
    freedom         0 = no freedom, 1 = max freedom
    it              Interpolation of profiles/bgcurves
    thickness       thickness of straightened images
    rol_ave         sets the degree of image smoothening
    end_region      for end fitting
    n_end_fits      number of end fits to perform, will interpolate

    """

    def __init__(self, img_g, img_r, bg_g, bg_r, coors=None, mag=1, iterations=3, parallel=False, resolution=5,
                 freedom=0.3, periodic=True):

        super().__init__(img_g, coors, mag, iterations, periodic, parallel, resolution)

        self.img_g = img_g
        self.img_r = img_r
        self.bg_g = bg_g
        self.bg_r = bg_r
        self.freedom = freedom
        self.itp = 1000
        self.thickness = 50
        self.rol_ave = 50
        self.end_region = 0.2
        self.n_end_fits = 20

    def calc_offsets(self):
        """

        """
        # Straighten
        straight_g = straighten(self.img_g, self.newcoors, int(self.thickness * self.mag))
        straight_r = straighten(self.img_r, self.newcoors, int(self.thickness * self.mag))

        # Smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.itp), int(self.rol_ave * self.mag), self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        bgcurve_g = interp_1d_array(self.bg_g, 2 * self.itp)
        bgcurve_r = interp_1d_array(self.bg_r, 2 * self.itp)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_background_2col)(straight_g[:, x * int(self.mag * self.resolution)],
                                                  straight_r[:, x * int(self.mag * self.resolution)], bgcurve_g,
                                                  bgcurve_r)
                for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.fit_background_2col(straight_g[:, x * int(self.mag * self.resolution)],
                                                      straight_r[:, x * int(self.mag * self.resolution)], bgcurve_g,
                                                      bgcurve_r)
        return offsets

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

            if math.isclose(popt[0], (self.itp / 2) * (1 - self.freedom)) or math.isclose(popt[0], (self.itp / 2) * (
                        1 + self.freedom)):
                o = np.nan
            else:
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
        :param a_g: green exponential height
        :param c_g: green exponential width
        :param a_r: red exponential height
        :param c_r: red exponential width
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


class Segmenter3(SegmenterGeneric):
    """

    Segments embryos to midpoint of decline

    Used to segment embryos without a background curve
    e.g. for generating background curves


    """

    def __init__(self, img, coors=None, mag=1, iterations=2, parallel=False, resolution=5, periodic=True):

        super().__init__(img, coors, mag, iterations, periodic, parallel, resolution)

        self.itp = 1000
        self.thickness = 50
        self.rol_ave = 50

    def calc_offsets(self):
        """

        """
        # Straighten
        straight = straighten(self.img, self.newcoors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), self.rol_ave, self.periodic)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.func)(straight, x) for x in range(len(straight[0, :]))))
        else:
            offsets = np.zeros(len(straight[0, :]))
            for x in range(len(straight[0, :])):
                offsets[x] = self.func(straight, x)

        return offsets

    def func(self, straight, x):
        """

        """
        return ((self.itp / 2) - np.argmin(
            np.absolute(straight[:, x] - np.mean(
                [np.mean(straight[int(0.9 * self.itp):, x]), np.mean(straight[:int(0.1 * self.itp), x])]))) * (
                    len(straight[:, 0]) / self.itp)) / (self.itp / self.thickness)


class Segmenter4(SegmenterGeneric):
    """

    Segments embryos to peak

    Used to segment embryos without a background curve
    e.g. for generating background curves


    """

    def __init__(self, img, coors=None, mag=1, iterations=2, parallel=False, resolution=5, periodic=True):

        super().__init__(img, coors, mag, iterations, periodic, parallel, resolution)

        self.itp = 1000
        self.thickness = 50
        self.rol_ave = 50

    def calc_offsets(self):
        """

        """
        # Straighten
        straight = straighten(self.img, self.newcoors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), self.rol_ave, self.periodic)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.func)(straight, x) for x in range(len(straight[0, :]))))
        else:
            offsets = np.zeros(len(straight[0, :]))
            for x in range(len(straight[0, :])):
                offsets[x] = self.func(straight, x)

        return offsets

    def func(self, straight, x):
        """

        """
        return ((self.itp / 2) - np.argmax(straight[:, x]) * (len(straight[:, 0]) / self.itp)) / (
            self.itp / self.thickness)


class Segmenter5(SegmenterGeneric):
    """
    Fit profiles to cytoplasmic background + membrane background

    """

    def __init__(self, img, cytbg, membg, coors=None, mag=1, iterations=2, parallel=False, resolution=5, freedom=0.3,
                 periodic=True):
        super().__init__(img, coors, mag, iterations, periodic, parallel, resolution)

        self.cytbg = cytbg
        self.membg = membg
        self.freedom = freedom
        self.thickness = 50
        self.itp = 1000
        self.rol_ave = 50
        self.end_region = 0.2

        self._profile = None
        self._cytbg = None
        self._membg = None

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.newcoors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)
        membg = interp_1d_array(self.membg, 2 * self.itp)

        # Fit
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile)(straight[:, x * int(self.mag * self.resolution)], cytbg, membg)
                for x in range(len(straight[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.fit_profile(straight[:, x * int(self.mag * self.resolution)], cytbg, membg)

        return offsets

    def fit_profile(self, profile, cytbg, membg):
        """
        Takes cross sections from images and finds optimal offset for alignment

        """

        self._profile = profile
        self._cytbg = cytbg
        self._membg = membg

        res = differential_evolution(self.mse, bounds=((350, 650), (-1, 5)))
        o = (res.x[0] - self.itp / 2) / (self.itp / self.thickness)

        return o

    def fix_ends(self, profile, cytbg, membg, a):
        """

        """
        pa = np.mean(profile[:int(len(profile) * self.end_region)])
        px = np.mean(profile[int(len(profile) * (1 - self.end_region)):])
        b1a = np.mean(cytbg[:int(len(cytbg) * self.end_region)])
        b1x = np.mean(cytbg[int(len(cytbg) * (1 - self.end_region)):])
        b2a = np.mean(membg[:int(len(membg) * self.end_region)])
        b2x = np.mean(membg[int(len(membg) * (1 - self.end_region)):])
        m1 = (px - pa) / (b1x - b1a + a * (b2x - b2a))
        c1 = pa - (m1 * b1a)
        c2 = - m1 * a * b2a

        return m1, c1, c2

    def mse(self, l_a):
        """

        """
        y = self.total_curve(l_a)
        return np.mean((self._profile - y) ** 2)

    def total_curve(self, l_a):
        """

        """

        l, a = l_a
        cytbg = self._cytbg[int(l):int(l) + self.itp]
        membg = self._membg[int(l):int(l) + self.itp]
        m1, c1, c2 = self.fix_ends(self._profile, cytbg, membg, a)

        # Profile estimate
        return m1 * (cytbg + a * membg) + c1 + c2

    def cyt_only(self, l_a):
        """

        """
        l, a = l_a
        cytbg = self._cytbg[int(l):int(l) + self.itp]
        membg = self._membg[int(l):int(l) + self.itp]
        m1, c1, c2 = self.fix_ends(self._profile, cytbg, membg, a)

        # Profile estimate
        return m1 * cytbg + c1


################## ANALYSIS #################


class MembraneQuant:
    """
    Fit profiles to cytoplasmic background + membrane background

    WORK IN PROGRESS

    """

    def __init__(self, img, cytbg, membg, coors=None, mag=1):
        self.img = img
        self.coors = coors
        self.mag = mag
        self.cytbg = cytbg
        self.membg = membg
        self.thickness = 50
        self.itp = 1000
        self.rol_ave = 50
        self.end_region = 0.2

    def run(self):
        self.img_straight = straighten(self.img, self.coors, int(50 * self.mag))

        # Rolling ave
        img_straight = rolling_ave_2d(self.img_straight, int(self.mag * 20))

        plt.imshow(img_straight, cmap='gray', vmin=0)
        plt.show()

        cytbg = self.cytbg[25:75]
        membg = self.membg[25:75]

        # Get cortical signals
        fbc_spa = np.zeros(self.img_straight.shape)
        mem_spa = np.zeros(self.img_straight.shape)

        # fits = np.zeros(len(img_straight[0, :]))
        for i in range(len(img_straight[0, :])):
            # Input for curve fitter
            x = np.stack((membg, cytbg, img_straight[:, i]), axis=0)

            # Fit to find offset
            popt, pcov = curve_fit(self.func, x, img_straight[:, i], p0=0)
            m1, c1, c2 = self.fix_ends(popt[0], x)

            fbc_spa[:, i] = m1 * cytbg
            mem_spa[:, i] = m1 * popt[0] * membg

        plt.imshow(mem_spa + fbc_spa, cmap='gray', vmin=0)
        plt.show()

        plt.imshow(mem_spa, cmap='gray', vmin=0)
        plt.show()

        plt.imshow(fbc_spa, cmap='gray', vmin=0)
        plt.show()

    def fix_ends(self, a, x):
        """
        For a given value of a, calculates parameters m1, c1 and c2, which ensure ends fit to profile

        """

        profile = x[2, :]
        cytbg = x[1, :]
        membg = x[0, :]

        pa = np.mean(profile[:int(len(profile) * self.end_region)])
        px = np.mean(profile[int(len(profile) * (1 - self.end_region)):])
        b1a = np.mean(cytbg[:int(len(cytbg) * self.end_region)])
        b1x = np.mean(cytbg[int(len(cytbg) * (1 - self.end_region)):])
        b2a = np.mean(membg[:int(len(membg) * self.end_region)])
        b2x = np.mean(membg[int(len(membg) * (1 - self.end_region)):])

        g = a
        m1 = (px - pa) / (b1x - b1a + g * (b2x - b2a))
        c1 = pa - (m1 * b1a)
        c2 = - m1 * g * b2a

        return m1, c1, c2

    def func(self, x, a):
        """
        Used to fit paramter a, which specifies how much of the profile is made up of membrane contribution

        """

        m1, c1, c2 = self.fix_ends(a, x)
        y = m1 * (x[1, :] + a * x[0, :]) + c1 + c2

        return y

    def func2(self, x, a):
        m1, c1, c2 = self.fix_ends(a, x)
        y = m1 * x[1, :] + c1

        return y


class Quantifier:
    """

    WORK IN PROGRESS

    Performs quantification on an image
    Functions to perform specified by funcs argument
    Results saved as individual .txt files for each quantification

    Input data:
    img                image (as array)
    coors              coordinates specifying the location of the cortex
    bg                 background curve

    Arguments:
    direc              directory to save results
    name               name tag for results files (e.g. g -> g_mem.txt)
    mag                magnification of the image (1 = 60x)
    funcs              functions to perform


    """

    def __init__(self, img, coors, bg=None, mag=1):
        self.img = img
        self.coors = coors
        self.bg = bg
        self.mag = mag
        self.thickness = 50
        self.img_straight = None
        self.res = self.Res()

    class Res:
        """
        Holds the results from quantifications

        """

        def __init__(self):
            self.mema = None
            self.proa = None
            self.fbca = None
            self.memp = None
            self.prop = None
            self.fbcp = None
            self.memw = None
            self.prow = None
            self.fbcw = None
            self.spa = None
            self.cyt = None
            self.tot = None
            self.cse = None
            self.asi = None
            self.ext = None
            self.mbm = None

    def run(self, funcs):
        """

        """
        # Straighten image
        self.img_straight = straighten(self.img, self.coors, int(50 * self.mag))

        # Perform analysis
        if funcs is None:
            pass
        else:
            if funcs == 'all':
                funcs = ['mem', 'spa', 'cyt', 'mbm', 'tot', 'cse', 'asi', 'ext']
            for func in funcs:
                getattr(self, func)()

    def save(self, direc):
        """

        """
        d = vars(self.res)
        for key, value in d.items():
            if value is not None:
                np.savetxt('%s/%s.txt' % (direc, key), value, fmt='%.4f', delimiter='\t')

    def mem(self):
        """
        Average of the cortical intensity
        Performed in anterior, posterior and whole cell

        """

        self.res.proa = bounded_mean_2d(self.img_straight, [0.4, 0.6])
        self.res.prop = bounded_mean_2d(self.img_straight, [0.9, 0.1])
        self.res.prow = bounded_mean_2d(self.img_straight, [0, 1])

        self.res.fbca = fix_ends(bounded_mean_2d(self.img_straight, [0.4, 0.6]),
                                 interp_1d_array(self.bg[25:75], int(50 * self.mag)))
        self.res.fbcp = fix_ends(bounded_mean_2d(self.img_straight, [0.9, 0.1]),
                                 interp_1d_array(self.bg[25:75], int(50 * self.mag)))
        self.res.fbcw = fix_ends(bounded_mean_2d(self.img_straight, [0, 1]),
                                 interp_1d_array(self.bg[25:75], int(50 * self.mag)))

        self.res.mema = [np.trapz(self.res.proa - self.res.fbca)]
        self.res.memp = [np.trapz(self.res.prop - self.res.fbcp)]
        self.res.memw = [np.trapz(self.res.prow - self.res.fbcw)]

    # def pro(self):
    #     """
    #
    #     """
    #     img = straighten(self.img, self.coors, int(self.thickness * self.mag))
    #     self.res.proa = bounded_mean_2d(img, [0.4, 0.6])
    #     self.res.prop = bounded_mean_2d(img, [0.9, 0.1])
    #     self.res.prow = bounded_mean_2d(img, [0, 1])

    def spa(self):
        """
        Spatial profile of intensity around the cortex
        Change to 10 pixel patches instead, then interpolate to 1000

        """

        # Interpolate
        img_straight = rolling_ave_2d(self.img_straight, int(self.mag * 10))

        # Get cortical signals
        sigs = np.zeros([len(img_straight[0, :])])
        fbc_spa = np.zeros(self.img_straight.shape)
        mem_spa = np.zeros(self.img_straight.shape)
        for x in range(len(img_straight[0, :])):
            profile = img_straight[:, x]
            bg2 = fix_ends(profile, interp_1d_array(self.bg[25:75], len(profile)))
            fbc_spa[:, x] = bg2
            mem_spa[:, x] = profile - bg2
            sigs[x] = np.trapz(profile - bg2)
        self.res._fbc_spa = fbc_spa
        self.res._mem_spa = mem_spa
        self.res._spa = sigs
        self.res.spa = interp_1d_array(sigs, 100)

    def cyt(self):
        """
        Average cytoplasmic concentration

        """
        img2 = polycrop(self.img, self.coors, -20 * self.mag)
        self.res.cyt = [np.nanmean(img2[np.nonzero(img2)])]

    def mbm(self):
        """
        Membrane mean signal

        """
        if not hasattr(getattr(self, 'res'), '_spa'):
            self.spa()
        nc = norm_coors(self.coors)
        self.res.mbm = [np.average(self.res._spa, weights=abs(nc[:, 1]))]

    def tot(self):
        """
        Total signal over entire embryo

        """

        if self.res.cyt is None:
            self.cyt()
        if self.res.mbm is None:
            self.mbm()

        # Normalise coordinates
        nc = norm_coors(self.coors)

        # Calculate volume
        r1 = max(nc[:, 0]) - min(nc[:, 0]) / 2
        r2 = max(nc[:, 1]) - min(nc[:, 1]) / 2
        vol = 4 / 3 * np.pi * r2 * r2 * r1

        # Calculate surface area
        e = (1 - (r2 ** 2) / (r1 ** 2)) ** 0.5
        sa = 2 * np.pi * r2 * r2 * (1 + (r1 / (r2 * e)) * np.arcsin(e))

        self.res.tot = [self.res.cyt[0] + ((sa / vol) * self.res.mbm[0])]

    def cse(self, thickness=10, extend=1.5):
        """
        Returns cross section across the long axis of the embryo

        :param thickness: thickness of cross section to average over
        :param extend: how much to extend line over length of embryo (1 = no extension)
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
        Asymmetry index

        """
        if self.res.spa is None:
            self.spa()
        ant = bounded_mean_1d(self.res.spa, (0.25, 0.75))
        post = bounded_mean_1d(self.res.spa, (0.75, 0.25))
        self.res.asi = [(ant - post) / (2 * (ant + post))]

    def ext(self):
        """
        Mean concentration in a 50 pixel thick region surrounding the embryo

        """
        img = polycrop(self.img, self.coors, 60 * self.mag) - polycrop(self.img, self.coors, 10 * self.mag)
        self.res.ext = [np.nanmean(img[np.nonzero(img)])]


############### MISC FUNCTIONS ##############


def fix_ends(curve1, curve2):
    """
    Used for background subtraction. Returns fitted bgcurve which can then be subtracted from the signal curve
    Bg fitted by fixing ends

    Fixes ends of curve 2 to ends of curve 1

    :param curve1:
    :param curve2:
    :return:
    """

    # Fix ends
    line = np.polyfit(
        [np.mean(curve2[:int(len(curve2) * 0.2)]), np.mean(curve2[int(len(curve2) * 0.8):])],
        [np.mean(curve1[:int(len(curve1) * 0.2)]), np.mean(curve1[int(len(curve1) * 0.8):])], 1)

    # Create new bgcurve
    curve2 = curve2 * line[0] + line[1]

    return curve2


def polycrop(img, polyline, enlarge):
    """
    Crops image according to polyline coordinates
    Expand or contract selection with enlarge parameter

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


def straighten(img, coors, thickness):
    """
    Creates straightened image based on coordinates

    :param img:
    :param coors: Coordinates. Should be 1 pixel length apart in a loop
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


def offset_line(line, offset):
    """
    Moves a straight line of coordinates perpendicular to itself

    :param line:
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
    Extends a straight line of coordinates along itself

    Should adjust to allow shrinking
    :param line:
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


def bg_subtraction(img, coors, mag):
    """
    Subtracts the background from an image

    :param img:
    :param coors:
    :param mag:
    :return:
    """

    a = polycrop(img, coors, 60 * mag) - polycrop(img, coors, 10 * mag)
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


def norm_coors(coors):
    """
    Aligns coordinates to their long axis

    :param coors:
    :return:
    """

    # PCA
    M = (coors - np.mean(coors.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M).T

    # Find long axis
    if (max(score[0, :]) - min(score[0, :])) < (max(score[1, :]) - min(score[1, :])):
        score = np.fliplr(score)

    return score


# General functions

def interp_1d_array(array, n):
    """
    Interpolates a one dimensional array into n points

    :param array:
    :param n:
    :return:
    """

    return np.interp(np.linspace(0, len(array) - 1, n), np.array(range(len(array))), array)


def interp_2d_array(array, n, ax=0):
    """
    Interpolates values along y axis into n points, for each x value
    :param array:
    :param n:
    :param ax:
    :return:
    """

    if ax == 0:
        interped = np.zeros([n, len(array[0, :])])
        for x in range(len(array[0, :])):
            interped[:, x] = interp_1d_array(array[:, x], n)
        return interped
    elif ax == 1:
        interped = np.zeros([len(array[:, 0]), n])
        for x in range(len(array[:, 0])):
            interped[x, :] = interp_1d_array(array[x, :], n)
        return interped
    else:
        return None


def rolling_ave_1d(array, window, periodic=True):
    """

    :param array:
    :param window:
    :param periodic:
    :return:
    """
    if not periodic:
        ave = np.zeros([len(array)])
        ave[int(window / 2):-int(window / 2)] = np.convolve(array, np.ones(window) / window, mode='valid')
        return ave

    elif periodic:
        ave = np.zeros([len(array)])
        starts = np.append(range(len(array) - int(window / 2), len(array)),
                           range(len(array) - int(window / 2)))
        ends = np.append(np.array(range(int(np.ceil(window / 2)), len(array))),
                         np.array(range(int(np.ceil(window / 2)))))

        for x in range(len(array)):
            if starts[x] < x < ends[x]:
                ave[x] = np.mean(array[starts[x]:ends[x]])
            else:
                ave[x] = np.mean(np.append(array[starts[x]:], array[:ends[x]]))
        return ave


def rolling_ave_2d(array, window, periodic=True):
    """
    Returns rolling average across the x axis of an image (used for straightened profiles)

    :param array: image data
    :param window: number of pixels to average over. Odd number is best
    :param periodic: is true, rolls over at ends
    :return: ave
    """

    if not periodic:
        ave = np.zeros([len(array[:, 0]), len(array[0, :])])
        for y in range(len(array[:, 0])):
            ave[y, :] = np.convolve(array[y, :], np.ones(window) / window, mode='same')
        return ave

    elif periodic:
        ave = np.zeros([len(array[:, 0]), len(array[0, :])])
        starts = np.append(range(len(array[0, :]) - int(window / 2), len(array[0, :])),
                           range(len(array[0, :]) - int(window / 2)))
        ends = np.append(np.array(range(int(np.ceil(window / 2)), len(array[0, :]))),
                         np.array(range(int(np.ceil(window / 2)))))

        for x in range(len(array[0, :])):
            if starts[x] < x < ends[x]:
                ave[:, x] = np.mean(array[:, starts[x]:ends[x]], axis=1)
            else:
                ave[:, x] = np.mean(np.append(array[:, starts[x]:], array[:, :ends[x]], axis=1), axis=1)
        return ave


def bounded_mean_1d(array, bounds):
    """
    Averages 1D array over region specified by bounds

    Should add interpolation step first

    :param array:
    :param bounds:
    :return:
    """

    if bounds[0] < bounds[1]:
        mean = np.mean(array[int(len(array) * bounds[0]): int(len(array) * bounds[1] + 1)])
    else:
        mean = np.mean(np.hstack((array[:int(len(array) * bounds[1] + 1)], array[int(len(array) * bounds[0]):])))
    return mean


def bounded_mean_2d(array, bounds):
    """
    Averages 2D array in y dimension over region specified by bounds

    Should add axis parameter
    Should add interpolation step first

    :param array:
    :param bounds:
    :return:
    """

    if bounds[0] < bounds[1]:
        mean = np.mean(array[:, int(len(array[0, :]) * bounds[0]): int(len(array[0, :]) * bounds[1])], 1)
    else:
        mean = np.mean(
            np.hstack((array[:, :int(len(array[0, :]) * bounds[1])], array[:, int(len(array[0, :]) * bounds[0]):])), 1)
    return mean


def load_image(filename):
    """
    Given the filename of a TIFF, creates numpy array with pixel intensities

    :param filename:
    :return:
    """

    img = np.array(Image.open(filename), dtype=np.float64)
    img[img == 0] = np.nan
    return img


def saveimg(img, direc):
    """
    Saves 2D array as .tif file

    :param img:
    :param direc:
    :return:
    """

    im = Image.fromarray(img)
    im.save(direc)


def saveimg_jpeg(img, direc, cmin, cmax):
    """
    Saves 2D array as jpeg, according to min and max pixel intensities

    :param img:
    :param direc:
    :param cmin:
    :param cmax:
    :return:
    """

    a = toimage(img, cmin=cmin, cmax=cmax)
    a.save(direc)


def norm_to_bounds(array, bounds=(0, 1), percentile=10):
    """
    Normalise array to lie between two bounds

    :param array:
    :param bounds:
    :param percentile:
    :return:
    """

    line = np.polyfit([np.percentile(array, percentile), np.percentile(array, 100 - percentile)],
                      [bounds[0], bounds[1]],
                      1)
    return array * line[0] + line[1]
