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
- ipynbs for segmentation examples
- notebooks to explain methods
- make more use of norm_coors function (rotate_coors, cse)
- tidy test datasets (include af-corrected image?), write info file
- write readme
- function for reverse straightening
- incorporate uncertainty into fitting algorithms???
- interpolation needs to change with mag
- code needs to be massively simplified and made more consistent
    - eg. do everything at the start to account for differences in mag, thickness, itp...
- change fit type 2 funcs to y = m1 * cytbg + m2 * membg + c
- make number of iterations argument to segment function instead, and parallel
- parallel option for quantification, and resolution parameter
- add differential evolution method to classes that don't yet have one
- fit type classes for each profile (e.g, free cytbg offset, not free cytbg offset...) 

"""


############ PROFILE TYPES #############


class Profile1:
    """
    Parameters
    l = offset
    a = gaussian height
    w = gaussian width
    o = cytbg offset

    """

    def __init__(self, itp, thickness, end_region):
        self.itp = itp
        self.thickness = thickness
        self.end_region = end_region

    def total_profile(self, profile, cytbg, l, a, w, o):
        cytbg_offset = (self.itp / self.thickness) * o
        bgcurve_seg = cytbg[int(l + cytbg_offset):int(l + cytbg_offset) + self.itp]

        line = np.polyfit(
            [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
             np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])],
            [np.mean(profile[:int(len(profile) * self.end_region)]),
             np.mean(profile[int(len(profile) * (1 - self.end_region)):])], 1)

        p0 = line[0]
        p1 = line[1]

        g0 = a * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / w)
        g1 = a * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / w))
        g = np.append(g0, g1)
        y = p0 * (bgcurve_seg + g) + p1
        return y

    def cyt_profile(self, profile, cytbg, l, o, a=None, w=None):
        cytbg_offset = (self.itp / self.thickness) * o
        bgcurve_seg = cytbg[int(l + cytbg_offset):int(l + cytbg_offset) + self.itp]

        line = np.polyfit(
            [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
             np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])],
            [np.mean(profile[:int(len(profile) * self.end_region)]),
             np.mean(profile[int(len(profile) * (1 - self.end_region)):])], 1)

        p0 = line[0]
        p1 = line[1]
        y = p0 * bgcurve_seg + p1
        return y

    def mem_profile(self, profile, cytbg, l, a, w, o):
        cytbg_offset = (self.itp / self.thickness) * o
        bgcurve_seg = cytbg[int(l + cytbg_offset):int(l + cytbg_offset) + self.itp]

        line = np.polyfit(
            [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
             np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])],
            [np.mean(profile[:int(len(profile) * self.end_region)]),
             np.mean(profile[int(len(profile) * (1 - self.end_region)):])], 1)

        p0 = line[0]
        p1 = line[1]

        g0 = a * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / w)
        g1 = a * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / w))
        g = np.append(g0, g1)
        y = p0 * g
        return y


class Profile2:
    def __init__(self, itp, thickness, end_region):
        self.itp = itp
        self.thickness = thickness
        self.end_region = end_region

    def fix_ends(self, profile, cytbg, membg, a):
        """
        Needs to throw a warning if profile, cytbg and membg are different lengths

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

    def total_profile(self, profile, cytbg, membg, l, a, o):
        j = (self.itp / self.thickness) * o
        m1, c1, c2 = self.fix_ends(profile, cytbg[int(l + j):int(l + j) + self.itp], membg[int(l):int(l) + self.itp], a)
        return m1 * (cytbg[int(l + j):int(l + j) + self.itp] + a * membg[int(l):int(l) + self.itp]) + c1 + c2

    def cyt_profile(self, profile, cytbg, membg, l, a, o):
        j = (self.itp / self.thickness) * o
        m1, c1, c2 = self.fix_ends(profile, cytbg[int(l + j):int(l + j) + self.itp], membg[int(l):int(l) + self.itp], a)
        return m1 * cytbg[int(l + j):int(l + j) + self.itp] + c1


############### SEGMENTATION ################


# Parent class

class SegmenterParent:
    """
    Parent class for segmentation subclasses
    Subclasses add calc_offsets method

    Input data:
    img             image to display if plot = TRUE
    coors           original coordinates

    Input parameters:
    mag             magnification (1 = 60x)
    iterations      number of times to runR algorithm
    periodic        True if input coordinates form a closed polygon
    parallel        True: preform segmentation in parallel
    resolution      will perform fitting algorithm at gaps set by this, interpolating between
    save            True: will save output coordinates as .txt file
    direc           Directory to save coordinates to
    plot            True: will display output coordinates overlaid onto img

    """

    def __init__(self, img=None, img_g=None, img_r=None, coors=None, mag=1, iterations=3, periodic=True, parallel=False,
                 resolution=5, thickness=50):

        # Inputs
        self.img = img
        self.img_g = img_g
        self.img_r = img_r
        self.input_coors = coors
        self.coors = coors
        self.mag = mag
        self.iterations = iterations
        self.parallel = parallel
        self.resolution = resolution
        self.periodic = periodic
        self.method = None
        self.thickness = thickness

    def segment(self):
        """
        Performs segmentation algorithm

        """

        # Interpolate coors to one pixel distance between points
        self.coors = self.interp_coors(self.coors)

        for i in range(self.iterations):

            # Calculate offsets
            offsets = self.calc_offsets()

            # Interpolate nans
            nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
            offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])

            # Interpolate
            offsets = interp_1d_array(offsets, len(self.coors))

            # Offset coordinates
            self.coors = offset_coordinates(self.coors, offsets)

            # Filter
            if self.periodic:
                self.coors = np.vstack(
                    (savgol_filter(self.coors[:, 0], 19, 1, mode='wrap'),
                     savgol_filter(self.coors[:, 1], 19, 1, mode='wrap'))).T
            elif not self.periodic:
                self.coors = np.vstack(
                    (savgol_filter(self.coors[:, 0], 19, 1, mode='nearest'),
                     savgol_filter(self.coors[:, 1], 19, 1, mode='nearest'))).T

            # Interpolate to one px distance between points
            self.coors = self.interp_coors(self.coors)

        # Rotate
        if self.periodic:
            self.coors = self.rotate_coors(self.coors)

    def save(self, direc):
        np.savetxt(direc, self.coors, fmt='%.4f', delimiter='\t')

    def _plot(self):
        if self.img_g is not None:
            self.comp_plot(self.img_g, self.img_r)
        if self.img is not None:
            plt.imshow(self.img, cmap='gray', vmin=0)

    @staticmethod
    def comp_plot(g, r):
        """

        """
        rgb = np.dstack((r, g, np.zeros(g.shape)))
        cmax = max(rgb.flatten())
        rgb /= cmax
        rgb[rgb <= 0] = 0
        plt.imshow(rgb)

    def plot(self):
        """

        """
        self._plot()
        plt.plot(self.coors[:, 0], self.coors[:, 1], c='r')
        plt.scatter(self.coors[0, 0], self.coors[0, 1], c='r')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def plot_straight(self):
        """

        """
        if self.img_g is not None:
            self.comp_plot(straighten(self.img_g, self.coors, self.thickness * self.mag),
                           straighten(self.img_r, self.coors, self.thickness * self.mag))
        if self.img is not None:
            plt.imshow(straighten(self.img, self.coors, self.thickness * self.mag), cmap='gray', vmin=0)
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
        # ax.imshow(self.img, cmap='gray', vmin=0)
        self._plot()
        ax.text(0.03, 0.88,
                'Specify ROI clockwise from the posterior (4 points minimum)'
                '\nBACKSPACE to remove last point'
                '\nPress ENTER when done',
                color='white',
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        roi = self.ROI()
        # self.coors = self.fit_spline(np.vstack((roi.xpoints, roi.ypoints)).T)
        self.coors = np.vstack((roi.xpoints, roi.ypoints)).T


# Fit type 1

class Segmenter1aSingle(SegmenterParent, Profile1):
    """

    Single channel segmentation, based on background cytoplasmic curves

    Cytbg offset fixed
    Gaussian width free

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

    """

    def __init__(self, img, cytbg, coors=None, mag=1, iterations=3, parallel=False, resolution=5, freedom=0.3,
                 periodic=True, thickness=50, itp=1000, rol_ave=50, cytbg_offset=4, end_region=0.2):
        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel,
                                 resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg = cytbg
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('bgcurve must be twice as wide as thickness')

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight[:, x * int(self.mag * self.resolution)], cytbg)
                for x in range(len(straight[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight[:, x * int(self.mag * self.resolution)], cytbg)
        return offsets

    def calc_offset(self, profile, cytbg):
        try:
            params = self.fit_profile_a(profile, cytbg)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)):
                o = np.nan
            elif math.isclose(params[0], (self.itp / 2) * (1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0, 20],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf, 200]),
                               p0=[self.itp / 2, 0, 100])
        return popt

    def fit_profile_a_func(self, x, l, a, w):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=l, a=a, w=w, o=self.cytbg_offset)

        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50), (50, 100)),
                                     args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, l_a_w, profile, cytbg):
        l, a, w = l_a_w
        y = self.total_profile(profile, cytbg, l=l, a=a, w=w, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Segmenter1aDouble(SegmenterParent, Profile1):
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

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, coors=None, mag=1, iterations=3, parallel=False, resolution=5,
                 freedom=0.3, periodic=True, thickness=50, itp=1000, rol_ave=50, end_region=0.2, cytbg_offset=4):

        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag, iterations=iterations,
                                 periodic=periodic,
                                 parallel=parallel, resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg_g = cytbg_g
        self.cytbg_r = cytbg_r
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.error_check()

    def error_check(self):
        if len(self.cytbg_g) != 2 * self.thickness:
            raise Exception('bg_g must be twice as wide as thickness')

        if len(self.cytbg_r) != 2 * self.thickness:
            raise Exception('bg_r must be twice as wide as thickness')

    def calc_offsets(self):
        """

        """
        # Straighten
        straight_g = straighten(self.img_g, self.coors, int(self.thickness * self.mag))
        straight_r = straighten(self.img_r, self.coors, int(self.thickness * self.mag))

        # Smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.itp), int(self.rol_ave * self.mag), self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg_g = interp_1d_array(self.cytbg_g, 2 * self.itp)
        cytbg_r = interp_1d_array(self.cytbg_r, 2 * self.itp)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight_g[:, x * int(self.mag * self.resolution)],
                                          straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                          cytbg_r)
                for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight_g[:, x * int(self.mag * self.resolution)],
                                              straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                              cytbg_r)
        return offsets

    def calc_offset(self, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """

        try:
            params = self.fit_profiles_de(profile_g, profile_r, cytbg_g, cytbg_r)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)) or math.isclose(params[0],
                                                                                            (self.itp / 2) * (
                                                                                                        1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profiles_curve_fit(self, profile_g, profile_r, cytbg_g, cytbg_r):
        x = np.stack((np.append(cytbg_g, cytbg_r), np.append((np.hstack((profile_g, np.zeros([self.itp])))),
                                                             (np.hstack((profile_r, np.zeros([self.itp])))))), axis=0)
        popt, pcov = curve_fit(self._func, x, np.append(profile_g, profile_r),
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0, 20, 0, 20],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf, 200, np.inf, 200]),
                               p0=[self.itp / 2, 0, 100, 0, 100])
        return popt

    def _func(self, x, l, a_g, w_g, a_r, w_r):
        profile_g = x[1, :self.itp]
        bgcurve_g = x[0, :2 * self.itp]
        profile_r = x[1, 2 * self.itp:3 * self.itp]
        bgcurve_r = x[0, 2 * self.itp:]
        y = np.zeros([2 * self.itp])
        y[:self.itp] = self.total_profile(profile_g, bgcurve_g, l=l, a=a_g, w=w_g, o=self.cytbg_offset)
        y[self.itp:2 * self.itp] = self.total_profile(profile_r, bgcurve_r, l=l, a=a_r, w=w_r, o=self.cytbg_offset)
        return y

    def fit_profiles_de(self, profile_g, profile_r, cytbg_g, cytbg_r):
        res = differential_evolution(self._mse, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50), (0, 50), (50, 100),
            (50, 100)), args=(profile_g, profile_r, cytbg_g, cytbg_r))
        return res.x

    def _mse(self, l_ag_ar_wg_wr, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """
        l, ag, ar, wg, wr = l_ag_ar_wg_wr
        yg = self.total_profile(profile_g, cytbg_g, l=l, a=ag, w=wg, o=self.cytbg_offset)
        yr = self.total_profile(profile_r, cytbg_r, l=l, a=ar, w=wr, o=self.cytbg_offset)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


class Segmenter1bSingle(SegmenterParent, Profile1):
    """
    Cytbg offset fixed
    Gaussian width fixed

    """

    def __init__(self, img, cytbg=None, coors=None, mag=1, iterations=3, parallel=False, resolution=5, freedom=0.3,
                 periodic=True, thickness=50, itp=1000, rol_ave=50, cytbg_offset=2, end_region=0.2, gwidth=70):
        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel,
                                 resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg = cytbg
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.gwidth = gwidth
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('bgcurve must be twice as wide as thickness')

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight[:, x * int(self.mag * self.resolution)], cytbg)
                for x in range(len(straight[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight[:, x * int(self.mag * self.resolution)], cytbg)
        return offsets

    def calc_offset(self, profile, cytbg):
        try:
            params = self.fit_profile_a(profile, cytbg)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)):
                o = np.nan
            elif math.isclose(params[0], (self.itp / 2) * (1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf]),
                               p0=[self.itp / 2, 0])
        return popt

    def fit_profile_a_func(self, x, l, a):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=l, a=a, w=self.gwidth, o=self.cytbg_offset)

        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50)),
                                     args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, l_a, profile, cytbg):
        l, a = l_a
        y = self.total_profile(profile, cytbg, l=l, a=a, w=self.gwidth, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Segmenter1bDouble(SegmenterParent, Profile1):
    """
    Cytbg offset fixed
    Gaussuan width fixed

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, coors=None, mag=1, iterations=3, parallel=False, resolution=5,
                 freedom=0.3, periodic=True, thickness=50, itp=1000, rol_ave=50, cytbg_offset=2, end_region=0.2,
                 gwidth=70):
        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag, iterations=iterations,
                                 periodic=periodic,
                                 parallel=parallel, resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg_g = cytbg_g
        self.cytbg_r = cytbg_r
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.gwidth = gwidth
        self.error_check()

    def error_check(self):
        if len(self.cytbg_g) != 2 * self.thickness:
            raise Exception('bg_g must be twice as wide as thickness')

        if len(self.cytbg_r) != 2 * self.thickness:
            raise Exception('bg_r must be twice as wide as thickness')

    def calc_offsets(self):
        """

        """
        # Straighten
        straight_g = straighten(self.img_g, self.coors, int(self.thickness * self.mag))
        straight_r = straighten(self.img_r, self.coors, int(self.thickness * self.mag))

        # Smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.itp), int(self.rol_ave * self.mag), self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg_g = interp_1d_array(self.cytbg_g, 2 * self.itp)
        cytbg_r = interp_1d_array(self.cytbg_r, 2 * self.itp)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight_g[:, x * int(self.mag * self.resolution)],
                                          straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                          cytbg_r)
                for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight_g[:, x * int(self.mag * self.resolution)],
                                              straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                              cytbg_r)
        return offsets

    def calc_offset(self, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """

        try:
            params = self.fit_profiles_curve_fit(profile_g, profile_r, cytbg_g, cytbg_r)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)) or math.isclose(params[0],
                                                                                            (self.itp / 2) * (
                                                                                                        1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profiles_curve_fit(self, profile_g, profile_r, cytbg_g, cytbg_r):
        x = np.stack((np.append(cytbg_g, cytbg_r), np.append((np.hstack((profile_g, np.zeros([self.itp])))),
                                                             (np.hstack((profile_r, np.zeros([self.itp])))))), axis=0)
        popt, pcov = curve_fit(self._func, x, np.append(profile_g, profile_r),
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0, 0],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf, np.inf]),
                               p0=[self.itp / 2, 0, 0])
        return popt

    def _func(self, x, l, a_g, a_r):
        profile_g = x[1, :self.itp]
        bgcurve_g = x[0, :2 * self.itp]
        profile_r = x[1, 2 * self.itp:3 * self.itp]
        bgcurve_r = x[0, 2 * self.itp:]
        y = np.zeros([2 * self.itp])
        y[:self.itp] = self.total_profile(profile_g, bgcurve_g, l=l, a=a_g, w=self.gwidth, o=self.cytbg_offset)
        y[self.itp:2 * self.itp] = self.total_profile(profile_r, bgcurve_r, l=l, a=a_r, w=self.gwidth,
                                                      o=self.cytbg_offset)
        return y

    def fit_profiles_de(self, profile_g, profile_r, cytbg_g, cytbg_r):
        res = differential_evolution(self._mse, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50), (0, 50)),
                                     args=(profile_g, profile_r, cytbg_g, cytbg_r))
        return res.x

    def _mse(self, l_ag_ar, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """
        l, ag, ar = l_ag_ar
        yg = self.total_profile(profile_g, cytbg_g, l=l, a=ag, w=self.gwidth, o=self.cytbg_offset)
        yr = self.total_profile(profile_r, cytbg_r, l=l, a=ar, w=self.gwidth, o=self.cytbg_offset)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


# Fit type 2

class Segmenter2Single(SegmenterParent, Profile2):
    """
    Fit profiles to cytoplasmic background + membrane background

    """

    def __init__(self, img, cytbg, membg, coors=None, mag=1, iterations=2, parallel=False, resolution=5, freedom=0.3,
                 periodic=True, thickness=50, itp=1000, rol_ave=50, end_region=0.2, cytbg_offset=3.5):

        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel,
                                 resolution=resolution, thickness=thickness)

        Profile2.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg = cytbg
        self.membg = membg
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('cytbg must be twice as wide as thickness')

        if len(self.membg) != 2 * self.thickness:
            raise Exception('membg must be twice as wide as thickness')

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

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
        res = differential_evolution(self.mse, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (-5, 50)),
                                     args=(profile, cytbg, membg))
        o = (res.x[0] - self.itp / 2) / (self.itp / self.thickness)

        return o

    def mse(self, l_a, profile, cytbg, membg):
        l, a = l_a
        y = self.total_profile(profile, cytbg, membg, l=l, a=a, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Segmenter2Double(SegmenterParent, Profile2):
    """
    Fit profiles to cytoplasmic background + membrane background
    Two channels

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, membg_g, membg_r, coors=None, mag=1, iterations=2,
                 parallel=False, resolution=5, freedom=0.3, periodic=True, thickness=50, itp=1000, rol_ave=50,
                 end_region=0.2, cytbg_offset=3.5):

        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag, iterations=iterations,
                                 periodic=periodic,
                                 parallel=parallel, resolution=resolution, thickness=thickness)

        Profile2.__init__(self, itp=itp, thickness=thickness, end_region=end_region)

        self.cytbg_g = cytbg_g
        self.cytbg_r = cytbg_r
        self.membg_g = membg_g
        self.membg_r = membg_r
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.error_check()

    def error_check(self):
        if len(self.cytbg_g) != 2 * self.thickness:
            raise Exception('cytbg_g must be twice as wide as thickness')

        if len(self.cytbg_r) != 2 * self.thickness:
            raise Exception('cytbg_r must be twice as wide as thickness')

        if len(self.membg_g) != 2 * self.thickness:
            raise Exception('membg_g must be twice as wide as thickness')

        if len(self.membg_r) != 2 * self.thickness:
            raise Exception('membg_r must be twice as wide as thickness')

    def calc_offsets(self):
        # Straighten
        straight_g = straighten(self.img_g, self.coors, int(self.thickness * self.mag))
        straight_r = straighten(self.img_r, self.coors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.itp), int(self.rol_ave * self.mag), self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg_g = interp_1d_array(self.cytbg_g, 2 * self.itp)
        membg_g = interp_1d_array(self.membg_g, 2 * self.itp)
        cytbg_r = interp_1d_array(self.cytbg_r, 2 * self.itp)
        membg_r = interp_1d_array(self.membg_r, 2 * self.itp)

        # Fit
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile)(straight_g[:, x * int(self.mag * self.resolution)],
                                          straight_r[:, x * int(self.mag * self.resolution)], cytbg_g, cytbg_r,
                                          membg_g, membg_r)
                for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.fit_profile(straight_g[:, x * int(self.mag * self.resolution)],
                                              straight_r[:, x * int(self.mag * self.resolution)], cytbg_g, cytbg_r,
                                              membg_g, membg_r)

        return offsets

    def fit_profile(self, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r):
        res = differential_evolution(self.mse, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (-5, 50), (-5, 50)),
                                     args=(profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r))
        o = (res.x[0] - self.itp / 2) / (self.itp / self.thickness)
        return o

    def mse(self, l_ag_ar, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r):
        l, ag, ar = l_ag_ar
        yg = self.total_profile(profile_g, cytbg_g, membg_g, l=l, a=ag, o=self.cytbg_offset)
        yr = self.total_profile(profile_r, cytbg_r, membg_r, l=l, a=ar, o=self.cytbg_offset)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


# Misc

class Segmenter3(SegmenterParent):
    """

    Segments embryos to midpoint of decline

    Used to segment embryos without a background curve
    e.g. for generating background curves


    """

    def __init__(self, img, coors=None, mag=1, iterations=2, parallel=False, resolution=5, periodic=True,
                 thickness=50, itp=1000, rol_ave=50):

        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel,
                                 resolution=resolution, thickness=thickness)

        self.itp = itp
        self.rol_ave = rol_ave

    def calc_offsets(self):
        """

        """
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

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


class Segmenter4(SegmenterParent):
    """

    Segments embryos to peak

    Used to segment embryos without a background curve
    e.g. for generating background curves


    """

    def __init__(self, img, coors=None, mag=1, iterations=2, parallel=False, resolution=5, periodic=True,
                 thickness=50, itp=1000, rol_ave=50):
        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel, resolution=resolution, thickness=thickness)
        self.itp = itp
        self.rol_ave = rol_ave

    def calc_offsets(self):
        """

        """
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

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


############ MEMBRANE QUANTIFICATION ###########


# Fit type 1

class Quantifier1a(Profile1):
    """
    Cytbg offset fixed
    Gaussian width free

    Seems to give exactly the same answer as Quantifier1c, suggests that gaussian width isn't important

    """

    def __init__(self, img, coors, mag, cytbg, thickness=50, itp=1000, cytbg_offset=4, end_region=0.2, rol_ave=10,
                 periodic=True):

        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)

        self.img = img
        self.coors = coors
        self.mag = mag
        self.thickness = thickness
        self.cytbg = cytbg
        self.periodic = periodic
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('cytbg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            params = self.fit_profile_a(profile, cytbg)
            pro = interp_1d_array(profile, int(self.thickness * self.mag))
            fbc = interp_1d_array(
                self.cyt_profile(profile, cytbg, l=int(self.itp / 2), o=self.cytbg_offset),
                int(self.thickness * self.mag))
            self.sigs[x] = np.trapz(pro - fbc)

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([0, 20], [np.inf, 200]), p0=[0, 100])
        return popt

    def fit_profile_a_func(self, x, a, w):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=int(self.itp / 2), a=a, w=w, o=self.cytbg_offset)
        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=((0, 50), (50, 100)), args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, a_w, profile, cytbg):
        a, w = a_w
        y = self.total_profile(profile, cytbg, l=int(self.itp / 2), a=a, w=w, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Quantifier1b(Profile1):
    """
    Cytbg offset free
    Gaussian width free
    Used for callibration of gaussian width and cytbg offset

    """

    def __init__(self, img, coors, mag, cytbg, thickness=50, itp=1000, end_region=0.2, rol_ave=10,
                 periodic=True):
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)

        self.img = img
        self.coors = coors
        self.mag = mag
        self.thickness = thickness
        self.cytbg = cytbg
        self.periodic = periodic
        self.rol_ave = rol_ave
        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('cytbg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            params = self.fit_profile_b(profile, cytbg)
            pro = interp_1d_array(profile, int(self.thickness * self.mag))
            fbc = interp_1d_array(
                self.cyt_profile(profile, cytbg, l=int(self.itp / 2), o=params[2]),
                int(self.thickness * self.mag))
            self.sigs[x] = np.trapz(pro - fbc)

    # def fit_profile_a(self, profile, cytbg):
    #     x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
    #     popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
    #                            bounds=([0, 20, -5], [np.inf, 200, 5]), p0=[0, 100, 0])
    #     return popt
    #
    # def fit_profile_a_func(self, x, a, w, o):
    #     profile = x[1, :self.itp]
    #     bgcurve = x[0, :]
    #     y = self.total_profile(profile, bgcurve, l=int(self.itp / 2), a=a, w=w, o=o)
    #     return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=((0, 50), (50, 100), (-5, 5)),
                                     args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, a_w_o, profile, cytbg):
        a, w, o = a_w_o
        y = self.total_profile(profile, cytbg, l=int(self.itp / 2), a=a, w=w, o=o)
        return np.mean((profile - y) ** 2)


class Quantifier1c(Profile1):
    """
    Cytbg offset fixed
    Gaussian width fixed

    Main method for quantification

    """

    def __init__(self, img, coors, mag, cytbg, thickness=50, itp=1000, cytbg_offset=2, end_region=0.2, rol_ave=10,
                 periodic=True, gwidth=70):
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)

        self.img = img
        self.coors = coors
        self.mag = mag
        self.thickness = thickness
        self.cytbg = cytbg
        self.periodic = periodic
        self.gwidth = gwidth
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('cytbg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            params = self.fit_profile_a(profile, cytbg)
            pro = interp_1d_array(profile, int(self.thickness * self.mag))
            fbc = interp_1d_array(
                self.cyt_profile(profile, cytbg, l=int(self.itp / 2), o=self.cytbg_offset),
                int(self.thickness * self.mag))
            self.sigs[x] = np.trapz(pro - fbc)

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([0, 20]), p0=[0])
        return popt

    def fit_profile_a_func(self, x, a):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=int(self.itp / 2), a=a, w=self.gwidth, o=self.cytbg_offset)
        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=((0, 50), (50, 100)), args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, a, profile, cytbg):
        y = self.total_profile(profile, cytbg, l=int(self.itp / 2), a=a, w=self.gwidth, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


# Fit type 2

class Quantifier2a(Profile2):
    """
    Cytbg offset fixed

    Main method for quantification

    """

    def __init__(self, img, coors, mag, cytbg, membg, thickness=50, itp=1000, rol_ave=10, periodic=True,
                 end_region=0.2,
                 cytbg_offset=3.5):

        Profile2.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.img = img
        self.coors = coors
        self.mag = mag
        self.cytbg = cytbg
        self.membg = membg
        self.rol_ave = rol_ave
        self.periodic = periodic
        self.cytbg_offset = cytbg_offset
        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('cytbg must be twice as wide as thickness')
        if len(self.membg) != 2 * self.thickness:
            raise Exception('membg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)
        membg = interp_1d_array(self.membg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            a = self.fit_profile_a(profile, cytbg, membg)
            l = int(self.itp / 2)
            j = (self.cytbg_offset * self.itp) / self.thickness
            m1, c1, c2 = self.fix_ends(profile, cytbg[int(l + j):int(l + j) + self.itp],
                                       membg[int(l):int(l) + self.itp], a)
            self.sigs[x] = m1 * a

    def fit_profile_a(self, profile, cytbg, membg):
        res = differential_evolution(self.fit_profile_a_func, bounds=((-5, 50),), args=(profile, cytbg, membg))
        return res.x[0]

    def fit_profile_a_func(self, a, profile, cytbg, membg):
        y = self.total_profile(profile, cytbg, membg, l=int(self.itp / 2), a=a, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Quantifier2b(Profile2):
    """
    Cytbg offset free

    For calibration of cytbg offset

    """

    def __init__(self, img, coors, mag, cytbg, membg, thickness=50, itp=1000, rol_ave=10, periodic=True,
                 end_region=0.2):
        Profile2.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.img = img
        self.coors = coors
        self.mag = mag
        self.cytbg = cytbg
        self.membg = membg
        self.rol_ave = rol_ave
        self.periodic = periodic
        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if len(self.cytbg) != 2 * self.thickness:
            raise Exception('cytbg must be twice as wide as thickness')
        if len(self.membg) != 2 * self.thickness:
            raise Exception('membg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)
        membg = interp_1d_array(self.membg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            res = self.fit_profile_a(profile, cytbg, membg)
            l = int(self.itp / 2)
            j = (res[1] * self.itp) / self.thickness
            m1, c1, c2 = self.fix_ends(profile, cytbg[int(l + j):int(l + j) + self.itp],
                                       membg[int(l):int(l) + self.itp], res[0])
            self.sigs[x] = m1 * res[0]

    def fit_profile_a(self, profile, cytbg, membg):
        res = differential_evolution(self.fit_profile_a_func, bounds=((-5, 50), (-5, 5)), args=(profile, cytbg, membg))
        return res.x

    def fit_profile_a_func(self, a_o, profile, cytbg, membg):
        a, o = a_o
        y = self.total_profile(profile, cytbg, membg, l=int(self.itp / 2), a=a, o=o)
        return np.mean((profile - y) ** 2)


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
    # newimg[newimg == 0] = np.nan
    return newimg


def polycrop2(img, polyline, enlarge):
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
    newimg[newimg == 0] = np.nan
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

    To save this in a fiji readable format segment:
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


def asi(signals):
    ant = bounded_mean_1d(signals, (0.25, 0.75))
    post = bounded_mean_1d(signals, (0.75, 0.25))
    return (ant - post) / (2 * (ant + post))


# General functions

def interp_1d_array(array, n):
    """
    Interpolates a one dimensional array into n points

    :param array:
    :param n:
    :return:
    """

    return np.interp(np.linspace(0, len(array) - 1, n), np.array(range(len(array))), array)


def interp_2d_array(array, n, ax=1):
    """
    Interpolates values along y axis into n points, for each x value
    :param array:
    :param n:
    :param ax:
    :return:
    """

    if ax == 1:
        interped = np.zeros([n, len(array[0, :])])
        for x in range(len(array[0, :])):
            interped[:, x] = interp_1d_array(array[:, x], n)
        return interped
    elif ax == 0:
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


def bounded_mean_1d(array, bounds, weights=None):
    """
    Averages 1D array over region specified by bounds

    Should add interpolation step first

    Array and weights should be same length

    :param array:
    :param bounds:
    :return:
    """

    if weights is None:
        weights = np.ones([len(array)])
    if bounds[0] < bounds[1]:
        mean = np.average(array[int(len(array) * bounds[0]): int(len(array) * bounds[1] + 1)],
                          weights=weights[int(len(array) * bounds[0]): int(len(array) * bounds[1] + 1)])
    else:
        mean = np.average(np.hstack((array[:int(len(array) * bounds[1] + 1)], array[int(len(array) * bounds[0]):])),
                          weights=np.hstack(
                              (weights[:int(len(array) * bounds[1] + 1)], weights[int(len(array) * bounds[0]):])))
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
