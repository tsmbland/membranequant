from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution
from scipy.ndimage.interpolation import map_coordinates
from joblib import Parallel, delayed
import multiprocessing
from scipy.interpolate import splprep, splev, CubicSpline
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import convolve
import cv2
from matplotlib.widgets import Slider

"""
Functions and Classes for segmentation and quantification of images

To do:
- ipynbs for segmentation examples
- notebooks to explain methods
- make more use of norm_coors function (rotate_coors, cse)
- tidy test datasets (include af-corrected image?), write info file
- write readme
- function for reverse straightening
- incorporate uncertainty into fitting algorithms???
- parallel option for quantification
- use convolve function for rolling average functions (scipy.ndimage.filters.convolve)
- scipy.ndimage.filters.laplace for model??
- more specific import statements
- need to fix issue in segmenter where fitting fails

- adjust ranges for profile fits, or way to automatically set range
- change coordinate filtering method
- way to remove polar body
- artefacts at edges when not periodic

"""


############ PROFILE TYPES #############


class Profile:
    @staticmethod
    def total_profile(profile, cytbg, membg, l, c, m, o):
        return (c * cytbg[int(l + o):int(l + o) + len(profile)]) + (m * membg[int(l):int(l) + len(profile)])

    @staticmethod
    def cyt_profile(profile, cytbg, membg, l, c, m, o):
        return c * cytbg[int(l + o):int(l + o) + len(profile)]

    @staticmethod
    def mem_profile(profile, cytbg, membg, l, c, m, o):
        return m * membg[int(l):int(l) + len(profile)]


############### SEGMENTATION ################

class SegmenterParent:
    """

    """

    def __init__(self, img=None, img_g=None, img_r=None, coors=None, mag=1, periodic=True,
                 thickness=50, savgol_window=19, savgol_order=1):

        # Inputs
        self.img = img
        self.img_g = img_g
        self.img_r = img_r
        self.input_coors = coors
        self.coors = coors
        self.mag = mag
        self.periodic = periodic
        self.thickness = int(thickness * mag)
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order

    def run(self, parallel=False, iterations=3):
        """
        Performs segmentation algorithm

        """

        # Interpolate coors to one pixel distance between points
        self.coors = self.interp_coors(self.coors, self.periodic)

        for i in range(iterations):
            # Calculate offsets
            offsets = self.calc_offsets(parallel)

            # Interpolate nans
            nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
            offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])

            # Interpolate
            offsets = interp_1d_array(offsets, len(self.coors))

            # Offset coordinates
            self.coors = offset_coordinates(self.coors, offsets)

            # Filter
            if self.savgol_window is not None:
                if self.periodic:
                    self.coors = np.vstack(
                        (savgol_filter(self.coors[:, 0], self.savgol_window, self.savgol_order, mode='wrap'),
                         savgol_filter(self.coors[:, 1], self.savgol_window, self.savgol_order, mode='wrap'))).T
                elif not self.periodic:
                    self.coors = np.vstack(
                        (savgol_filter(self.coors[:, 0], self.savgol_window, self.savgol_order, mode='nearest'),
                         savgol_filter(self.coors[:, 1], self.savgol_window, self.savgol_order, mode='nearest'))).T

            # Interpolate to one px distance between points
            self.coors = self.interp_coors(self.coors, self.periodic)

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
            self.comp_plot(straighten(self.img_g, self.coors, self.thickness),
                           straighten(self.img_r, self.coors, self.thickness))
        if self.img is not None:
            plt.imshow(straighten(self.img, self.coors, self.thickness), cmap='gray', vmin=0)
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
    def interp_coors(coors, periodic=True):
        """
        Interpolates coordinates to one pixel distances (or as close as possible to one pixel)
        Linear interpolation

        :param coors:
        :return:
        """

        if periodic:
            c = np.append(coors, [coors[0, :]], axis=0)
        else:
            c = coors
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

    def fit_spline(self, coors, periodic=True):
        """
        Fits a spline to points specifying the coordinates of the cortex, then interpolates to pixel distances

        :param coors:
        :return:
        """

        # Append the starting x,y coordinates
        if periodic:
            x = np.r_[coors[:, 0], coors[0, 0]]
            y = np.r_[coors[:, 1], coors[0, 1]]
        else:
            x = coors[:, 0]
            y = coors[:, 1]

        # Fit spline
        tck, u = splprep([x, y], s=0, per=periodic)

        # Evaluate spline
        xi, yi = splev(np.linspace(0, 1, 1000), tck)

        # Interpolate
        return self.interp_coors(np.vstack((xi, yi)).T, periodic=periodic)

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


class Segmenter1Single(SegmenterParent):
    """
    Fit profiles to cytoplasmic background + membrane background

    Parameters:
    mag                magnification of image wrt 60x
    resolution         gap between segmentation points
    freedom            amount of freedom allowed in offset (0=min, 1=max)
    periodic           True if coordinates form a closed loop
    thickness          thickness of cross section over which to perform segmentation
    itp                amount to interpolate image prior to segmentation (this many points per pixel in original image)
    rol_ave            width of rolling average
    cytbg_offset       offset between cytbg and membg

    By default itp, rol_ave, cytbg_offset and resolution will adjust if mag is not 1 to ensure similar behavior at
    different magnifications

    Freedom parameter: max offset is +- 0.5 * freedom * thickness

    """

    def __init__(self, img, cytbg, membg, coors=None, mag=1, resolution=1, freedom=0.3,
                 periodic=True, thickness=50, itp=10, rol_ave=50, cytbg_offset=4.5, savgol_window=19, savgol_order=1):

        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, periodic=periodic,
                                 thickness=thickness * mag, savgol_window=savgol_window, savgol_order=savgol_order)

        self.itp = itp / mag
        self.thickness2 = int(itp * self.thickness)
        self.cytbg = interp_1d_array(cytbg, 2 * self.thickness2)
        self.membg = interp_1d_array(membg, 2 * self.thickness2)
        self.freedom = freedom
        self.rol_ave = int(rol_ave * mag)
        self.cytbg_offset = int(itp * cytbg_offset)
        self.lmin = (self.thickness2 / 2) * (1 - self.freedom)
        self.lmax = (self.thickness2 / 2) * (1 + self.freedom)
        self.resolution = int(resolution * mag)

        self.cyts = np.zeros(len(coors[:, 0]) // self.resolution)
        self.mems = np.zeros(len(coors[:, 0]) // self.resolution)

    def calc_offsets(self, parallel):
        # Straighten
        straight = straighten(self.img, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.thickness2, method='cubic'), self.rol_ave,
                                  self.periodic)

        # Fit
        if parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile)(straight[:, x * self.resolution], self.cytbg, self.membg)
                for x in range(len(straight[0, :]) // self.resolution)))
        else:
            offsets = np.zeros(len(straight[0, :]) // self.resolution)
            for x in range(len(straight[0, :]) // self.resolution):
                results = self.fit_profile(straight[:, x * self.resolution], self.cytbg, self.membg)
                offsets[x] = results[0]
                self.cyts[x] = results[1]
                self.mems[x] = results[2]

        # plt.plot(offsets)
        # plt.show()

        return offsets

    def fit_profile(self, profile, cytbg, membg):
        res = differential_evolution(self.mse, bounds=((self.lmin, self.lmax), (-2000, 20000), (-2000, 30000)),
                                     args=(profile, cytbg, membg))

        # if res.x[2] < 0:
        #     plt.plot(profile)
        #     plt.plot(
        #         Profile.total_profile(profile, cytbg, membg, l=res.x[0], c=res.x[1], m=res.x[2], o=self.cytbg_offset))
        #     plt.plot(
        #         Profile.cyt_profile(profile, cytbg, membg, l=res.x[0], c=res.x[1], m=res.x[2], o=self.cytbg_offset))
        #     plt.plot(
        #         Profile.mem_profile(profile, cytbg, membg, l=res.x[0], c=res.x[1], m=res.x[2], o=self.cytbg_offset))
        #     plt.show()

        o = (res.x[0] - self.thickness2 / 2) / self.itp
        return o, res.x[1], res.x[2]

    def mse(self, l_c_m, profile, cytbg, membg):
        l, c, m = l_c_m
        y = Profile.total_profile(profile, cytbg, membg, l=l, c=c, m=m, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Segmenter1SingleUC(SegmenterParent):
    """
    Fit profiles to cytoplasmic background + membrane background

    Parameters:
    mag                magnification of image wrt 60x
    resolution         gap between segmentation points
    freedom            amount of freedom allowed in offset (0=min, 1=max)
    periodic           True if coordinates form a closed loop
    thickness          thickness of cross section over which to perform segmentation
    itp                amount to interpolate image prior to segmentation (this many points per pixel in original image)
    rol_ave            width of rolling average
    cytbg_offset       offset between cytbg and membg

    By default itp, rol_ave, cytbg_offset and resolution will adjust if mag is not 1 to ensure similar behavior at
    different magnifications


    Uniform cytoplasm


    """

    def __init__(self, img, cytbg, membg, coors=None, mag=1, resolution=1, freedom=0.3,
                 periodic=True, thickness=50, itp=10, rol_ave=50, cytbg_offset=4.5, parallel=False,
                 resolution_cyt=50, savgol_window=19, savgol_order=1):

        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, periodic=periodic,
                                 thickness=thickness * mag, savgol_window=savgol_window, savgol_order=savgol_order)

        self.itp = itp / mag
        self.thickness2 = int(itp * self.thickness)
        self.cytbg = interp_1d_array(cytbg, 2 * self.thickness2)
        self.membg = interp_1d_array(membg, 2 * self.thickness2)
        self.freedom = freedom
        self.rol_ave = int(rol_ave * mag)
        self.cytbg_offset = int(itp * cytbg_offset)
        self.lmin = (self.thickness2 / 2) * (1 - self.freedom)
        self.lmax = (self.thickness2 / 2) * (1 + self.freedom)
        self.resolution = int(resolution * mag)
        self.resolution_cyt = int(resolution_cyt * mag)
        self.parallel = parallel

    def calc_offsets(self, parallel):
        # Straighten
        straight = straighten(self.img, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.thickness2, method='cubic'), self.rol_ave,
                                  self.periodic)

        # Fit global cytoplasm
        c = self.fit_profile_1(straight, self.cytbg, self.membg)

        # Fit
        if parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_2)(straight[:, x * self.resolution], self.cytbg, self.membg, c)
                for x in range(len(straight[0, :]) // self.resolution)))
        else:
            offsets = np.zeros(len(straight[0, :]) // self.resolution)
            for x in range(len(straight[0, :]) // self.resolution):
                offsets[x] = self.fit_profile_2(straight[:, x * self.resolution], self.cytbg, self.membg, c)

        return offsets

    def fit_profile_1(self, straight, cytbg, membg):
        """
        For finding optimal global cytoplasm

        """

        res = differential_evolution(self.fit_profile_1_func, bounds=((-2000, 20000),), args=(straight, cytbg, membg))
        return res.x[0]

    def fit_profile_1_func(self, c, straight, cytbg, membg):
        if self.parallel:
            mses = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_2b)(straight[:, x * self.resolution_cyt], cytbg, membg, c)
                for x in range(len(straight[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight[0, :]) // self.resolution_cyt)
            for x in range(len(straight[0, :]) // self.resolution_cyt):
                mses[x] = self.fit_profile_2b(straight[:, x * self.resolution_cyt], cytbg, membg, c)
        return np.mean(mses)

    def fit_profile_2(self, profile, cytbg, membg, c):
        """
        For finding optimal local membrane, alignment
        Returns offset

        """

        res = differential_evolution(self.fit_profile_2_func, bounds=((self.lmin, self.lmax), (-2000, 30000)),
                                     args=(profile, cytbg, membg, c))
        o = (res.x[0] - self.thickness2 / 2) / self.itp
        return o

    def fit_profile_2b(self, profile, cytbg, membg, c):
        """
        For finding optimal local membrane, alignment
        Returns mse

        """

        res = differential_evolution(self.fit_profile_2_func, bounds=((self.lmin, self.lmax), (-2000, 30000)),
                                     args=(profile, cytbg, membg, c))
        return res.fun

    def fit_profile_2_func(self, l_m, profile, cytbg, membg, c):
        l, m = l_m
        y = Profile.total_profile(profile, cytbg, membg, l=l, c=c, m=m, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Segmenter1Double(SegmenterParent):
    """
    Fit profiles to cytoplasmic background + membrane background
    Two channels

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, membg_g, membg_r, coors=None, mag=1,
                 resolution=1, freedom=0.3, periodic=True, thickness=50, itp=10, rol_ave=50,
                 cytbg_offset_g=4.5, cytbg_offset_r=4.5, savgol_window=19, savgol_order=1):

        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag,
                                 periodic=periodic, thickness=thickness * mag, savgol_window=savgol_window,
                                 savgol_order=savgol_order)

        self.itp = itp / mag
        self.thickness2 = itp * self.thickness
        self.cytbg_g = interp_1d_array(cytbg_g, 2 * self.thickness2)
        self.cytbg_r = interp_1d_array(cytbg_r, 2 * self.thickness2)
        self.membg_g = interp_1d_array(membg_g, 2 * self.thickness2)
        self.membg_r = interp_1d_array(membg_r, 2 * self.thickness2)
        self.freedom = freedom
        self.rol_ave = int(rol_ave * mag)
        self.cytbg_offset_g = int(itp * cytbg_offset_g)
        self.cytbg_offset_r = int(itp * cytbg_offset_r)
        self.lmin = (self.thickness2 / 2) * (1 - self.freedom)
        self.lmax = (self.thickness2 / 2) * (1 + self.freedom)
        self.resolution = int(resolution * mag)

    def calc_offsets(self, parallel):
        # Straighten
        straight_g = straighten(self.img_g, self.coors, self.thickness)
        straight_r = straighten(self.img_r, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.thickness2, method='cubic'), self.rol_ave,
                                    self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.thickness2, method='cubic'), self.rol_ave,
                                    self.periodic)

        # Fit
        if parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile)(straight_g[:, x * self.resolution],
                                          straight_r[:, x * self.resolution], self.cytbg_g, self.cytbg_r,
                                          self.membg_g, self.membg_r)
                for x in range(len(straight_g[0, :]) // self.resolution)))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // self.resolution)
            for x in range(len(straight_g[0, :]) // self.resolution):
                offsets[x] = self.fit_profile(straight_g[:, x * self.resolution],
                                              straight_r[:, x * self.resolution], self.cytbg_g, self.cytbg_r,
                                              self.membg_g, self.membg_r)

        return offsets

    def fit_profile(self, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r):
        res = differential_evolution(self.mse, bounds=(
            (self.lmin, self.lmax), (-2000, 20000), (-2000, 20000), (-2000, 30000), (-2000, 30000)),
                                     args=(profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r))
        o = (res.x[0] - self.thickness2 / 2) / self.itp
        return o

    def mse(self, l_cg_cr_mg_mr, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r):
        l, cg, cr, mg, mr = l_cg_cr_mg_mr
        yg = Profile.total_profile(profile_g, cytbg_g, membg_g, l=l, c=cg, m=mg, o=self.cytbg_offset_g)
        yr = Profile.total_profile(profile_r, cytbg_r, membg_r, l=l, c=cr, m=mr, o=self.cytbg_offset_r)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


class Segmenter1DoubleUC(SegmenterParent):
    """
    Fit profiles to cytoplasmic background + membrane background
    Two channels

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, membg_g, membg_r, coors=None, mag=1,
                 resolution=1, freedom=0.3, periodic=True, thickness=50, itp=10, rol_ave=50,
                 cytbg_offset_g=4.5, cytbg_offset_r=4.5, parallel=False, resolution_cyt=50, savgol_window=19,
                 savgol_order=1):

        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag,
                                 periodic=periodic, thickness=thickness * mag, savgol_window=savgol_window,
                                 savgol_order=savgol_order)

        self.itp = itp / mag
        self.thickness2 = itp * self.thickness
        self.cytbg_g = interp_1d_array(cytbg_g, 2 * self.thickness2)
        self.cytbg_r = interp_1d_array(cytbg_r, 2 * self.thickness2)
        self.membg_g = interp_1d_array(membg_g, 2 * self.thickness2)
        self.membg_r = interp_1d_array(membg_r, 2 * self.thickness2)
        self.freedom = freedom
        self.rol_ave = int(rol_ave * mag)
        self.cytbg_offset_g = int(itp * cytbg_offset_g)
        self.cytbg_offset_r = int(itp * cytbg_offset_r)
        self.lmin = (self.thickness2 / 2) * (1 - self.freedom)
        self.lmax = (self.thickness2 / 2) * (1 + self.freedom)
        self.resolution = int(resolution * mag)
        self.resolution_cyt = int(resolution_cyt * mag)
        self.parallel = parallel

    def calc_offsets(self, parallel):
        # Straighten
        straight_g = straighten(self.img_g, self.coors, self.thickness)
        straight_r = straighten(self.img_r, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.thickness2, method='cubic'), self.rol_ave,
                                    self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.thickness2, method='cubic'), self.rol_ave,
                                    self.periodic)

        # Find global cytoplasms
        c_g, c_r = self.fit_profile_1(straight_g, straight_r, self.cytbg_g, self.cytbg_r, self.membg_g, self.membg_r)

        # Fit
        if parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_2)(straight_g[:, x * self.resolution],
                                            straight_r[:, x * self.resolution], self.cytbg_g, self.cytbg_r,
                                            self.membg_g, self.membg_r, c_g, c_r)
                for x in range(len(straight_g[0, :]) // self.resolution)))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // self.resolution)
            for x in range(len(straight_g[0, :]) // self.resolution):
                offsets[x] = self.fit_profile_2(straight_g[:, x * self.resolution],
                                                straight_r[:, x * self.resolution], self.cytbg_g, self.cytbg_r,
                                                self.membg_g, self.membg_r, c_g, c_r)

        return offsets

    def fit_profile_1(self, straight_g, straight_r, cytbg_g, cytbg_r, membg_g, membg_r):
        """
        For finding optimal global cytoplasm

        """

        res = differential_evolution(self.fit_profile_1_func, bounds=((-2000, 20000), (-2000, 20000)),
                                     args=(straight_g, straight_r, cytbg_g, cytbg_r, membg_g, membg_r))
        return res.x[0], res.x[1]

    def fit_profile_1_func(self, cg_cr, straight_g, straight_r, cytbg_g, cytbg_r, membg_g, membg_r):
        c_g, c_r = cg_cr
        if self.parallel:
            mses = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_2b)(straight_g[:, x * self.resolution_cyt],
                                             straight_r[:, x * self.resolution_cyt], cytbg_g, cytbg_r, membg_g, membg_r,
                                             c_g, c_r)
                for x in range(len(straight_g[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight_g[0, :]) // self.resolution_cyt)
            for x in range(len(straight_g[0, :]) // self.resolution_cyt):
                mses[x] = self.fit_profile_2b(straight_g[:, x * self.resolution_cyt],
                                              straight_g[:, x * self.resolution_cyt], cytbg_g, cytbg_r, membg_g,
                                              membg_r, c_g, c_r)
        return np.mean(mses)

    def fit_profile_2(self, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, c_g, c_r):
        """
        For finding optimal local membrane, alignment
        Returns offset

        """

        res = differential_evolution(self.fit_profile_2_func,
                                     bounds=((self.lmin, self.lmax), (-2000, 30000), (-2000, 30000)),
                                     args=(profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, c_g, c_r))
        o = (res.x[0] - self.thickness2 / 2) / self.itp
        return o

    def fit_profile_2b(self, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, c_g, c_r):
        """
        For finding optimal local membrane, alignment
        Returns mse

        """

        res = differential_evolution(self.fit_profile_2_func,
                                     bounds=((self.lmin, self.lmax), (-2000, 30000), (-2000, 30000)),
                                     args=(profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, c_g, c_r))
        return res.fun

    def fit_profile_2_func(self, l_mg_mr, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, c_g, c_r):
        l, m_g, m_r = l_mg_mr
        yg = Profile.total_profile(profile_g, cytbg_g, membg_g, l=l, c=c_g, m=m_g, o=self.cytbg_offset_g)
        yr = Profile.total_profile(profile_r, cytbg_r, membg_r, l=l, c=c_r, m=m_r, o=self.cytbg_offset_r)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


class Segmenter1DoubleMix(SegmenterParent):
    """
    Fit profiles to cytoplasmic background + membrane background
    Two channels

    Uniform r, polarised g

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, membg_g, membg_r, coors=None, mag=1,
                 resolution=1, freedom=0.3, periodic=True, thickness=50, itp=10, rol_ave=50,
                 end_region=0.2, cytbg_offset_g=4.5, cytbg_offset_r=4.5, parallel=False, resolution_cyt=50,
                 savgol_window=19, savgol_order=1):

        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag,
                                 periodic=periodic, thickness=thickness * mag, savgol_window=savgol_window,
                                 savgol_order=savgol_order)

        self.itp = itp / mag
        self.thickness2 = itp * self.thickness
        self.cytbg_g = interp_1d_array(cytbg_g, 2 * self.thickness2)
        self.cytbg_r = interp_1d_array(cytbg_r, 2 * self.thickness2)
        self.membg_g = interp_1d_array(membg_g, 2 * self.thickness2)
        self.membg_r = interp_1d_array(membg_r, 2 * self.thickness2)
        self.freedom = freedom
        self.rol_ave = int(rol_ave * mag)
        self.cytbg_offset_g = int(itp * cytbg_offset_g)
        self.cytbg_offset_r = int(itp * cytbg_offset_r)
        self.lmin = (self.thickness2 / 2) * (1 - self.freedom)
        self.lmax = (self.thickness2 / 2) * (1 + self.freedom)
        self.resolution = int(resolution * mag)
        self.resolution_cyt = int(resolution_cyt * mag)
        self.parallel = parallel

    def calc_offsets(self, parallel):
        # Straighten
        straight_g = straighten(self.img_g, self.coors, self.thickness)
        straight_r = straighten(self.img_r, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.thickness2, method='cubic'), self.rol_ave,
                                    self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.thickness2, method='cubic'), self.rol_ave,
                                    self.periodic)

        # Find global cytoplasms
        c_r = self.fit_profile_1(straight_r, self.cytbg_r, self.membg_r)

        # Fit
        if parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_3)(straight_g[:, x * self.resolution],
                                            straight_r[:, x * self.resolution], self.cytbg_g, self.cytbg_r,
                                            self.membg_g, self.membg_r, c_r)
                for x in range(len(straight_g[0, :]) // self.resolution)))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // self.resolution)
            for x in range(len(straight_g[0, :]) // self.resolution):
                offsets[x] = self.fit_profile_3(straight_g[:, x * self.resolution],
                                                straight_r[:, x * self.resolution], self.cytbg_g, self.cytbg_r,
                                                self.membg_g, self.membg_r, c_r)

        return offsets

    def fit_profile_1(self, straight, cytbg, membg):
        """
        For finding optimal global cytoplasm

        """

        res = differential_evolution(self.fit_profile_1_func, bounds=((-2000, 20000),), args=(straight, cytbg, membg))
        return res.x[0]

    def fit_profile_1_func(self, c, straight, cytbg, membg):
        if self.parallel:
            mses = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_2b)(straight[:, x * self.resolution_cyt], cytbg, membg, c)
                for x in range(len(straight[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight[0, :]) // self.resolution_cyt)
            for x in range(len(straight[0, :]) // self.resolution_cyt):
                mses[x] = self.fit_profile_2b(straight[:, x * self.resolution_cyt], cytbg, membg, c)
        return np.mean(mses)

    def fit_profile_2b(self, profile, cytbg, membg, c):
        """
        For finding optimal local membrane, alignment
        Returns mse

        """

        res = differential_evolution(self.fit_profile_2_func, bounds=((self.lmin, self.lmax), (-2000, 30000)),
                                     args=(profile, cytbg, membg, c))
        return res.fun

    def fit_profile_2_func(self, l_m, profile, cytbg, membg, c):
        l, m = l_m
        y = Profile.total_profile(profile, cytbg, membg, l=l, c=c, m=m, o=self.cytbg_offset_g)
        return np.mean((profile - y) ** 2)

    def fit_profile_3(self, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, cr):
        """
        For finding optimal local membrane, alignment
        Returns offset

        """

        res = differential_evolution(self.fit_profile_3_func,
                                     bounds=((self.lmin, self.lmax), (-2000, 20000), (-2000, 30000), (-2000, 30000)),
                                     args=(profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, cr))
        o = (res.x[0] - self.thickness2 / 2) / self.itp
        return o

    def fit_profile_3_func(self, l_cg_mg_mr, profile_g, profile_r, cytbg_g, cytbg_r, membg_g, membg_r, cr):
        l, cg, mg, mr = l_cg_mg_mr
        yg = Profile.total_profile(profile_g, cytbg_g, membg_g, l=l, c=cg, m=mg, o=self.cytbg_offset_g)
        yr = Profile.total_profile(profile_r, cytbg_r, membg_r, l=l, c=cr, m=mr, o=self.cytbg_offset_r)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


class Segmenter2(SegmenterParent):
    """

    Segments embryos to midpoint of decline

    Used to run embryos without a background curve
    e.g. for generating background curves


    """

    def __init__(self, img, coors=None, mag=1, resolution=1, periodic=True,
                 thickness=50, itp=100, rol_ave=50, savgol_window=19, savgol_order=1):

        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, periodic=periodic,
                                 thickness=thickness * mag, savgol_window=savgol_window, savgol_order=savgol_order)

        self.itp = itp / mag
        self.thickness2 = itp * self.thickness
        self.rol_ave = rol_ave * mag
        self.resolution = int(resolution * mag)

    def calc_offsets(self, parallel):
        """

        """
        # Straighten
        straight = straighten(self.img, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.thickness2, method='cubic'), self.rol_ave,
                                  self.periodic)

        # Calculate offsets
        if parallel:
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
        return ((self.thickness2 / 2) - np.argmin(
            np.absolute(straight[:, x] - np.mean([np.mean(straight[int(0.9 * self.thickness2):, x]),
                                                  np.mean(straight[:int(0.1 * self.thickness2), x])]))) * (
                    len(straight[:, 0]) / self.thickness2)) / self.itp


class Segmenter3(SegmenterParent):
    """

    Segments embryos to peak

    Used to run embryos without a background curve
    e.g. for generating background curves


    """

    def __init__(self, img, coors=None, mag=1, resolution=1, periodic=True,
                 thickness=50, itp=100, rol_ave=50, savgol_window=19, savgol_order=1):
        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, periodic=periodic,
                                 thickness=thickness * mag, savgol_window=savgol_window, savgol_order=savgol_order)
        self.itp = itp / mag
        self.thickness2 = itp * self.thickness
        self.rol_ave = rol_ave * mag
        self.resolution = int(resolution * mag)

    def calc_offsets(self, parallel):
        """

        """
        # Straighten
        straight = straighten(self.img, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.thickness2, method='cubic'), self.rol_ave,
                                  self.periodic)

        # Calculate offsets
        if parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.func)(straight[:, x]) for x in range(len(straight[0, :]))))
        else:
            offsets = np.zeros(len(straight[0, :]))
            for x in range(len(straight[0, :])):
                offsets[x] = self.func(straight[:, x])

        return offsets

    def func(self, profile):
        """

        """
        return ((self.thickness2 / 2) - np.argmax(profile) * (len(profile) / self.thickness2)) / self.itp


class Segmenter4(SegmenterParent):
    """

    Segments embryos to greatest slope

    """

    def __init__(self, img, coors=None, mag=1, resolution=1, periodic=True,
                 thickness=50, itp=100, rol_ave=50, savgol_window=19, savgol_order=1):
        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, periodic=periodic,
                                 thickness=thickness * mag, savgol_window=savgol_window, savgol_order=savgol_order)
        self.itp = itp / mag
        self.thickness2 = itp * self.thickness
        self.rol_ave = rol_ave * mag
        self.resolution = int(resolution * mag)

    def calc_offsets(self, parallel):
        """

        """
        # Straighten
        straight = straighten(self.img, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.thickness2, method='cubic'), self.rol_ave,
                                  self.periodic)

        # Calculate offsets
        if parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.func)(straight[:, x]) for x in range(len(straight[0, :]))))
        else:
            offsets = np.zeros(len(straight[0, :]))
            for x in range(len(straight[0, :])):
                offsets[x] = self.func(straight[:, x])

        return offsets

    def func(self, profile):
        """

        """
        profile = savgol_filter(profile, 39, 1, mode='mirror')
        diff = np.diff(profile)

        offset = ((self.thickness2 / 2) - (0.5 + np.argmax(diff)) * (len(profile) / self.thickness2)) / self.itp

        return offset


############ MEMBRANE QUANTIFICATION ###########


class Quantifier:
    """


    """

    def __init__(self, img, coors, mag, cytbg, membg, thickness=50, rol_ave=20, periodic=True,
                 cytbg_offset=4.5, psize=0.255, itp=10):
        self.img = img
        self.coors = coors
        self.thickness = thickness * mag
        self.periodic = periodic
        self.itp = itp / mag
        self.thickness2 = int(itp * self.thickness)
        self.cytbg = interp_1d_array(cytbg, 2 * self.thickness2)
        self.membg = interp_1d_array(membg, 2 * self.thickness2)
        self.rol_ave = int(rol_ave * mag)
        self.cytbg_offset = int(itp * cytbg_offset)
        self.psize = psize / mag

        # Results
        self.sigs = np.zeros([len(coors[:, 0])])
        self.cyts = np.zeros([len(coors[:, 0])])
        self.straight = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_fit = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_mem = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_cyt = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_pos = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_neg = np.zeros([self.thickness, len(self.coors[:, 0])])

    def run(self):
        self.straight = rolling_ave_2d(straighten(self.img, self.coors, int(self.thickness)), int(self.rol_ave),
                                       self.periodic)

        straight = interp_2d_array(self.straight, self.thickness2, method='cubic')

        # Get cortical/cytoplasmic signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            c, m = self.fit_profile_a(profile, self.cytbg, self.membg)

            self.sigs[x] = m
            self.cyts[x] = c

            self.straight_cyt[:, x] = interp_1d_array(Profile.cyt_profile(profile, self.cytbg, self.membg,
                                                                          l=int(self.thickness2 / 2),
                                                                          c=c,
                                                                          m=m, o=self.cytbg_offset), self.thickness)
            self.straight_mem[:, x] = interp_1d_array(Profile.mem_profile(profile, self.cytbg, self.membg,
                                                                          l=int(self.thickness2 / 2),
                                                                          c=c,
                                                                          m=m, o=self.cytbg_offset), self.thickness)
            self.straight_fit[:, x] = interp_1d_array(Profile.total_profile(profile, self.cytbg, self.membg,
                                                                            l=int(self.thickness2 / 2),
                                                                            c=c, m=m, o=self.cytbg_offset),
                                                      self.thickness)
            self.straight_resids[:, x] = self.straight[:, x] - self.straight_fit[:, x]
            self.straight_resids_pos[:, x] = np.clip(self.straight_resids[:, x], a_min=0, a_max=None)
            self.straight_resids_neg[:, x] = abs(np.clip(self.straight_resids[:, x], a_min=None, a_max=0))

    def fit_profile_a(self, profile, cytbg, membg):
        res = differential_evolution(self.fit_profile_a_func, bounds=((-2000, 20000), (-2000, 30000)),
                                     args=(profile, cytbg, membg))

        return res.x[0], res.x[1]

    def fit_profile_a_func(self, c_m, profile, cytbg, membg):
        c, m = c_m
        y = Profile.total_profile(profile, cytbg, membg, l=int(self.thickness2 / 2), c=c, m=m, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class QuantifierUC:
    """
    Uniform cytoplasmic pool

    """

    def __init__(self, img, coors, mag, cytbg, membg, thickness=50, rol_ave=20, periodic=True,
                 cytbg_offset=4.5, psize=0.255, resolution_cyt=20, parallel=False, itp=10):

        self.img = img
        self.coors = coors
        self.thickness = thickness * mag
        self.periodic = periodic
        self.itp = itp / mag
        self.thickness2 = int(itp * self.thickness)
        self.cytbg = interp_1d_array(cytbg, 2 * self.thickness2)
        self.membg = interp_1d_array(membg, 2 * self.thickness2)
        self.rol_ave = int(rol_ave * mag)
        self.cytbg_offset = int(itp * cytbg_offset)
        self.resolution_cyt = int(resolution_cyt * mag)
        self.parallel = parallel
        self.psize = psize / mag

        # Results
        self.sigs = np.zeros([len(coors[:, 0])])
        self.cyts = np.zeros([len(coors[:, 0])])
        self.straight = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_fit = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_mem = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_cyt = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_pos = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_neg = np.zeros([self.thickness, len(self.coors[:, 0])])

    def run(self):
        self.straight = rolling_ave_2d(straighten(self.img, self.coors, int(self.thickness)), int(self.rol_ave),
                                       self.periodic)

        # Straighten
        straight = straighten(self.img, self.coors, self.thickness)

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.thickness2, method='cubic'), self.rol_ave,
                                  self.periodic)

        c = self.fit_profile_1(straight, self.cytbg, self.membg)

        if self.parallel:
            ms = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_2a)(straight[:, x], self.cytbg, self.membg, c)
                for x in range(len(straight[0, :]))))
        else:
            ms = np.zeros([len(straight[0, :])])
            for x in range(len(straight[0, :])):
                ms[x] = self.fit_profile_2a(straight[:, x], self.cytbg, self.membg, c)

        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            self.straight_cyt[:, x] = interp_1d_array(Profile.cyt_profile(profile, self.cytbg, self.membg,
                                                                          l=int(self.thickness2 / 2),
                                                                          c=c,
                                                                          m=ms[x], o=self.cytbg_offset), self.thickness)
            self.straight_mem[:, x] = interp_1d_array(Profile.mem_profile(profile, self.cytbg, self.membg,
                                                                          l=int(self.thickness2 / 2),
                                                                          c=c,
                                                                          m=ms[x], o=self.cytbg_offset), self.thickness)
            self.straight_fit[:, x] = interp_1d_array(Profile.total_profile(profile, self.cytbg, self.membg,
                                                                            l=int(self.thickness2 / 2),
                                                                            c=c, m=ms[x], o=self.cytbg_offset),
                                                      self.thickness)
            self.straight_resids[:, x] = self.straight[:, x] - self.straight_fit[:, x]
            self.straight_resids_pos[:, x] = np.clip(self.straight_resids[:, x], a_min=0, a_max=None)
            self.straight_resids_neg[:, x] = abs(np.clip(self.straight_resids[:, x], a_min=None, a_max=0))

            self.sigs[x] = ms[x]
            self.cyts[x] = c

    def fit_profile_1(self, straight, cytbg, membg):
        """
        For finding optimal global cytoplasm

        """

        res = differential_evolution(self.fit_profile_1_func, bounds=((-2000, 20000),),
                                     args=(straight, cytbg, membg))
        return res.x[0]

    def fit_profile_1_func(self, c, straight, cytbg, membg):
        if self.parallel:
            mses = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile_2b)(straight[:, x * self.resolution_cyt], self.cytbg, self.membg, c)
                for x in range(len(straight[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight[0, :]) // self.resolution_cyt)
            for x in range(len(straight[0, :]) // self.resolution_cyt):
                mses[x] = self.fit_profile_2b(straight[:, x * self.resolution_cyt], cytbg, membg, c)
        return np.mean(mses)

    def fit_profile_2a(self, profile, cytbg, membg, c):
        """
        For finding optimal local membrane
        Returns m

        """

        res = differential_evolution(self.fit_profile_2_func, bounds=((-2000, 30000),),
                                     args=(profile, cytbg, membg, c))
        return res.x[0]

    def fit_profile_2b(self, profile, cytbg, membg, c):
        """
        For finding optimal local membrane
        Returns mse

        """

        res = differential_evolution(self.fit_profile_2_func, bounds=((-2000, 30000),),
                                     args=(profile, cytbg, membg, c))
        return res.fun

    def fit_profile_2_func(self, m, profile, cytbg, membg, c):
        y = Profile.total_profile(profile, cytbg, membg, l=int(self.thickness2 / 2), c=c, m=m, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


############### MISC FUNCTIONS ##############


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
    # newcoors_x = np.zeros([thickness, len(coors[:, 0])])
    # newcoors_y = np.zeros([thickness, len(coors[:, 0])])

    for section in range(thickness):
        sectioncoors = offset_coordinates(coors, offsets[section])
        a = map_coordinates(img.T, [sectioncoors[:, 0], sectioncoors[:, 1]])
        a[a == 0] = np.mean(a)  # if selection goes outside of the image
        img2[section, :] = a
        # newcoors_x[section, :] = sectioncoors[:, 0]
        # newcoors_y[section, :] = sectioncoors[:, 1]
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

        if run != 0.:
            bisectorgrad = rise / run
            tangentgrad = -1 / bisectorgrad
            xchange = ((offsets[i] ** 2) / (1 + tangentgrad ** 2)) ** 0.5
            ychange = xchange / abs(bisectorgrad)

        else:
            xchange = ((offsets[i] ** 2) / (1 ** 2)) ** 0.5
            ychange = 0

        newxs[i] = xcoors[i] + np.sign(rise) * np.sign(offsets[i]) * xchange
        newys[i] = ycoors[i] - np.sign(run) * np.sign(offsets[i]) * ychange

    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)

    return newcoors


def interp_1d_array(array, n, method='linear'):
    """
    Interpolates a one dimensional array into n points

    :param array:
    :param n:
    :return:
    """

    if method == 'linear':
        return np.interp(np.linspace(0, len(array) - 1, n), np.array(range(len(array))), array)
    elif method == 'cubic':
        return CubicSpline(np.arange(len(array)), array)(np.linspace(0, len(array) - 1, n))


def interp_2d_array(array, n, ax=1, method='linear'):
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
            interped[:, x] = interp_1d_array(array[:, x], n, method)
        return interped
    elif ax == 0:
        interped = np.zeros([len(array[:, 0]), n])
        for x in range(len(array[:, 0])):
            interped[x, :] = interp_1d_array(array[x, :], n, method)
        return interped
    else:
        return None


def rolling_ave_2d(array, window, periodic=True):
    """
    Returns rolling average across the x axis of an image (used for straightened profiles)

    :param array: image data
    :param window: number of pixels to average over. Odd number is best
    :param periodic: is true, rolls over at ends
    :return: ave
    """

    if window == 1:
        return array

    if not periodic:
        ave = np.zeros([len(array[:, 0]), len(array[0, :])])
        for y in range(len(array[:, 0])):
            ave[y, :] = convolve(array[y, :], np.ones(window) / window, mode='mirror')
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
    By subtracting average signal 50 - 100 pixels from embryo

    :param img:
    :param coors:
    :param mag:
    :return:
    """

    a = polycrop(img, coors, 75 * mag) - polycrop(img, coors, 25 * mag)
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


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


def dosage(img, coors, expand=5):
    ys = np.zeros(img.shape)
    for y in range(len(img[:, 0])):
        ys[y, :] = y
    xs = np.zeros(img.shape)
    for x in range(len(img[:, 0])):
        xs[:, x] = x
    M = (coors - np.mean(coors.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M).T
    ys2 = (ys - np.mean(coors[:, 1]))
    xs2 = (xs - np.mean(coors[:, 0]))
    newxs = abs(coeff.T[0, 0] * xs2 + coeff.T[0, 1] * ys2)
    newys = abs(coeff.T[1, 0] * xs2 + coeff.T[1, 1] * ys2)
    if (max(score[0, :]) - min(score[0, :])) < (max(score[1, :]) - min(score[1, :])):
        newxs, newys = newys, newxs
    a = polycrop2(img, coors, expand)
    indices = ~np.isnan(a)
    return np.average(a[indices], weights=newys[indices])


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


def coor_angles(coors):
    """
    Calculates angle at each coordinate
    Uses law of cosines to find angles

    :param coors:
    :return:
    """

    c = np.r_[coors, coors[:2, :]]

    distances = ((np.diff(c[:, 0]) ** 2) + (np.diff(c[:, 1]) ** 2)) ** 0.5
    distances2 = np.zeros([len(coors[:, 0])])
    for i in range(len(coors[:, 0])):
        distances2[i] = (((c[i + 2, 0] - c[i, 0]) ** 2) + ((c[i + 2, 1] - c[i, 1]) ** 2)) ** 0.5
    angles = np.zeros([len(coors[:, 0])])
    for i in range(len(coors[:, 0])):
        angles[i] = np.arccos(((distances[i] ** 2) + (distances[i + 1] ** 2) - (distances2[i] ** 2)) / (
            2 * distances[i] * distances[i + 1]))
    angles = np.roll(angles, 1)

    return angles


def view_stack(stack, vmin, vmax):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
    sframe = Slider(axframe, 'Time point', 0, len(stack[:, 0, 0]), valinit=0, valfmt='%d')

    def update(i):
        ax.clear()
        ax.imshow(stack[int(i), :, :], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    sframe.on_changed(update)
    plt.show()
