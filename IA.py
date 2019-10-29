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
from scipy.linalg import lstsq
import cv2
from matplotlib.widgets import Slider
import os
import shutil
import random
import sys
import pickle
import glob
import seaborn as sns

"""
Functions and Classes for segmentation and quantification of membrane and cytoplasmic protein concentrations from 
midplane confocal images of C. elegans zygotes


"""


############# QUANT CLASS ############


class MembraneQuant:
    """
    Fit profiles to cytoplasmic background + membrane background

    Input data:
    img                image
    cytbg              cytoplasmic background curve, should be 2x as thick as thickness parameter
    membg              membrane background curve, as above
    coors              coordinates defining cortex. Can use output from def_ROI function


    Parameters:
    resolution         gap between segmentation points
    freedom            amount of freedom allowed in offset (0=min, 1=max, max offset is +- 0.5 * freedom * thickness)
    periodic           True if coordinates form a closed loop
    thickness          thickness of cross section over which to perform segmentation
    itp                amount to interpolate image prior to segmentation (this many points per pixel in original image)
    rol_ave            width of rolling average
    cytbg_offset       offset cytoplasmic background curve by this many pixels. Can get better fitting but shouldn't be
                       necessary if background curves are well defined
    savgol_window      for coordinate fitting
    savgol_order       for coordinate fitting
    resolution_cyt     for cytoplasmic fitting. Can get large performance increase by increasing this, at small cost to
                       accuracy
    parallel           TRUE = perform fitting in parallel using all available cores
    method             different fitting methods, see below



    """

    def __init__(self, img, cytbg, membg, coors=None, resolution=1, freedom=0.3,
                 periodic=True, thickness=50, itp=10, rol_ave=20, parallel=False, method='1', cytbg_offset=0,
                 savgol_window=19, savgol_order=1, resolution_cyt=30, rotate=True):

        self.img = img * 0.0001
        self.coors_init = coors
        self.coors = coors
        self.periodic = periodic
        self.thickness = thickness
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.itp = itp
        self.thickness_itp = int(itp * self.thickness)
        self.cytbg = cytbg
        self.membg = membg
        self.cytbg_itp = interp_1d_array(self.cytbg, 2 * self.thickness_itp)
        self.membg_itp = interp_1d_array(self.membg, 2 * self.thickness_itp)
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.resolution = resolution
        self.resolution_cyt = resolution_cyt
        self.parallel = parallel
        self.method = method
        self.rotate = rotate

        # Results
        self.offsets = np.zeros(len(self.coors[:, 0]))
        self.cyts = np.zeros(len(self.coors[:, 0]))
        self.mems = np.zeros(len(self.coors[:, 0]))
        self.straight = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_filtered = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_fit = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_mem = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_cyt = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_pos = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_neg = np.zeros([self.thickness, len(self.coors[:, 0])])

    class Profile:
        @staticmethod
        def total_profile(width, cytbg, membg, l, c, m, o):
            return (c * cytbg[int(l + o):int(l + o) + width]) + (m * membg[int(l):int(l) + width])

        @staticmethod
        def cyt_profile(width, cytbg, membg, l, c, m, o):
            return c * cytbg[int(l + o):int(l + o) + width]

        @staticmethod
        def mem_profile(width, cytbg, membg, l, c, m, o):
            return m * membg[int(l):int(l) + width]

    def fit(self):

        # Filter/smoothen/interpolate straight image
        self.straight = straighten(self.img, self.coors, self.thickness)
        self.straight_filtered = rolling_ave_2d(self.straight, self.rol_ave, self.periodic)
        straight = interp_2d_array(self.straight_filtered, self.thickness_itp, method='cubic')

        # Fit
        if self.method == '1':
            if self.parallel:
                results = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(self._fit_profile)(straight[:, x * self.resolution], self.cytbg_itp, self.membg_itp)
                    for x in range(len(straight[0, :]) // self.resolution)))
                self.offsets = interp_1d_array(results[:, 0], len(self.coors[:, 0]))
                self.cyts = interp_1d_array(results[:, 1], len(self.coors[:, 0]))
                self.mems = interp_1d_array(results[:, 2], len(self.coors[:, 0]))
            else:
                for x in range(len(straight[0, :]) // self.resolution):
                    results = self._fit_profile(straight[:, x * self.resolution], self.cytbg_itp, self.membg_itp)
                    self.offsets[x] = results[0]
                    self.cyts[x] = results[1]
                    self.mems[x] = results[2]

        elif self.method == '2':
            # Fit uniform cytoplasm
            c = self._fit_profile_1(straight, self.cytbg_itp, self.membg_itp)
            self.cyts[:] = c

            # Fit local membranes
            if self.parallel:
                results = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(self._fit_profile_2)(straight[:, x], self.cytbg_itp, self.membg_itp, c)
                    for x in range(len(straight[0, :]))))
                self.offsets = interp_1d_array(results[:, 0], len(self.coors[:, 0]))
                self.mems = interp_1d_array(results[:, 1], len(self.coors[:, 0]))

            else:
                for x in range(len(straight[0, :]) // self.resolution):
                    results = self._fit_profile_2(straight[:, x], self.cytbg_itp, self.membg_itp, c)
                    self.offsets[x] = results[0]
                    self.mems[x] = results[1]

        elif self.method == '3':
            # Fit uniform cytoplasm, uniform membrane
            c, m = self._fit_profile_ucum(straight, self.cytbg_itp, self.membg_itp)
            self.mems[:] = m
            self.cyts[:] = c

            # Fit local offsets
            if self.parallel:
                results = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(self._fit_profile_ucum_2)(straight[:, x], self.cytbg_itp, self.membg_itp, c, m)
                    for x in range(len(straight[0, :]))))
                self.offsets = interp_1d_array(results, len(self.coors[:, 0]))

            else:
                for x in range(len(straight[0, :]) // self.resolution):
                    results = self._fit_profile_ucum_2(straight[:, x], self.cytbg_itp, self.membg_itp, c, m)
                    self.offsets[x] = results
        else:
            print('Method does not exist')

        self.sim_images()

    """
    METHOD 1: Non-uniform cytoplasm
    - fastest method, however too much freedom so fits can look odd

    """

    def _fit_profile(self, profile, cytbg, membg):
        bounds = (
            ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
            (0, 2 * max(profile)), (0, 2 * max(profile)))
        res = differential_evolution(self._mse, bounds=bounds, args=(profile, cytbg, membg), tol=0.1)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o, res.x[1], res.x[2]

    def _mse(self, l_c_m, profile, cytbg, membg):
        l, c, m = l_c_m
        y = self.Profile.total_profile(len(profile), cytbg, membg, l=l, c=c, m=m, o=int(self.itp * self.cytbg_offset))
        return np.mean((profile - y) ** 2)

    """
    METHOD 2: Uniform cytoplasm
    - for most purposes this is best
    
    """

    def _fit_profile_1(self, straight, cytbg, membg):
        """
        For finding optimal global cytoplasm

        """

        res = differential_evolution(self._fit_profile_1_func, bounds=((0, 2 * np.percentile(straight, 95)),),
                                     args=(straight, cytbg, membg), tol=0.1)
        return res.x[0]

    def _fit_profile_1_func(self, c, straight, cytbg, membg):
        if self.parallel:
            mses = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self._fit_profile_2b)(straight[:, x * self.resolution_cyt], cytbg, membg, c)
                for x in range(len(straight[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight[0, :]) // self.resolution_cyt)
            for x in range(len(straight[0, :]) // self.resolution_cyt):
                mses[x] = self._fit_profile_2b(straight[:, x * self.resolution_cyt], cytbg, membg, c)
        return np.mean(mses)

    def _fit_profile_2(self, profile, cytbg, membg, c):
        """
        For finding optimal local membrane, alignment
        Returns offset

        """
        bounds = (
            ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
            (0, 2 * max(profile)))

        res = differential_evolution(self._fit_profile_2_func, bounds=bounds,
                                     args=(profile, cytbg, membg, c), tol=0.1)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o, res.x[1]

    def _fit_profile_2b(self, profile, cytbg, membg, c):
        """
        For finding optimal local membrane, alignment
        Returns _mse

        """
        bounds = (
            ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
            (0, 2 * max(profile)))

        res = differential_evolution(self._fit_profile_2_func, bounds=bounds,
                                     args=(profile, cytbg, membg, c), tol=0.1)
        return res.fun

    def _fit_profile_2_func(self, l_m, profile, cytbg, membg, c):
        l, m = l_m
        y = self.Profile.total_profile(len(profile), cytbg, membg, l=l, c=c, m=m, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)

    """
    METHOD 3: Uniform cytoplasm, uniform membrane
    - slowest method
    - best method if both membrane and cytoplasmic species can be assumed to be uniform
    
    """

    def _fit_profile_ucum(self, straight, cytbg, membg):
        res = differential_evolution(self._fit_profile_ucum_func, bounds=(
            (0, 2 * np.percentile(straight, 95)), (0, 2 * np.percentile(straight, 95))),
                                     args=(straight, cytbg, membg), tol=0.1)
        return res.x[0], res.x[1]

    def _fit_profile_ucum_func(self, c_m, straight, cytbg, membg):
        c, m = c_m
        if self.parallel:
            mses = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self._fit_profile_ucum_2b)(straight[:, x * self.resolution_cyt], cytbg, membg, c, m)
                for x in range(len(straight[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight[0, :]) // self.resolution_cyt)
            for x in range(len(straight[0, :]) // self.resolution_cyt):
                mses[x] = self._fit_profile_ucum_2b(straight[:, x * self.resolution_cyt], cytbg, membg, c, m)
        return np.mean(mses)

    def _fit_profile_ucum_2(self, profile, cytbg, membg, c, m):
        bounds = (
            ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),)

        res = differential_evolution(self._fit_profile_ucum_2_func, bounds=bounds,
                                     args=(profile, cytbg, membg, c, m), tol=0.1)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o

    def _fit_profile_ucum_2b(self, profile, cytbg, membg, c, m):
        bounds = (
            ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),)

        res = differential_evolution(self._fit_profile_ucum_2_func, bounds=bounds,
                                     args=(profile, cytbg, membg, c, m), tol=0.1)
        return res.fun

    def _fit_profile_ucum_2_func(self, l, profile, cytbg, membg, c, m):
        y = self.Profile.total_profile(len(profile), cytbg, membg, l=l, c=c, m=m, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)

    """
    Misc
    
    """

    def sim_images(self):
        """
        Creates simulated images based on fit results

        """
        for x in range(len(self.coors[:, 0])):
            c = self.cyts[x]
            m = self.mems[x]
            l = int(self.offsets[x] * self.itp + (self.thickness_itp / 2))
            o = int(self.itp * self.cytbg_offset)
            self.straight_cyt[:, x] = interp_1d_array(
                self.Profile.cyt_profile(self.thickness_itp, self.cytbg_itp, self.membg_itp,
                                         l=l, c=c, m=m, o=o), self.thickness)
            self.straight_mem[:, x] = interp_1d_array(
                self.Profile.mem_profile(self.thickness_itp, self.cytbg_itp, self.membg_itp,
                                         l=l, c=c, m=m, o=o), self.thickness)
            self.straight_fit[:, x] = interp_1d_array(
                self.Profile.total_profile(self.thickness_itp, self.cytbg_itp, self.membg_itp,
                                           l=l, c=c, m=m, o=o), self.thickness)
            self.straight_resids[:, x] = self.straight[:, x] - self.straight_fit[:, x]
            self.straight_resids_pos[:, x] = np.clip(self.straight_resids[:, x], a_min=0, a_max=None)
            self.straight_resids_neg[:, x] = abs(np.clip(self.straight_resids[:, x], a_min=None, a_max=0))

    def adjust_coors(self):
        """
        Can do after a preliminary fit to refine coordinates
        Must refit after doing this

        """

        # Interpolate, remove nans
        offsets = self.offsets
        nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
        offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])
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
        self.coors = interp_coors(self.coors, self.periodic)

        # Rotate
        if self.periodic:
            if self.rotate:
                self.coors = rotate_coors(self.coors)

        # Reset
        self.reset_res()

    def reset(self):
        """
        Resets entire class to it's initial state

        """

        self.coors = self.coors_init
        self.reset_res()

    def reset_res(self):
        """
        Clears results

        """
        self.offsets = np.zeros(len(self.coors[:, 0]))
        self.cyts = np.zeros(len(self.coors[:, 0]))
        self.mems = np.zeros(len(self.coors[:, 0]))
        self.straight = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_filtered = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_fit = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_mem = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_cyt = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_pos = np.zeros([self.thickness, len(self.coors[:, 0])])
        self.straight_resids_neg = np.zeros([self.thickness, len(self.coors[:, 0])])

    def save(self, direc):
        """
        Save all results in specific directory

        WARNING: if directory already exists it will delete it

        """

        if os.path.isdir(direc):
            shutil.rmtree(direc)
        os.mkdir(direc)
        np.savetxt(direc + '/offsets.txt', self.offsets, fmt='%.4f', delimiter='\t')
        np.savetxt(direc + '/cyts.txt', self.cyts, fmt='%.4f', delimiter='\t')
        np.savetxt(direc + '/mems.txt', self.mems, fmt='%.4f', delimiter='\t')
        np.savetxt(direc + '/cyts_1000.txt', interp_1d_array(self.cyts, 1000), fmt='%.4f', delimiter='\t')
        np.savetxt(direc + '/mems_1000.txt', interp_1d_array(self.mems, 1000), fmt='%.4f', delimiter='\t')
        np.savetxt(direc + '/coors.txt', self.coors, fmt='%.4f', delimiter='\t')
        saveimg(self.img, direc + '/img.tif')
        saveimg(self.straight, direc + '/straight.tif')
        saveimg(self.straight_filtered, direc + '/straight_filtered.tif')
        saveimg(self.straight_fit, direc + '/straight_fit.tif')
        saveimg(self.straight_mem, direc + '/straight_mem.tif')
        saveimg(self.straight_cyt, direc + '/straight_cyt.tif')
        saveimg(self.straight_resids, direc + '/straight_resids.tif')
        saveimg(self.straight_resids_pos, direc + '/straight_resids_pos.tif')
        saveimg(self.straight_resids_neg, direc + '/straight_resids_neg.tif')


######### REFERENCE PROFILES #########


def cytbg(img, coors, thickness, freedom=10):
    """
    Generates average cross-cortex profile for image of a cytoplasmic-only protein

    """

    # Straighten
    straight = straighten(img, coors, thickness + 2 * freedom)

    # Align
    straight_aligned = np.zeros([thickness, len(coors[:, 0])])
    for i in range(len(coors[:, 0])):
        profile = savgol_filter(straight[:, i], 11, 1)
        target = (np.mean(profile[:10]) + np.mean(profile[-10:])) / 2
        centre = (thickness / 2) + np.argmin(
            abs(profile[int(thickness / 2): int((thickness / 2) + (2 * freedom))] - target))
        straight_aligned[:, i] = straight[int(centre - thickness / 2): int(centre + thickness / 2), i]

    # Average
    return np.mean(straight_aligned, axis=1)


def membg(img, coors, thickness, freedom=10):
    """
    Generates average cross-cortex profile for image of a cortex-only protein

    Add interpolation to allow sub-pixel alignment?

    """

    # Straighten
    straight = straighten(img, coors, thickness + 2 * freedom)

    # Align
    straight_aligned = np.zeros([thickness, len(coors[:, 0])])
    for i in range(len(coors[:, 0])):
        profile = savgol_filter(straight[:, i], 9, 2)
        centre = (thickness / 2) + np.argmax(profile[int(thickness / 2): int(thickness / 2 + 2 * freedom)])
        straight_aligned[:, i] = straight[int(centre - thickness / 2): int(centre + thickness / 2), i]

    # Average
    return np.mean(straight_aligned, axis=1)


######## AF/BACKGROUND REMOVAL #######


def make_mask(shape, coors):
    return cv2.fillPoly(np.zeros(shape) * np.nan, [np.int32(coors)], 1)


def af_correlation_pbyp(img1, img2, mask, sigma=1, plot=None, c=None):
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
    a, resids, _, _, _ = np.polyfit(xdata, ydata, 1, full=True)

    # Scatter plot
    if plot == 'scatter':
        plt.scatter(xdata, ydata, s=0.001, c=c)
        xline = np.linspace(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99), 20)
        yline = a[0] * xline + a[1]
        plt.plot(xline, yline, c='r')
        plt.xlim(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99))
        plt.ylim(np.percentile(ydata, 0.01), np.percentile(ydata, 99.99))

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

    return a


def af_correlation_mean(img1, img2, mask, plot=None, c=None):
    """

    """
    # Mask
    img1 *= mask
    img2 *= mask

    # Average per embryo
    xdata = np.zeros([len(img1[0, 0, :])])
    ydata = np.zeros([len(img1[0, 0, :])])
    for e in range(len(img1[0, 0, :])):
        ydata[e] = np.nanmean(img1[:, :, e])
        xdata[e] = np.nanmean(img2[:, :, e])

    # Fit to line
    a, resids, _, _, _ = np.polyfit(xdata, ydata, 1, full=True)

    # Scatter plot
    if plot == 'scatter':
        plt.scatter(xdata, ydata, c=c)
        xline = np.linspace(np.percentile(xdata, 0.01), np.percentile(xdata, 99.99), 20)
        yline = a[0] * xline + a[1]
        plt.plot(xline, yline, c=c)

    else:
        pass

    return a


def af_correlation_pbyp_3channel(img1, img2, img3, mask, plot=None, ax=None, c=None):
    """
    AF correlation taking into account red channel

    :param img1: GFP channel
    :param img2: AF channel
    :param img3: RFP channel
    :param mask:
    :param plot:
    :return:
    """

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
    p, resids, rank, s = lstsq(np.c_[xdata, ydata, np.ones(len(xdata))], zdata)

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

    return p, ax


def af_correlation_mean_3channel(img1, img2, img3, mask, plot=None, ax=None, c=None):
    """
    AF correlation taking into account red channel

    :param img1: GFP channel
    :param img2: AF channel
    :param img3: RFP channel
    :param mask:
    :param plot:
    :return:
    """

    # Mask
    img1 *= mask
    img2 *= mask
    img3 *= mask

    # Average per embryo
    xdata = np.zeros([len(img1[0, 0, :])])
    ydata = np.zeros([len(img1[0, 0, :])])
    zdata = np.zeros([len(img1[0, 0, :])])
    for e in range(len(img1[0, 0, :])):
        ydata[e] = np.nanmean(img3[:, :, e])
        xdata[e] = np.nanmean(img2[:, :, e])
        zdata[e] = np.nanmean(img1[:, :, e])

    # Fit to surface
    p, resids, rank, s = lstsq(np.c_[xdata, ydata, np.ones(len(xdata))], zdata)

    # Scatter plot
    if plot == 'scatter':
        # Set up figure
        if not ax:
            ax = plt.figure().add_subplot(111, projection='3d')

        # Plot surface
        xx, yy = np.meshgrid([min(xdata), max(xdata)], [min(ydata), max(ydata)])
        zz = p[0] * xx + p[1] * yy + p[2]
        ax.plot_surface(xx, yy, zz, alpha=0.2, color=c)

        # Scatter plot
        ax.scatter(xdata, ydata, zdata, c=c)

        # Tidy plot
        ax.set_xlabel('AF')
        ax.set_ylabel('RFP')
        ax.set_zlabel('GFP')

    return p, ax


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


def bg_subtraction(img, coors, band=(25, 75)):
    a = polycrop(img, coors, band[1]) - polycrop(img, coors, band[0])
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


########## IMAGE HANDLING ###########


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


def saveimg_jpeg(img, direc, cmin=None, cmax=None):
    """
    Saves 2D array as jpeg, according to min and max pixel intensities

    :param img:
    :param direc:
    :param cmin:
    :param cmax:
    :return:
    """

    plt.imsave(direc, img, vmin=cmin, vmax=cmax, cmap='gray')


############## ROI #################

class ROI:
    def __init__(self, spline):

        # Inputs
        self.spline = spline
        self.fig = plt.gcf()
        self.ax = plt.gca()

        # Internal
        self._point0 = None
        self._points = None
        self._line = None
        self._fitted = False
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.ax.text(0.03, 0.88,
                     'Specify ROI clockwise from the posterior (4 points minimum)'
                     '\nBACKSPACE: undo'
                     '\nENTER: Proceed',
                     color='white',
                     transform=self.ax.transAxes, fontsize=8)

        # Outputs
        self.xpoints = []
        self.ypoints = []
        self.roi = None

        plt.show(block=True)

    def button_press_callback(self, event):
        if not self._fitted:
            if event.inaxes:
                # Add points to list
                self.xpoints.extend([event.xdata])
                self.ypoints.extend([event.ydata])

                # Display points
                self.display_points()
                self.fig.canvas.draw()

    def key_press_callback(self, event):
        if event.key == 'backspace':
            if not self._fitted:

                # Remove last drawn point
                if len(self.xpoints) != 0:
                    self.xpoints = self.xpoints[:-1]
                    self.ypoints = self.ypoints[:-1]
                self.display_points()
                self.fig.canvas.draw()
            else:

                # Remove line
                self._fitted = False
                self._line.pop(0).remove()
                self.roi = None
                self.fig.canvas.draw()

        if event.key == 'enter':
            if not self._fitted:
                coors = np.vstack((self.xpoints, self.ypoints)).T

                # Spline
                if self.spline:
                    self.roi = fit_spline(coors, periodic=True)

                # Linear interpolation
                else:
                    self.roi = interp_coors(coors, periodic=True)

                # Display line
                self._line = self.ax.plot(self.roi[:, 0], self.roi[:, 1], c='b')
                self.fig.canvas.draw()

                self._fitted = True
            else:
                # Close figure window
                plt.close(self.fig)

    def display_points(self):

        # Remove existing points
        try:
            self._point0.remove()
            self._points.remove()
        except (ValueError, AttributeError) as error:
            pass

        # Plot all points
        if len(self.xpoints) != 0:
            self._points = self.ax.scatter(self.xpoints, self.ypoints, c='b')
            self._point0 = self.ax.scatter(self.xpoints[0], self.ypoints[0], c='r')


def def_ROI(img, spline=True):
    """
    Instructions:
    - click to lay down points
    - backspace at any time to remove last point
    - press enter to select area (if spline=True will fit spline to points, otherwise will fit straight lines)
    - at this point can press backspace to go back to laying points
    - press enter again to close and return ROI

    :param img: input image (2d)
    :param spline: if true, fits spline to inputted coordinates
    :return: cell boundary coordinates
    """

    plt.imshow(img, cmap='gray', vmin=0)
    roi = ROI(spline)
    coors = roi.roi
    return coors


########### MISC FUNCTIONS ###########


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

    Combine with 2d function

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


def rolling_ave_1d(array, window, periodic=True):
    """

    :param array:
    :param window:
    :param periodic:
    :return:

    Combine with 2d function

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


def fit_spline(coors, periodic=True):
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
    return interp_coors(np.vstack((xi, yi)).T, periodic=periodic)


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


def asi(mems):
    """
    Calculates asymmetry index based on membrane concentration profile

    """

    ant = bounded_mean_1d(mems, (0.33, 0.67))
    post = bounded_mean_1d(mems, (0.83, 0.17))
    return (ant - post) / (2 * (ant + post))


def calc_dosage(mems, cyts, coors, c=0.7343937511951732):
    """
    Calculate total dosage based on membrane and cytoplasmic concentrations
    Relies on calibration factor (c) to relate cytoplasmic and cortical concentrations

    """

    # Normalise coors
    nc = norm_coors(coors)

    # Add cytoplasmic and cortical
    mbm = np.average(mems, weights=abs(nc[:, 1]))  # units x-1
    cym = np.average(cyts, weights=abs(nc[:, 1] ** 2))  # units x-2

    tot = cym + c * mbm  # units x-2
    return tot


def calc_vol(normcoors):
    r1 = max(normcoors[:, 0]) - min(normcoors[:, 0]) / 2
    r2 = max(normcoors[:, 1]) - min(normcoors[:, 1]) / 2
    return 4 / 3 * np.pi * r2 * r2 * r1


def calc_sa(normcoors):
    r1 = max(normcoors[:, 0]) - min(normcoors[:, 0]) / 2
    r2 = max(normcoors[:, 1]) - min(normcoors[:, 1]) / 2
    e = (1 - (r2 ** 2) / (r1 ** 2)) ** 0.5
    return 2 * np.pi * r2 * r2 * (1 + (r1 / (r2 * e)) * np.arcsin(e))


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


def rotated_embryo(img, coors, l, h=None):
    """
    Takes an image and rotates according to coordinates so that anterior is on left, posterior on right

    :param img:
    :param coors:
    :param l: length of each side in returned image
    :return:
    """

    if not h:
        h = l

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
    line1 = offset_line(line0, h / 2)
    line2 = offset_line(line0, -h / 2)
    end1 = np.array(
        [np.linspace(line1[0, 0], line2[0, 0], h), np.linspace(line1[0, 1], line2[0, 1], h)]).T
    end2 = np.array(
        [np.linspace(line1[1, 0], line2[1, 0], h), np.linspace(line1[1, 1], line2[1, 1], h)]).T

    # Get cross section
    num_points = l
    zvals = np.zeros([h, l])
    for section in range(h):
        xvalues = np.linspace(end1[section, 0], end2[section, 0], num_points)
        yvalues = np.linspace(end1[section, 1], end2[section, 1], num_points)
        zvals[section, :] = map_coordinates(img.T, [xvalues, yvalues])

    # Mirror
    zvals = np.fliplr(zvals)

    return zvals


########### OTHER #############

def params_load(direc):
    return pickle.load(open('%s/params.pkl' % direc, 'rb'))


def bar(ax, data, pos, c, size=5, label=None):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1, label=label)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=size)

    ax.set_xticks(list(ax.get_xticks()) + [pos])
    ax.set_xlim([0, max(ax.get_xticks()) + 4])


def direcslist(dest, levels=0, exclude=('!',), exclusive=None):
    """
    Gives a list of directories in a given directory (full path)


    :param dest:
    :param levels:
    :param exclude: exclude directories containing this string
    :param exclusive: exclude directories that don't contain this string
    :return:
    """
    lis = glob.glob('%s/*/' % dest)

    for level in range(levels):
        newlis = []
        for e in lis:
            newlis.extend(glob.glob('%s/*/' % e))
        lis = newlis
        lis = [x[:-1] for x in lis]

    if exclude is not None:
        for i in exclude:
            lis = [x for x in lis if i not in x]

    if exclusive is not None:
        for i in exclusive:
            lis = [x for x in lis if i in x]

    return lis


def importall(direcs, key):
    data = []
    for d in direcs:
        data.extend([np.loadtxt('%s/%s' % (d, key))])
    data = np.array(data)
    return data
