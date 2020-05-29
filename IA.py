import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution, brute
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import splprep, splev, CubicSpline
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import convolve
from scipy.linalg import lstsq
from scipy.special import erf
from skimage import io
from joblib import Parallel, delayed
import multiprocessing
import cv2
import os
import shutil
import random
import glob
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

"""
Functions for segmentation and quantification of membrane and cytoplasmic protein concentrations from midplane confocal 
images of C. elegans zygotes


To do:
- Double check straighten, offset_coors, interp_roi, rotate_roi, norm_roi
- New rotated_embryo function
- why is fitting slower in parallel?
- More annotations, detailed docstrings for functions etc.

"""


############# QUANT CLASS ############


class ImageQuant:
    """
    Quantification works by taking cross sections across the membrane, and fitting the resulting profile as the sum of
    a cytoplasmic signal component and a membrane signal component

    Input data:
    img                image

    Background curves:
    cytbg              cytoplasmic background curve, should be 2x as thick as thickness parameter
    membg              membrane background curve, as above
    sigma              if either of above are not specified, assume gaussian/error function with width set by sigma

    ROI:
    roi                coordinates defining cortex. Can use output from def_roi function

    Fitting parameters:
    freedom            amount of freedom allowed in ROI (0=min, 1=max, max offset is +- 0.5 * freedom * thickness)
    periodic           True if coordinates form a closed loop
    thickness          thickness of cross section over which to perform quantification
    itp                amount to interpolate image prior to segmentation (this many points per pixel in original image)
    rol_ave            width of rolling average
    resolution_cyt     for cytoplasmic fitting. Can get large performance increase by increasing this, at small cost to
                       accuracy
    rotate             if True, will automatically rotate ROI so that the first/last points are at the end of the long
                       axis
    zerocap            if True, prevents negative membane and cytoplasm values
    nfits              performs this many fits at regular intervals around ROI
    iterations         if >1, adjusts ROI and re-fits
    interp             interpolation type (linear or cubic)
    uni_cyt            globally fit uniform cytoplasm
    uni_mem            globally fit uniform membrane
    bg_subtract        if True, will estimate and subtract background signal prior to quantification

    Computation:
    parallel           TRUE = perform fitting in parallel
    cores              number of cores to use if parallel is True (if none will use all available)

    Saving:
    save_path          destination to save results, will create if it doesn't already exist


    """

    def __init__(self, img, cytbg=None, membg=None, sigma=None, roi=None, freedom=0.5,
                 periodic=True, thickness=50, itp=10, rol_ave=10, parallel=False, cores=None,
                 resolution_cyt=1, rotate=False, zerocap=True, nfits=None,
                 iterations=1, interp='cubic', save_path=None, bg_subtract=False, uni_cyt=False, uni_mem=False):

        # Image / stack
        self.img = img

        # ROI
        self.roi_init = roi
        self.roi = roi
        self.periodic = periodic

        # Background subtraction
        self.bg_subtract = bg_subtract

        # Fitting mode
        if not uni_cyt and not uni_mem:
            self.method = 0
        elif uni_cyt and not uni_mem:
            self.method = 1
        elif uni_cyt and uni_mem:
            self.method = 2
        else:
            self.method = None
            raise Exception('Uniform membrane not supported with non-uniform cytoplasm')

        # Fitting parameters
        self.iterations = iterations
        self.thickness = thickness
        self.itp = itp
        self.thickness_itp = int(itp * self.thickness)
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.resolution_cyt = resolution_cyt
        self.rotate = rotate
        self.zerocap = zerocap
        self.sigma = sigma
        self.nfits = nfits
        self.interp = interp

        # Saving
        self.save_path = save_path

        # Background curves
        if cytbg is None:
            self.cytbg = (1 + error_func(np.arange(thickness * 2), thickness, self.sigma)) / 2
            self.cytbg_itp = (1 + error_func(np.arange(2 * self.thickness_itp), self.thickness_itp,
                                             self.sigma * self.itp)) / 2
        else:
            self.cytbg = cytbg
            self.cytbg_itp = interp_1d_array(self.cytbg, 2 * self.thickness_itp)
        if membg is None:
            self.membg = gaus(np.arange(thickness * 2), thickness, self.sigma)
            self.membg_itp = gaus(np.arange(2 * self.thickness_itp), self.thickness_itp, self.sigma * self.itp)
        else:
            self.membg = membg
            self.membg_itp = interp_1d_array(self.membg, 2 * self.thickness_itp)

        # Check for appropriate thickness
        if len(self.cytbg) != len(self.membg):
            raise Exception('Error: cytbg and membg must be the same length, and thickness must be half this length')
        elif self.thickness != 0.5 * len(self.cytbg):
            raise Exception('Error: thickness must be exactly half the length of the background curves')

        # Computation
        self.parallel = parallel
        if cores is not None:
            self.cores = cores
        else:
            self.cores = multiprocessing.cpu_count()

        # Results containers
        self.offsets = None
        self.cyts = None
        self.mems = None
        self.offsets_full = None
        self.cyts_full = None
        self.mems_full = None

        # Simulated images
        self.straight = None
        self.straight_filtered = None
        self.straight_fit = None
        self.straight_mem = None
        self.straight_cyt = None
        self.straight_resids = None
        self.straight_resids_pos = None
        self.straight_resids_neg = None

        if self.roi is not None:
            self.reset_res()

    """
    Run

    """

    def run(self):

        # Fitting
        self.fit()
        if self.iterations > 1:
            for i in range(self.iterations - 1):
                self.adjust_roi()
                self.fit()

        # Simulate images
        self.sim_images()

        # Save
        self.save()

    def fit(self):

        # Specify number of fits
        if self.nfits is None:
            self.nfits = len(self.roi[:, 0])

        # Straighten image
        self.straight = straighten(self.img, self.roi, self.thickness)

        # Background subtract
        if self.bg_subtract:
            self.straight -= np.mean(self.straight[:5, :])

        # Smoothen
        if self.rol_ave != 0:
            self.straight_filtered = rolling_ave_2d(self.straight, self.rol_ave, self.periodic)
        else:
            self.straight_filtered = self.straight

        # Interpolate
        straight = interp_2d_array(self.straight_filtered, self.thickness_itp, method=self.interp)
        straight = interp_2d_array(straight, self.nfits, ax=0, method=self.interp)

        # Fit
        if self.method == 0:
            """
            Non-uniform cytoplasm and non-uniform membrane
            
            """

            if self.parallel:
                results = np.array(Parallel(n_jobs=self.cores)(
                    delayed(self._fit_profile)(straight[:, x]) for x in range(len(straight[0, :]))))
                self.offsets = results[:, 0]
                self.cyts = results[:, 1]
                self.mems = results[:, 2]
            else:
                for x in range(len(straight[0, :])):
                    self.offsets[x], self.cyts[x], self.mems[x] = self._fit_profile(straight[:, x])

        elif self.method == 1:
            """
            Uniform cytoplasm, non-uniform membrane
            
            """
            # Fit uniform cytoplasm
            c = self._fit_profile_1(straight)
            self.cyts[:] = c

            # Fit local membranes
            if self.parallel:
                results = np.array(Parallel(n_jobs=self.cores)(
                    delayed(self._fit_profile_2)(straight[:, x], c) for x in range(len(straight[0, :]))))
                self.offsets = results[:, 0]
                self.mems = results[:, 1]
            else:
                for x in range(len(straight[0, :])):
                    self.offsets[x], self.mems[x] = self._fit_profile_2(straight[:, x], c)

        elif self.method == 2:
            """
            Uniform cytoplasm and uniform membrane
            
            """

            # Fit uniform cytoplasm, uniform membrane
            c, m = self._fit_profile_ucum(straight)
            self.mems[:] = m
            self.cyts[:] = c

            # Fit local offsets
            if self.parallel:
                self.offsets = np.array(Parallel(n_jobs=self.cores)(
                    delayed(self._fit_profile_ucum_2)(straight[:, x], c, m) for x in range(len(straight[0, :]))))
            else:
                for x in range(len(straight[0, :])):
                    self.offsets[x] = self._fit_profile_ucum_2(straight[:, x], c, m)

        # Interpolate
        self.offsets_full = interp_1d_array(self.offsets, len(self.roi[:, 0]), method=self.interp)
        self.cyts_full = interp_1d_array(self.cyts, len(self.roi[:, 0]), method=self.interp)
        self.mems_full = interp_1d_array(self.mems, len(self.roi[:, 0]), method=self.interp)

    """
    METHOD 0: Non-uniform cytoplasm

    """

    def _fit_profile(self, profile):
        if self.zerocap:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (0, 2 * max(profile)), (0, 2 * max(profile)))
        else:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (-0.2 * max(profile), 2 * max(profile)), (-0.2 * max(profile), 2 * max(profile)))
        res = differential_evolution(self._mse, bounds=bounds, args=(profile,), tol=0.2)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o, res.x[1], res.x[2]

    # def _fit_profile(self, profile, cytbg, membg):
    #     # This works but is slower
    #     bounds = (
    #         ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
    #         (0, 2 * max(profile)), (0, 2 * max(profile)))
    #     res = iterative_opt_multi(self._mse, bounds=bounds, args=(profile, cytbg, membg), N=5, iterations=4)
    #     o = (res[0] - self.thickness_itp / 2) / self.itp
    #     return o, res[1], res[2]

    def _mse(self, l_c_m, profile):
        l, c, m = l_c_m
        y = (c * self.cytbg_itp[int(l):int(l) + self.thickness_itp]) + (
            m * self.membg_itp[int(l):int(l) + self.thickness_itp])
        return np.mean((profile - y) ** 2)

    """
    METHOD 1: Uniform cytoplasm

    """

    def _fit_profile_1(self, straight):
        """
        For finding optimal global cytoplasm

        """

        if self.zerocap:
            bounds = (0, 2 * np.percentile(straight, 95))
        else:
            bounds = (-0.2 * np.percentile(straight, 95), 2 * np.percentile(straight, 95))

        res = iterative_opt(self._fit_profile_1_func, p_range=bounds, args=(straight,), N=5, iterations=7)

        return res[0]

    def _fit_profile_1_func(self, c, straight):
        if self.parallel:
            mses = np.array(Parallel(n_jobs=self.cores)(
                delayed(self._fit_profile_2b)(straight[:, x * self.resolution_cyt], c)
                for x in range(len(straight[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight[0, :]) // self.resolution_cyt)
            for x in range(len(straight[0, :]) // self.resolution_cyt):
                mses[x] = self._fit_profile_2b(straight[:, x * self.resolution_cyt], c)
        return np.mean(mses)

    def _fit_profile_2(self, profile, c):
        """
        For finding optimal local membrane, alignment
        Returns offset

        """

        if self.zerocap:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (0, 2 * max(profile)))
        else:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (-0.2 * max(profile), 2 * max(profile)))
        res = iterative_opt_2d(self._fit_profile_2_func, p1_range=bounds[0], p2_range=bounds[1], N=5, iterations=7,
                               args=(profile, c))
        o = (res[0] - self.thickness_itp / 2) / self.itp
        return o, res[1]

    def _fit_profile_2b(self, profile, c):
        """
        For finding optimal local membrane, alignment
        Returns _mse

        """
        if self.zerocap:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (0, 2 * max(profile)))
        else:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (-0.2 * max(profile), 2 * max(profile)))
        res = iterative_opt_2d(self._fit_profile_2_func, p1_range=bounds[0], p2_range=bounds[1], N=5, iterations=7,
                               args=(profile, c))
        return res[2]

    def _fit_profile_2_func(self, l_m, profile, c):
        l, m = l_m
        y = (c * self.cytbg_itp[int(l):int(l) + self.thickness_itp]) + (
            m * self.membg_itp[int(l):int(l) + self.thickness_itp])
        return np.mean((profile - y) ** 2)

    """
    METHOD 2: Uniform cytoplasm, uniform membrane

    """

    def _fit_profile_ucum(self, straight):
        """
        Fitting global cytoplasmic and cortical concs

        """

        if self.zerocap:
            bounds = (0, 2 * np.percentile(straight, 95))
        else:
            bounds = (-0.2 * np.percentile(straight, 95), 2 * np.percentile(straight, 95))
        res = iterative_opt_2d(self._fit_profile_ucum_func, p1_range=bounds,
                               p2_range=(0, 2 * np.percentile(straight, 95)), args=(straight,), N=5,
                               iterations=5)

        return res[0], res[1]

    def _fit_profile_ucum_func(self, c_m, straight):
        """


        """

        c, m = c_m
        if self.parallel:
            mses = np.array(Parallel(n_jobs=self.cores)(
                delayed(self._fit_profile_ucum_2b)(straight[:, x * self.resolution_cyt], c, m)
                for x in range(len(straight[0, :]) // self.resolution_cyt)))
        else:
            mses = np.zeros(len(straight[0, :]) // self.resolution_cyt)
            for x in range(len(straight[0, :]) // self.resolution_cyt):
                mses[x] = self._fit_profile_ucum_2b(straight[:, x * self.resolution_cyt], c, m)
        return np.mean(mses)

    def _fit_profile_ucum_2(self, profile, c, m):
        """
        Fitting local offsets, returns offsets

        """

        res, fun = iterative_opt(self._fit_profile_ucum_2_func, p_range=(
            (self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                                 args=(profile, c, m), N=5, iterations=3)

        o = (res - self.thickness_itp / 2) / self.itp
        return o

    def _fit_profile_ucum_2b(self, profile, c, m):
        """
        Fitting local offsets, returns error

        """

        res, fun = iterative_opt(self._fit_profile_ucum_2_func, p_range=(
            (self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                                 args=(profile, c, m), N=5, iterations=3)

        return fun

    def _fit_profile_ucum_2_func(self, l, profile, c, m):
        y = (c * self.cytbg_itp[int(l):int(l) + self.thickness_itp]) + (
            m * self.membg_itp[int(l):int(l) + self.thickness_itp])
        return np.mean((profile - y) ** 2)

    """
    Misc

    """

    def sim_images(self):
        """
        Creates simulated images based on fit results

        """
        for x in range(len(self.roi[:, 0])):
            c = self.cyts_full[x]
            m = self.mems_full[x]
            l = int(self.offsets_full[x] * self.itp + (self.thickness_itp / 2))
            self.straight_cyt[:, x] = interp_1d_array(c * self.cytbg_itp[l:l + self.thickness_itp], self.thickness,
                                                      method=self.interp)
            self.straight_mem[:, x] = interp_1d_array(m * self.membg_itp[l:l + self.thickness_itp], self.thickness,
                                                      method=self.interp)
            self.straight_fit[:, x] = interp_1d_array(
                (c * self.cytbg_itp[l:l + self.thickness_itp]) + (m * self.membg_itp[l:l + self.thickness_itp]),
                self.thickness, method=self.interp)
            self.straight_resids[:, x] = self.straight[:, x] - self.straight_fit[:, x]
            self.straight_resids_pos[:, x] = np.clip(self.straight_resids[:, x], a_min=0, a_max=None)
            self.straight_resids_neg[:, x] = abs(np.clip(self.straight_resids[:, x], a_min=None, a_max=0))

    def adjust_roi(self):
        """
        Can do after a preliminary fit to refine coordinates
        Must refit after doing this

        """

        # Interpolate, remove nans
        offsets = self.offsets
        nans, x = np.isnan(offsets), lambda z: z.nonzero()[0]
        offsets[nans] = np.interp(x(nans), x(~nans), offsets[~nans])
        offsets = interp_1d_array(offsets, len(self.roi))

        # Offset coordinates
        self.roi = offset_coordinates(self.roi, offsets)

        # Filter
        if self.periodic:
            self.roi = np.vstack(
                (savgol_filter(self.roi[:, 0], 19, 1, mode='wrap'),
                 savgol_filter(self.roi[:, 1], 19, 1, mode='wrap'))).T
        elif not self.periodic:
            self.roi = np.vstack(
                (savgol_filter(self.roi[:, 0], 19, 1, mode='nearest'),
                 savgol_filter(self.roi[:, 1], 19, 1, mode='nearest'))).T

        # Interpolate to one px distance between points
        self.roi = interp_roi(self.roi, self.periodic)

        # Rotate
        if self.periodic:
            if self.rotate:
                self.roi = rotate_roi(self.roi)

        # Reset
        self.reset_res()

    def reset(self):
        """
        Resets entire class to its initial state

        """

        self.roi = self.roi_init
        self.reset_res()

    def reset_res(self):
        """
        Clears results

        """

        if self.nfits is None:
            self.nfits = len(self.roi[:, 0])

        # Results
        self.offsets = np.zeros(self.nfits)
        self.cyts = np.zeros(self.nfits)
        self.mems = np.zeros(self.nfits)

        # Interpolated results
        self.offsets_full = np.zeros(len(self.roi[:, 0]))
        self.cyts_full = np.zeros(len(self.roi[:, 0]))
        self.mems_full = np.zeros(len(self.roi[:, 0]))

        # Simulated images
        self.straight = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_filtered = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_fit = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_mem = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_cyt = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_resids = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_resids_pos = np.zeros([self.thickness, len(self.roi[:, 0])])
        self.straight_resids_neg = np.zeros([self.thickness, len(self.roi[:, 0])])

    def save(self):
        """
        Save all results to save_path

        """

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        np.savetxt(self.save_path + '/offsets.txt', self.offsets, fmt='%.4f', delimiter='\t')
        np.savetxt(self.save_path + '/cyts.txt', self.cyts, fmt='%.4f', delimiter='\t')
        np.savetxt(self.save_path + '/mems.txt', self.mems, fmt='%.4f', delimiter='\t')
        # np.savetxt(direc + '/cyts_1000.txt', interp_1d_array(self.cyts, 1000), fmt='%.4f', delimiter='\t')
        # np.savetxt(direc + '/mems_1000.txt', interp_1d_array(self.mems, 1000), fmt='%.4f', delimiter='\t')
        np.savetxt(self.save_path + '/roi.txt', self.roi, fmt='%.4f', delimiter='\t')
        save_img(self.img, self.save_path + '/img.tif')
        save_img(self.straight, self.save_path + '/straight.tif')
        save_img(self.straight_filtered, self.save_path + '/straight_filtered.tif')
        save_img(self.straight_fit, self.save_path + '/straight_fit.tif')
        save_img(self.straight_mem, self.save_path + '/straight_mem.tif')
        save_img(self.straight_cyt, self.save_path + '/straight_cyt.tif')
        save_img(self.straight_resids, self.save_path + '/straight_resids.tif')
        save_img(self.straight_resids_pos, self.save_path + '/straight_resids_pos.tif')
        save_img(self.straight_resids_neg, self.save_path + '/straight_resids_neg.tif')


class StackQuant:
    """
    Wrapper for ImageQuant designed to allow easy quantification of tif stacks (can also quantify single frames)
    Takes 2D or 3D numpy array as input (output from load_image function)

    See ImageQuant documentation for parameter definitions

    """

    def __init__(self, img, save_path, parallel=False, cores=None, start_frame=None, end_frame=None,
                 **kwargs):

        # Image / stack
        self.img = img
        if len(self.img.shape) == 3:
            self.stack = True
        else:
            self.stack = False

        # Frames
        if self.stack:
            if start_frame is None:
                self.start_frame = 0
            else:
                self.start_frame = start_frame
            if end_frame is None:
                self.end_frame = self.img.shape[0]
            else:
                self.end_frame = end_frame

        # Destination
        self.save_path = save_path

        # Computation
        self.parallel = parallel
        if cores is not None:
            self.cores = cores
        else:
            self.cores = multiprocessing.cpu_count()

        # kwargs
        self.kwargs = kwargs

    def run(self):

        # Single frame
        if not self.stack:
            ImageQuant(self.img, parallel=False, cores=None, save_path=self.save_path,
                       **self.kwargs).run()

        # Stack
        else:
            if self.parallel:
                pool = multiprocessing.Pool(self.cores)
                pool.map(self.analyse_frame, range(self.start_frame, self.end_frame + 1))
            else:
                for i in range(self.start_frame, self.end_frame + 1):
                    self.analyse_frame(i)

    def analyse_frame(self, i):

        # Create directory
        direc = self.save_path + '/' + str(i).zfill(3)
        if os.path.isdir(direc):
            shutil.rmtree(direc)
        os.mkdir(direc)

        # Run
        ImageQuant(img=self.img[i, :, :], parallel=False, cores=None, save_path=direc,
                   **self.kwargs).run()

    def view_stack(self):
        view_stack(self.img)

    def plot_quantification(self):
        plot_quantification(self.save_path)

    def plot_fits(self):
        plot_fits(self.save_path)

    def plot_segmentation(self):
        plot_segmentation(self.save_path)

    def def_roi(self):
        r = ROI(self.img, spline=True)
        self.roi = r.roi

    def compile_res(self):
        compile_res(self.save_path)


class StackQuantGUI:
    """
    Graphical user interface for StackQuant

    """

    def __init__(self):

        """
        Input data

        """

        self.img = None
        self.cytbg = None
        self.membg = None

        """
        Parameters

        """
        self.stack = None
        self.ROI = None
        self.sigma = None
        self.nfits = None
        self.rol_ave = None
        self.start_frame = None
        self.end_frame = None
        self.interpolation_type = None
        self.iterations = None
        self.thickness = None
        self.freedom = None
        self.periodic = None
        self.parallel = None
        self.bg_subtract = None
        self.uni_cyt = None
        self.uni_mem = None
        self.mode = None

        """
        Help text

        """
        with open('docs/GUIhelp.txt', 'r') as file:
            self.help_info = file.read()

        """
        Clear cache

        """
        self.cachedirec = os.path.dirname(os.path.realpath(__file__)) + '/.caches/StackQuantGUI'
        shutil.rmtree(self.cachedirec)
        os.mkdir(self.cachedirec)

        """
        Open window

        """
        self.window = tk.Tk()
        self.window.resizable(width=False, height=False)

        """
        Upper buttons

        """

        self.button_open = tk.Button(master=self.window, text='Import tif...', command=self.button_open_event)
        self.label_nframes = tk.Label(master=self.window, text='')

        self.button_cytbg = tk.Button(master=self.window, text='Import cytoplasmic profile...',
                                      command=self.button_cytbg_event)
        self.label_cytbg = tk.Label(master=self.window, text='')

        self.button_membg = tk.Button(master=self.window, text='Import membrane profile...',
                                      command=self.button_membg_event)
        self.label_membg = tk.Label(master=self.window, text='')

        """
        Basic parameters inputs

        """

        self.label_sigma = tk.Label(master=self.window, text='Sigma')
        self.entry_sigma = tk.Spinbox(master=self.window,
                                      values=["{:.1f}".format(i) for i in list(np.arange(0.1, 10, 0.1))])
        self.entry_sigma.delete(0, 'end')
        self.entry_sigma.insert(0, '3.0')

        self.label_nfits = tk.Label(master=self.window, text='Number of fits')
        self.entry_nfits = tk.Entry(master=self.window)
        self.entry_nfits.insert(0, '100')

        self.label_rolave = tk.Label(master=self.window, text='Roling average window')
        self.entry_rolave = tk.Spinbox(master=self.window, values=list(range(0, 110, 10)))
        self.entry_rolave.delete(0, 'end')
        self.entry_rolave.insert(0, '10')

        self.label_start = tk.Label(master=self.window, text='Start frame')
        self.entry_start = tk.Spinbox(master=self.window, values=list(range(0, 10, 1)))
        self.entry_start.delete(0, 'end')
        self.entry_start.insert(0, '0')

        self.label_end = tk.Label(master=self.window, text='End frame')
        self.entry_end = tk.Spinbox(master=self.window, values=list(range(0, 10, 1)))
        self.entry_end.delete(0, 'end')
        self.entry_end.insert(0, '0')

        # self.label_interp = tk.Label(master=self.window, text='Interpolation type')
        # self.options_interp = ['cubic', 'linear']
        # self.variable_interp = StringVar(self.window)
        # self.variable_interp.set(self.options_interp[0])
        # self.menu_interp = tk.OptionMenu(self.window, self.variable_interp, *self.options_interp)

        self.label_iterations = tk.Label(master=self.window, text='Iterations')
        self.entry_iterations = tk.Spinbox(master=self.window, values=list(range(1, 100, 1)))
        self.entry_iterations.delete(0, 'end')
        self.entry_iterations.insert(0, '2')

        self.label_thickness = tk.Label(master=self.window, text='Thickness')
        self.entry_thickness = tk.Spinbox(master=self.window, values=list(range(10, 110, 10)))
        self.entry_thickness.delete(0, 'end')
        self.entry_thickness.insert(0, '50')

        self.label_freedom = tk.Label(master=self.window, text='ROI freedom')
        self.entry_freedom = tk.Spinbox(master=self.window,
                                        values=["{:.1f}".format(i) for i in list(np.arange(0, 1.1, 0.1))])
        self.entry_freedom.delete(0, 'end')
        self.entry_freedom.insert(0, '0.5')

        self.label_periodic = tk.Label(master=self.window, text='Periodic ROI')
        self.var_periodic = tk.IntVar(value=1)
        self.checkbutton_periodic = tk.Checkbutton(master=self.window, variable=self.var_periodic)

        self.label_parallel = tk.Label(master=self.window, text='Parallel processing')
        self.var_parallel = tk.IntVar(value=1)
        self.checkbutton_parallel = tk.Checkbutton(master=self.window, variable=self.var_parallel)

        self.label_bg = tk.Label(master=self.window, text='Subtract background')
        self.var_bg = tk.IntVar(value=1)
        self.checkbutton_bg = tk.Checkbutton(master=self.window, variable=self.var_bg)

        """
        Advanced parameters inputs

        """

        self.label_unicyt = tk.Label(master=self.window, text='Uniform cytoplasm')
        self.var_unicyt = tk.IntVar(value=0)
        self.checkbutton_unicyt = tk.Checkbutton(master=self.window, variable=self.var_unicyt)

        self.label_unimem = tk.Label(master=self.window, text='Uniform membrane')
        self.var_unimem = tk.IntVar(value=0)
        self.checkbutton_unimem = tk.Checkbutton(master=self.window, variable=self.var_unimem)

        """
        Lower buttons

        """

        self.button_view = tk.Button(master=self.window, text='View', command=self.button_view_event)
        self.button_ROI = tk.Button(master=self.window, text='Specify ROI', command=self.button_ROI_event)
        self.label_ROI = tk.Label(master=self.window, text='')
        self.button_run = tk.Button(master=self.window, text='Run quantification', command=self.button_run_event)
        self.label_running = tk.Label(master=self.window, text='')

        self.button_quant = tk.Button(master=self.window, text='View membrane quantification',
                                      command=self.button_quant_event)
        self.button_fits = tk.Button(master=self.window, text='View local fits', command=self.button_fits_event)
        self.button_seg = tk.Button(master=self.window, text='View segmentation', command=self.button_seg_event)
        self.button_save = tk.Button(master=self.window, text='Save to csv...', command=self.button_save_event)
        self.button_mode = tk.Button(master=self.window)
        self.button_help = tk.Button(master=self.window, text='Help', command=self.button_help_event)

        """
        Lay out grid

        """

        self.button_open.grid(row=0, column=0, sticky='W', padx=10, pady=5)
        self.label_nframes.grid(row=0, column=1, sticky='W', padx=10, pady=5)

        self.button_cytbg.grid(row=1, column=0, sticky='W', padx=10, pady=5)
        self.label_cytbg.grid(row=1, column=1, sticky='W', padx=10, pady=5)

        self.button_membg.grid(row=2, column=0, sticky='W', padx=10, pady=5)
        self.label_membg.grid(row=2, column=1, sticky='W', padx=10, pady=5)

        self.label_start.grid(row=3, column=0, sticky='W', padx=10)
        self.entry_start.grid(row=3, column=1, sticky='W', padx=10)
        self.label_end.grid(row=4, column=0, sticky='W', padx=10)
        self.entry_end.grid(row=4, column=1, sticky='W', padx=10)

        self.label_sigma.grid(row=5, column=0, sticky='W', padx=10)
        self.entry_sigma.grid(row=5, column=1, sticky='W', padx=10)
        self.label_nfits.grid(row=6, column=0, sticky='W', padx=10)
        self.entry_nfits.grid(row=6, column=1, sticky='W', padx=10)
        self.label_rolave.grid(row=7, column=0, sticky='W', padx=10)
        self.entry_rolave.grid(row=7, column=1, sticky='W', padx=10)

        # self.label_interp.grid(row=8, column=0, sticky='W', padx=30)
        # self.menu_interp.grid(row=8, column=1, sticky='W', padx=10)
        self.label_iterations.grid(row=9, column=0, sticky='W', padx=10)
        self.entry_iterations.grid(row=9, column=1, sticky='W', padx=10)
        self.label_thickness.grid(row=10, column=0, sticky='W', padx=10)
        self.entry_thickness.grid(row=10, column=1, sticky='W', padx=10)
        self.label_freedom.grid(row=11, column=0, sticky='W', padx=10)
        self.entry_freedom.grid(row=11, column=1, sticky='W', padx=10)
        self.label_periodic.grid(row=12, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_periodic.grid(row=12, column=1, sticky='W', padx=10, pady=3)
        self.label_bg.grid(row=13, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_bg.grid(row=13, column=1, sticky='W', padx=10, pady=3)
        self.label_parallel.grid(row=14, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_parallel.grid(row=14, column=1, sticky='W', padx=10, pady=3)
        self.label_unicyt.grid(row=15, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_unicyt.grid(row=15, column=1, sticky='W', padx=10, pady=3)
        self.label_unimem.grid(row=16, column=0, sticky='W', padx=10, pady=3)
        self.checkbutton_unimem.grid(row=16, column=1, sticky='W', padx=10, pady=3)

        self.button_view.grid(row=19, column=0, sticky='W', padx=10, pady=5)
        self.button_ROI.grid(row=20, column=0, sticky='W', padx=10, pady=5)
        self.label_ROI.grid(row=20, column=1, sticky='W', padx=10, pady=5)
        self.button_run.grid(row=21, column=0, sticky='W', padx=10, pady=5)
        self.label_running.grid(row=21, column=1, sticky='W', padx=10, pady=5)
        self.button_quant.grid(row=22, column=0, sticky='W', padx=10, pady=5)
        self.button_fits.grid(row=23, column=0, sticky='W', padx=10, pady=5)
        self.button_seg.grid(row=24, column=0, sticky='W', padx=10, pady=5)
        self.button_mode.grid(row=24, column=1, sticky='E', padx=10, pady=5)
        self.button_save.grid(row=25, column=0, sticky='W', padx=10, pady=5)
        self.button_help.grid(row=25, column=1, sticky='E', padx=10, pady=5)

        """
        Start window

        """

        self.toggle_set1('disable')
        self.button_run.configure(state='disable')
        self.toggle_set2('disable')
        self.button_mode_to_basic()  # basic mode by default
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    """
    Button functions

    """

    def button_open_event(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(master=root)
        root.destroy()
        # self.window.lift()

        # Load image
        self.img = load_image(file_path)

        # Clear cache
        shutil.rmtree(self.cachedirec)
        os.mkdir(self.cachedirec)

        # Activate set 1
        self.toggle_set1('normal')

        # Disable run + set 2
        self.button_run.configure(state='disable')
        self.toggle_set2('disable')
        self.label_running.config(text='')
        self.label_ROI.config(text='')

        # Set frame ranges
        if len(self.img.shape) == 3:
            self.stack = True
            self.entry_start.configure(values=list(range(0, self.img.shape[0], 1)))
            self.entry_end.configure(values=list(range(0, self.img.shape[0], 1)))
            self.entry_start.delete(0, 'end')
            self.entry_start.insert(0, 0)
            self.entry_end.delete(0, 'end')
            self.entry_end.insert(0, str(self.img.shape[0] - 1))
            self.label_nframes.config(text='%s frames loaded' % self.img.shape[0])
        else:
            self.stack = False
            self.entry_start.delete(0, 'end')
            self.entry_start.insert(0, 0)
            self.entry_end.delete(0, 'end')
            self.entry_end.insert(0, 0)
            self.entry_start.configure(state='disable')
            self.entry_end.configure(state='disable')
            self.label_start.configure(state='disable')
            self.label_end.configure(state='disable')
            self.label_nframes.config(text='1 frame loaded')

    def button_cytbg_event(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(master=root)
        root.destroy()

        # Load cytbg
        self.cytbg = np.loadtxt(file_path)

        # Update window
        self.label_cytbg.config(text='Cytoplasmic profile loaded')
        if self.cytbg is not None and self.membg is not None:
            self.entry_sigma.configure(state='normal')
            self.label_sigma.configure(state='normal')
            self.entry_sigma.delete(0, 'end')
            self.entry_sigma.configure(state='disable')
            self.label_sigma.configure(state='disable')

    def button_membg_event(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(master=root)
        root.destroy()

        # Load cytbg
        self.membg = np.loadtxt(file_path)

        # Update window
        self.label_membg.config(text='Membrane profile loaded')
        if self.cytbg is not None and self.membg is not None:
            self.entry_sigma.configure(state='normal')
            self.label_sigma.configure(state='normal')
            self.entry_sigma.delete(0, 'end')
            self.entry_sigma.configure(state='disable')
            self.label_sigma.configure(state='disable')

    def button_help_event(self):
        popup = tk.Tk()
        popup.wm_title('Help')
        popup.geometry('500x750')
        popup.resizable(width=False, height=False)

        text_box = tk.Text(master=popup, wrap=tk.WORD)
        text_box.insert('1.0', self.help_info)
        text_box.config(state=tk.DISABLED)
        text_box.pack(expand=True, fill='both')
        popup.mainloop()

    def button_view_event(self):
        view_stack(self.img)

    def button_mode_to_basic(self):
        self.mode = 0
        self.window.title('Basic mode')

        self.button_mode.config(text='Advanced mode', command=self.button_mode_to_advanced)
        self.button_cytbg.grid_remove()
        self.label_cytbg.grid_remove()
        self.button_membg.grid_remove()
        self.label_membg.grid_remove()
        self.label_unicyt.grid_remove()
        self.checkbutton_unicyt.grid_remove()
        self.label_unimem.grid_remove()
        self.checkbutton_unimem.grid_remove()

        self.entry_sigma.configure(state='normal')
        self.label_sigma.configure(state='normal')
        self.entry_sigma.delete(0, 'end')
        self.entry_sigma.insert(0, '3.0')
        if self.img is None:
            self.entry_sigma.configure(state='disabled')
            self.label_sigma.configure(state='disabled')

    def button_mode_to_advanced(self):
        self.mode = 1
        self.window.title('Advanced mode')

        self.button_mode.config(text='Basic mode', command=self.button_mode_to_basic)
        self.button_cytbg.grid()
        self.label_cytbg.grid()
        self.button_membg.grid()
        self.label_membg.grid()
        self.label_unicyt.grid()
        self.checkbutton_unicyt.grid()
        self.label_unimem.grid()
        self.checkbutton_unimem.grid()

        if self.cytbg is not None and self.membg is not None:
            self.entry_sigma.configure(state='normal')
            self.label_sigma.configure(state='normal')
            self.entry_sigma.delete(0, 'end')
            self.entry_sigma.configure(state='disable')
            self.label_sigma.configure(state='disable')

    def button_ROI_event(self):
        self.ROI = def_roi(self.img, start_frame=int(self.entry_start.get()), end_frame=int(self.entry_end.get()),
                           periodic=bool(self.var_periodic.get()))
        if self.ROI is not None:
            self.label_ROI.configure(text='ROI saved')
            self.button_run.configure(state='normal')

    def button_run_event(self):

        # Get parameters
        if self.entry_sigma.get() != '':
            self.sigma = float(self.entry_sigma.get())
        self.nfits = int(self.entry_nfits.get())
        self.rol_ave = int(self.entry_rolave.get())
        self.start_frame = int(self.entry_start.get())
        self.end_frame = int(self.entry_end.get())
        # self.interpolation_type = self.variable_interp.get()
        self.interpolation_type = 'cubic'
        self.iterations = int(self.entry_iterations.get())
        self.thickness = int(self.entry_thickness.get())
        self.freedom = float(self.entry_freedom.get())
        self.periodic = bool(self.var_periodic.get())
        self.parallel = bool(self.var_parallel.get())
        self.bg_subtract = bool(self.var_bg.get())
        self.uni_cyt = bool(self.var_unicyt.get())
        self.uni_mem = bool(self.var_unimem.get())

        # Clear cache
        shutil.rmtree(self.cachedirec)
        os.mkdir(self.cachedirec)

        try:
            # Set up quantifier class
            if self.mode == 0:  # basic mode
                q = StackQuant(img=self.img, roi=self.ROI, thickness=self.thickness, rol_ave=self.rol_ave,
                               freedom=self.freedom, parallel=self.parallel, nfits=self.nfits, sigma=self.sigma,
                               iterations=self.iterations, save_path=self.cachedirec, start_frame=self.start_frame,
                               end_frame=self.end_frame, interp=self.interpolation_type, periodic=self.periodic,
                               bg_subtract=self.bg_subtract, uni_cyt=False, uni_mem=False)

            else:  # advanced mode
                q = StackQuant(img=self.img, roi=self.ROI, thickness=self.thickness, rol_ave=self.rol_ave,
                               freedom=self.freedom, parallel=self.parallel, nfits=self.nfits,
                               iterations=self.iterations, save_path=self.cachedirec, start_frame=self.start_frame,
                               end_frame=self.end_frame, interp=self.interpolation_type, periodic=self.periodic,
                               bg_subtract=self.bg_subtract, cytbg=self.cytbg, membg=self.membg, uni_cyt=self.uni_cyt,
                               uni_mem=self.uni_mem, sigma=self.sigma)

            # Update window
            self.toggle_set2('disable')
            self.label_running.config(text='Running...')
            self.window.update()

            # Run
            q.run()

            # Update window
            self.toggle_set2('normal')
            self.label_running.config(text='Complete!')

        except Exception as e:
            print(e)
            self.label_running.config(text='Failed (check terminal)')

    def button_quant_event(self):
        plot_quantification(self.cachedirec)

    def button_fits_event(self):
        plot_fits(self.cachedirec)

    def button_seg_event(self):
        plot_segmentation(self.cachedirec)

    def button_save_event(self):
        # Pick save destination
        root = tk.Tk()
        root.withdraw()
        root.update()
        f = filedialog.asksaveasfile(master=root, mode='w', initialfile='Quantification.csv')
        root.destroy()

        # Compile results
        res = compile_res(self.cachedirec)

        # Save
        res.to_csv(f)

    """
    Enable / disable widgets

    """

    def toggle_set1(self, state):

        self.label_sigma.configure(state=state)
        self.entry_sigma.configure(state=state)
        self.label_nfits.configure(state=state)
        self.entry_nfits.configure(state=state)
        self.label_rolave.configure(state=state)
        self.entry_rolave.configure(state=state)
        self.label_start.configure(state=state)
        self.entry_start.configure(state=state)
        self.label_end.configure(state=state)
        self.entry_end.configure(state=state)

        # self.label_interp.configure(state=state)
        # self.menu_interp.configure(state=state)
        self.label_iterations.configure(state=state)
        self.entry_iterations.configure(state=state)
        self.label_thickness.configure(state=state)
        self.entry_thickness.configure(state=state)
        self.label_freedom.configure(state=state)
        self.entry_freedom.configure(state=state)
        self.label_periodic.configure(state=state)
        self.checkbutton_periodic.configure(state=state)
        self.label_parallel.configure(state=state)
        self.checkbutton_parallel.configure(state=state)
        self.label_bg.configure(state=state)
        self.checkbutton_bg.configure(state=state)

        self.label_unicyt.configure(state=state)
        self.checkbutton_unicyt.configure(state=state)
        self.label_unimem.configure(state=state)
        self.checkbutton_unimem.configure(state=state)

        self.button_view.configure(state=state)
        self.button_ROI.configure(state=state)

    def toggle_set2(self, state):
        self.button_quant.configure(state=state)
        self.button_fits.configure(state=state)
        self.button_seg.configure(state=state)
        self.button_save.configure(state=state)

    """
    Shutdown

    """

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Clear cache
            shutil.rmtree(self.cachedirec)
            os.mkdir(self.cachedirec)

            # Close
            plt.close('all')
            self.window.destroy()


def compile_res(direc):
    # Create empty dataframe
    df = pd.DataFrame({'Frame': [],
                       'Position': [],
                       'Membrane signal': [],
                       'Cytoplasmic signal': []})

    # Detect if single frame or stack
    if len(glob.glob('%s/*/' % direc)) != 0:
        stack = True
    else:
        stack = False

    if not stack:
        mems = np.loadtxt(direc + '/mems.txt')
        cyts = np.loadtxt(direc + '/cyts.txt')

        df = df.append(pd.DataFrame({'Frame': 0,
                                     'Position': range(len(mems)),
                                     'Membrane signal': mems,
                                     'Cytoplasmic signal': cyts}))

    else:
        start_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[0])))
        end_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[-1])))
        for i in range(start_frame, end_frame + 1):
            mems = np.loadtxt(direc + '/' + str(i).zfill(3) + '/mems.txt')
            cyts = np.loadtxt(direc + '/' + str(i).zfill(3) + '/cyts.txt')

            df = df.append(pd.DataFrame({'Frame': i,
                                         'Position': range(len(mems)),
                                         'Membrane signal': mems,
                                         'Cytoplasmic signal': cyts}))

    df = df.reindex(columns=['Frame', 'Position', 'Membrane signal', 'Cytoplasmic signal'])
    df = df.astype({'Frame': int, 'Position': int})
    return df


########### INTERACTIVE #############

def view_stack(img, start_frame=0, end_frame=None):
    """
    Interactive stack viewer

    """

    # Detect if single frame or stack
    if len(img.shape) == 3:
        stack = True
    else:
        stack = False

    # Set up figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # ax.set_xlim(0, 512)
    # ax.set_ylim(0, 512)

    # Calculate intensity ranges
    vmin, vmax = [1, 1.1] * np.percentile(img.flatten(), [0.1, 99.9])

    # Stack
    if stack:
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        if end_frame is None:
            end_frame = len(img[:, 0, 0]) - 1
        sframe = Slider(axframe, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')

        def update(i):
            # xlim = ax.get_xlim()
            # ylim = ax.get_ylim()
            ax.clear()
            ax.imshow(img[int(i), :, :], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xlim(*xlim)
            # ax.set_ylim(*ylim)

        sframe.on_changed(update)
        update(start_frame)

    # Single frame
    else:
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.canvas.set_window_title('')
    plt.show(block=True)


class ROI:
    """
    Instructions:
    - click to lay down points
    - backspace at any time to remove last point
    - press enter to select area (if spline=True will fit spline to points, otherwise will fit straight lines)
    - at this point can press backspace to go back to laying points
    - press enter again to close and return ROI

    :param img: input image
    :param spline: if true, fits spline to inputted coordinates
    :return: cell boundary coordinates
    """

    def __init__(self, img, spline, start_frame=0, end_frame=None, periodic=True):

        self.img = img
        self.spline = spline
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.periodic = periodic

        # Internal
        self._point0 = None
        self._points = None
        self._line = None
        self._fitted = False

        # Outputs
        self.xpoints = []
        self.ypoints = []
        self.roi = None

        # Set up figure
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        # Calculate intensity ranges
        self.vmin, self.vmax = [1, 1.1] * np.percentile(self.img.flatten(), [0.1, 99.9])

        # Stack
        if len(self.img.shape) == 3:
            plt.subplots_adjust(left=0.25, bottom=0.25)
            self.axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
            if self.end_frame is None:
                self.end_frame = len(self.img[:, 0, 0]) - 1
            self.sframe = Slider(self.axframe, 'Frame', self.start_frame, self.end_frame, valinit=self.start_frame,
                                 valfmt='%d')
            self.sframe.on_changed(self.select_frame)
            self.select_frame(self.start_frame)

        # Single frame
        else:
            self.ax.imshow(self.img, cmap='gray', vmin=self.vmin, vmax=self.vmax)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.text(0.03, 0.97,
                         'Specify ROI clockwise (4 points minimum)'
                         '\nClick to lay points'
                         '\nBACKSPACE: undo'
                         '\nENTER: Save and continue',
                         color='white',
                         transform=self.ax.transAxes, fontsize=8, va='top', ha='left')

        # Show figure
        self.fig.canvas.set_window_title('Specify ROI')
        self.fig.canvas.mpl_connect('close_event', lambda event: self.fig.canvas.stop_event_loop())
        self.fig.canvas.start_event_loop(timeout=-1)

    def select_frame(self, i):
        self.ax.clear()
        self.ax.imshow(self.img[int(i), :, :], cmap='gray', vmin=self.vmin, vmax=self.vmax)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.03, 0.97,
                     'Specify ROI clockwise (4 points minimum)'
                     '\nClick to lay points'
                     '\nBACKSPACE: undo'
                     '\nENTER: Save and continue',
                     color='white',
                     transform=self.ax.transAxes, fontsize=8, va='top', ha='left')
        self.display_points()
        self.fig.canvas.draw()

    def button_press_callback(self, event):
        if not self._fitted:
            if isinstance(event.inaxes, type(self.ax)):
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
                roi = np.vstack((self.xpoints, self.ypoints)).T

                # Spline
                if self.spline:
                    self.roi = spline_roi(roi, periodic=self.periodic)

                # Display line
                self._line = self.ax.plot(self.roi[:, 0], self.roi[:, 1], c='b')
                self.fig.canvas.draw()

                self._fitted = True

                # print(self.roi)

                plt.close(self.fig)  # comment this out to see spline fit
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
            self._points = self.ax.scatter(self.xpoints, self.ypoints, c='lime', s=10)
            self._point0 = self.ax.scatter(self.xpoints[0], self.ypoints[0], c='r', s=10)


def def_roi(img, spline=True, start_frame=0, end_frame=None, periodic=True):
    r = ROI(img, spline=spline, start_frame=start_frame, end_frame=end_frame, periodic=periodic)
    return r.roi


def plot_segmentation(direc):
    """
    Plot segmentation results

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Detect if single frame or stack
    if len(glob.glob('%s/*/' % direc)) != 0:
        stack = True
        start_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[0])))
        end_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[-1])))
    else:
        stack = False

    # Preload all data to specify ylim
    if stack:
        ylim_top = 0
        ylim_bottom = 0
        for i in range(start_frame, end_frame + 1):
            straight = load_image(direc + '/' + str(int(i)).zfill(3) + '/img.tif')
            ylim_top = max([ylim_top, np.max(straight)])
            ylim_bottom = min([ylim_top, np.min(straight)])
    else:
        straight = load_image(direc + '/img.tif')
        ylim_top = np.max(straight)
        ylim_bottom = np.min(straight)

    # Single frame
    if not stack:
        img = load_image(direc + '/img.tif')
        roi = np.loadtxt(direc + '/roi.txt')
        ax.imshow(img, cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
        ax.plot(roi[:, 0], roi[:, 1], c='lime')
        ax.set_xticks([])
        ax.set_yticks([])

    # Stack
    else:

        # Add frame slider
        plt.subplots_adjust(left=0.25, bottom=0.25)
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')

        def update(i):
            ax.clear()
            img = load_image(direc + '/' + str(int(i)).zfill(3) + '/img.tif')
            roi = np.loadtxt(direc + '/' + str(int(i)).zfill(3) + '/roi.txt')
            ax.imshow(img, cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
            ax.plot(roi[:, 0], roi[:, 1], c='lime')
            ax.set_xticks([])
            ax.set_yticks([])

        sframe.on_changed(update)
        update(start_frame)

    fig.canvas.set_window_title('Segmentation')
    plt.show(block=True)


def plot_quantification(direc):
    """
    Plot quantification results

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Detect if single frame or stack
    if len(glob.glob('%s/*/' % direc)) != 0:
        stack = True
    else:
        stack = False

    # Single frame
    if not stack:
        mems = np.loadtxt(direc + '/mems.txt')
        ax.plot(mems)
        ax.set_xlabel('Position')
        ax.set_ylabel('Membrane concentration')
        ax.set_ylim(bottom=0)

    # Stack
    else:

        start_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[0])))
        end_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[-1])))

        # Preload all data to specify ylim
        ylim_top = 0
        for i in range(start_frame, end_frame + 1):
            mems = np.loadtxt(direc + '/' + str(int(i)).zfill(3) + '/mems.txt')
            ylim_top = max([ylim_top, np.max(mems)])

        # Add frame silder
        plt.subplots_adjust(left=0.25, bottom=0.25)
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')

        def update(i):
            ax.clear()
            mems = np.loadtxt(direc + '/' + str(int(i)).zfill(3) + '/mems.txt')
            ax.plot(mems)
            ax.set_xlabel('Position')
            ax.set_ylabel('Membrane concentration')
            ax.set_ylim(0, ylim_top)

        sframe.on_changed(update)
        update(start_frame)

    fig.canvas.set_window_title('Membrane Quantification')
    plt.show(block=True)


class FitPlotter:
    def __init__(self, direc):
        self.direc = direc

        # Internal variables
        self.straight = None
        self.straight_cyt = None
        self.straight_mem = None
        self.cyts = None
        self.mems = None
        self.cyts_interp = None
        self.mems_interp = None
        self.pos = 10

        # Set up figure
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(3, 3)
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.fig.add_subplot(gs[1:, :])

        # Detect if single frame or stack
        if len(glob.glob('%s/*/' % direc)) != 0:
            stack = True
            start_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[0])))
            end_frame = int(os.path.basename(os.path.normpath(sorted(glob.glob('%s/*/' % direc))[-1])))
        else:
            stack = False

        # Preload all data to specify ylim
        if stack:
            self.ylim_top = 0
            self.ylim_bottom = 0
            for i in range(start_frame, end_frame + 1):
                straight = load_image(direc + '/' + str(int(i)).zfill(3) + '/straight_filtered.tif')
                straight_fit = load_image(direc + '/' + str(int(i)).zfill(3) + '/straight_fit.tif')
                self.ylim_top = max([self.ylim_top, np.max(straight), np.max(straight_fit)])
                self.ylim_bottom = min([self.ylim_top, np.min(straight), np.min(straight_fit)])
        else:
            straight = load_image(direc + '/straight_filtered.tif')
            straight_fit = load_image(direc + '/straight_fit.tif')
            self.ylim_top = max([np.max(straight), np.max(straight_fit)])
            self.ylim_bottom = min([np.min(straight), np.min(straight_fit)])

        # Frame slider
        if stack:
            plt.subplots_adjust(bottom=0.25, left=0.25)
            axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
            slider_frame = Slider(axframe, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')
            slider_frame.on_changed(lambda f: self.update_frame(self.direc + '/' + str(int(f)).zfill(3)))

        # Initial plot
        if not stack:
            self.update_frame(self.direc)
        else:
            self.update_frame(self.direc + '/' + str(int(start_frame)).zfill(3))

        # Show
        self.fig.canvas.set_window_title('Local fits')
        plt.show(block=True)

    def update_pos(self, p):
        self.pos = int(p)
        self.ax1_update()
        self.ax2_update()

    def update_frame(self, direc):
        self.straight = load_image(direc + '/straight_filtered.tif')
        self.straight_cyt = load_image(direc + '/straight_cyt.tif')
        self.straight_mem = load_image(direc + '/straight_mem.tif')
        self.cyts = np.loadtxt(direc + '/cyts.txt')
        self.mems = np.loadtxt(direc + '/mems.txt')
        self.cyts_interp = interp_1d_array(self.cyts, self.straight.shape[1])
        self.mems_interp = interp_1d_array(self.mems, self.straight.shape[1])

        # Position slider
        self.slider_pos = Slider(self.ax1, '', 0, len(self.straight[0, :]), valinit=self.pos, valfmt='%d',
                                 facecolor='none', edgecolor='none')
        self.slider_pos.on_changed(self.update_pos)

        self.ax1_update()
        self.ax2_update()

    def ax1_update(self):
        self.ax1.clear()
        self.ax1.imshow(self.straight, cmap='gray', vmin=self.ylim_bottom, vmax=1.1 * self.ylim_top)
        self.ax1.axvline(self.pos, c='r')
        self.ax1.set_xticks([0, len(self.cyts_interp)])
        self.ax1.set_xticklabels([0, len(self.cyts) - 1])
        self.ax1.set_yticks([])
        self.ax1.set_xlabel('Position')
        self.ax1.xaxis.set_label_position('top')

    def ax2_update(self):
        self.ax2.clear()
        self.ax2.plot(self.straight[:, self.pos], label='Actual')
        self.ax2.plot(self.straight_mem[:, self.pos] + self.straight_cyt[:, self.pos], label='Fit')
        # self.ax2.plot(self.straight_cyt[:, self.pos], label='Cytoplasmic component')
        # self.ax2.plot(self.straight_mem[:, self.pos], label='Membrane component')
        self.ax2.set_xticks([])
        self.ax2.set_ylabel('Intensity')
        self.ax2.legend(frameon=False, loc='upper left', fontsize='small')
        self.ax2.set_ylim(bottom=self.ylim_bottom, top=self.ylim_top)
        self.ax2.set_xlabel('Membrane signal (a.u.): %s\nCytoplasmic signal (a.u.): %s' % (
            "{0:.1f}".format(self.mems_interp[self.pos]), "{0:.1f}".format(self.cyts_interp[self.pos])), fontsize=8,
                            ha='left', x=0)


def plot_fits(direc):
    FitPlotter(direc)


########## FITTING ALGORITHMS ###########

def iterative_opt(func, p_range, N, iterations, args=()):
    """

    :param func: function to be minimised
    :param p_range: initial parameter range
    :param args:
    :param N: number of points to be evaluated per iteration
    :param iterations: number of iterations
    :return:
    """

    params = np.linspace(p_range[0], p_range[1], N)
    func_calls = 0

    for i in range(iterations):

        if i == 0:
            res = np.zeros([N])
            for i in range(N):
                res[i] = func(params[i], *args)
                func_calls += 1

        else:
            for i in range(1, N - 1):
                res[i] = func(params[i], *args)
                func_calls += 1
        a = np.argmin(res)
        fun = res[a]

        if a == 0:
            params = np.linspace(params[0], params[1], N)
            res = np.r_[res[0], np.zeros([N - 2]), res[1]]
        elif a == N - 1:
            params = np.linspace(params[-2], params[-1], N)
            res = np.r_[res[-2], np.zeros([N - 2]), res[-1]]
        else:
            params = np.linspace(params[a - 1], params[a + 1], N)
            res = np.r_[res[a - 1], np.zeros([N - 2]), res[a + 1]]
    return params[a], fun


def iterative_opt_2d(func, p1_range, p2_range, N, iterations, args=()):
    """

    :param func: function to be minimised
    :param p1_range: initial parameter range
    :param p2_range: initial parameter range
    :param args:
    :param N: number of points to be evaluated per iteration
    :param iterations: number of iterations
    :return:
    """

    params_1 = np.linspace(p1_range[0], p1_range[1], N)
    params_2 = np.linspace(p2_range[0], p2_range[1], N)
    func_calls = 0

    for i in range(iterations):
        res = np.zeros([N, N])

        for p1 in range(N):
            for p2 in range(N):
                res[p1, p2] = func((params_1[p1], params_2[p2]), *args)
                func_calls += 1

        a = np.where(res == np.min(res))
        a_1 = a[0][0]
        a_2 = a[1][0]

        if a_1 == 0:
            params_1 = np.linspace(params_1[0], params_1[1], N)
        elif a_1 == N - 1:
            params_1 = np.linspace(params_1[-2], params_1[-1], N)
        else:
            params_1 = np.linspace(params_1[a_1 - 1], params_1[a_1 + 1], N)

        if a_2 == 0:
            params_2 = np.linspace(params_2[0], params_2[1], N)
        elif a_2 == N - 1:
            params_2 = np.linspace(params_2[-2], params_2[-1], N)
        else:
            params_2 = np.linspace(params_2[a_2 - 1], params_2[a_2 + 1], N)

    return params_1[a[0][0]], params_2[a[1][0]], res[a][0]


def iterative_opt_multi(func, bounds, N, iterations, args=()):
    """

    :param func: function to minimise
    :param bounds: tuple of tuples specifying lower and upper bound for each parameter
    :param N: number of parameters to test per iteration (N**len(bounds))
    :param iterations: number of iterations
    :param args: optional additional arguments for func
    :return:

    Evaluates function on a grid of N**len(bounds) parameter combinations, before further exploration around the global
    minimum
    Assumes smooth funnel-like energy landscape with single minimum
    Not suitable for functions with a large number of parameters, use something like scipy.differential_evolution
    instead

    """

    for i in range(iterations):
        x0 = brute(func, bounds, Ns=N, args=args)
        bounds = list(bounds)
        for i, x in enumerate(x0):
            bounds[i] = list(bounds[i])
            gap = (bounds[i][1] - bounds[i][0]) / N
            if x - gap > bounds[i][0]:
                bounds[i][0] = x - gap
            if x + gap < bounds[i][1]:
                bounds[i][1] = x + gap
            bounds[i] = tuple(bounds[i])
        bounds = tuple(bounds)

    return x0


######### REFERENCE PROFILES #########


def generate_cytbg(img, roi, thickness, freedom=10):
    """
    Generates average cross-cortex profile for image of a cytoplasmic-only protein

    """

    # Straighten
    straight = straighten(img, roi, thickness + 2 * freedom)

    # Align
    straight_aligned = np.zeros([thickness, len(roi[:, 0])])
    for i in range(len(roi[:, 0])):
        profile = savgol_filter(straight[:, i], 11, 1)
        target = (np.mean(profile[:10]) + np.mean(profile[-10:])) / 2
        centre = (thickness / 2) + np.argmin(
            abs(profile[int(thickness / 2): int((thickness / 2) + (2 * freedom))] - target))
        straight_aligned[:, i] = straight[int(centre - thickness / 2): int(centre + thickness / 2), i]

    # Average
    return np.mean(straight_aligned, axis=1)


def generate_membg(img, roi, thickness, freedom=10):
    """
    Generates average cross-cortex profile for image of a cortex-only protein

    Add interpolation to allow sub-pixel alignment?

    """

    # Straighten
    straight = straighten(img, roi, thickness + 2 * freedom)

    # Align
    straight_aligned = np.zeros([thickness, len(roi[:, 0])])
    for i in range(len(roi[:, 0])):
        profile = savgol_filter(straight[:, i], 9, 2)
        centre = (thickness / 2) + np.argmax(profile[int(thickness / 2): int(thickness / 2 + 2 * freedom)])
        straight_aligned[:, i] = straight[int(centre - thickness / 2): int(centre + thickness / 2), i]

    # Average
    return np.mean(straight_aligned, axis=1)


######## AF/BACKGROUND REMOVAL #######


def make_mask(shape, roi):
    return cv2.fillPoly(np.zeros(shape) * np.nan, [np.int32(roi)], 1)


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


def bg_subtraction(img, roi, band=(25, 75)):
    a = polycrop(img, roi, band[1]) - polycrop(img, roi, band[0])
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


########## IMAGE HANDLING ###########


def load_image(filename):
    """
    Given the filename of a TIFF, creates numpy array with pixel intensities

    :param filename:
    :return:
    """

    # img = np.array(Image.open(filename), dtype=np.float64)
    # img[img == 0] = np.nan
    return io.imread(filename)


def save_img(img, direc):
    """
    Saves 2D array as .tif file

    :param img:
    :param direc:
    :return:
    """

    io.imsave(direc, img.astype('float32'))

    # im = Image.fromarray(img)
    # im.save(direc)


def save_img_jpeg(img, direc, cmin=None, cmax=None):
    """
    Saves 2D array as jpeg, according to min and max pixel intensities

    :param img:
    :param direc:
    :param cmin:
    :param cmax:
    :return:
    """

    plt.imsave(direc, img, vmin=cmin, vmax=cmax, cmap='gray')


########### MISC FUNCTIONS ###########


def straighten(img, roi, thickness):
    """
    Creates straightened image based on coordinates

    Ability to specify interpolation type?

    :param img:
    :param roi: Coordinates. Should be 1 pixel length apart in a loop
    :param thickness:
    :return:
    """

    img2 = np.zeros([thickness, len(roi[:, 0])])
    offsets = np.linspace(thickness / 2, -thickness / 2, thickness)
    # newcoors_x = np.zeros([thickness, len(coors[:, 0])])
    # newcoors_y = np.zeros([thickness, len(coors[:, 0])])

    for section in range(thickness):
        sectioncoors = offset_coordinates(roi, offsets[section])
        a = map_coordinates(img.T, [sectioncoors[:, 0], sectioncoors[:, 1]])
        a[a == 0] = np.mean(a)  # if selection goes outside of the image
        img2[section, :] = a
        # newcoors_x[section, :] = sectioncoors[:, 0]
        # newcoors_y[section, :] = sectioncoors[:, 1]
    return img2


def offset_coordinates(roi, offsets):
    """
    Reads in coordinates, adjusts according to offsets

    :param roi: two column array containing x and y coordinates. e.g. coors = np.loadtxt(filename)
    :param offsets: array the same length as coors. Direction?
    :return: array in same format as coors containing new coordinates

    To save this in a fiji readable format run:
    np.savetxt(filename, newcoors, fmt='%.4f', delimiter='\t')

    """

    xcoors = roi[:, 0]
    ycoors = roi[:, 1]

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


def interp_1d_array(array, n, method='cubic'):
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


def interp_2d_array(array, n, ax=1, method='cubic'):
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


def interp_roi(roi, periodic=True):
    """
    Interpolates coordinates to one pixel distances (or as close as possible to one pixel)
    Linear interpolation

    Ability to specify number of points

    :param roi:
    :return:
    """

    if periodic:
        c = np.append(roi, [roi[0, :]], axis=0)
    else:
        c = roi

    # Calculate distance between points in pixel units
    distances = ((np.diff(c[:, 0]) ** 2) + (np.diff(c[:, 1]) ** 2)) ** 0.5
    distances_cumsum = np.append([0], np.cumsum(distances))
    px = sum(distances) / round(sum(distances))  # effective pixel size
    newpoints = np.zeros((int(round(sum(distances))), 2))
    newcoors_distances_cumsum = 0

    for d in range(int(round(sum(distances)))):
        index = sum(distances_cumsum - newcoors_distances_cumsum <= 0) - 1
        newpoints[d, :] = (
            roi[index, :] + ((newcoors_distances_cumsum - distances_cumsum[index]) / distances[index]) * (
                c[index + 1, :] - c[index, :]))
        newcoors_distances_cumsum += px

    return newpoints


def rotate_roi(roi):
    """
    Rotates coordinate array so that most posterior point is at the beginning

    """

    # PCA to find long axis
    M = (roi - np.mean(roi.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M)

    # Find most extreme points
    a = np.argmin(np.minimum(score[0, :], score[1, :]))
    b = np.argmax(np.maximum(score[0, :], score[1, :]))

    # Find the one closest to user defined posterior
    dista = np.hypot((roi[0, 0] - roi[a, 0]), (roi[0, 1] - roi[a, 1]))
    distb = np.hypot((roi[0, 0] - roi[b, 0]), (roi[0, 1] - roi[b, 1]))

    # Rotate coordinates
    if dista < distb:
        newcoors = np.roll(roi, len(roi[:, 0]) - a, 0)
    else:
        newcoors = np.roll(roi, len(roi[:, 0]) - b, 0)

    return newcoors


def spline_roi(roi, periodic=True):
    """
    Fits a spline to points specifying the coordinates of the cortex, then interpolates to pixel distances

    :param roi:
    :return:
    """

    # Append the starting x,y coordinates
    if periodic:
        x = np.r_[roi[:, 0], roi[0, 0]]
        y = np.r_[roi[:, 1], roi[0, 1]]
    else:
        x = roi[:, 0]
        y = roi[:, 1]

    # Fit spline
    tck, u = splprep([x, y], s=0, per=periodic)

    # Evaluate spline
    xi, yi = splev(np.linspace(0, 1, 1000), tck)

    # Interpolate
    return interp_roi(np.vstack((xi, yi)).T, periodic=periodic)


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


def norm_roi(roi):
    """
    Aligns coordinates to their long axis

    :param roi:
    :return:
    """

    # PCA
    M = (roi - np.mean(roi.T, axis=1)).T
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


def asi(mems):
    """
    Calculates asymmetry index based on membrane concentration profile

    """

    ant = bounded_mean_1d(mems, (0.33, 0.67))
    post = bounded_mean_1d(mems, (0.83, 0.17))
    return (ant - post) / (2 * (ant + post))


def calc_dosage(mems, cyts, roi, c=0.7343937511951732):
    """
    Calculate total dosage based on membrane and cytoplasmic concentrations
    Relies on calibration factor (c) to relate cytoplasmic and cortical concentrations

    """

    # Normalise coors
    nc = norm_roi(roi)

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


# def rotated_embryo(img, coors, l, h=None):
#     """
#
#     Need to develop a new method for this
#     Ability to specify interpolation type
#
#     Takes an image and rotates according to coordinates so that anterior is on left, posterior on right
#
#     :param img:
#     :param coors:
#     :param l: length of each side in returned image
#     :return:
#     """
#
#     if not h:
#         h = l
#
#     # PCA
#     M = (coors - np.mean(coors.T, axis=1)).T
#     [latent, coeff] = np.linalg.eig(np.cov(M))
#     score = np.dot(coeff.T, M)
#
#     # Find ends
#     a = np.argmin(np.minimum(score[0, :], score[1, :]))
#     b = np.argmax(np.maximum(score[0, :], score[1, :]))
#
#     # Find the one closest to user defined posterior
#     dista = np.hypot((coors[0, 0] - coors[a, 0]), (coors[0, 1] - coors[a, 1]))
#     distb = np.hypot((coors[0, 0] - coors[b, 0]), (coors[0, 1] - coors[b, 1]))
#
#     if dista < distb:
#         line0 = np.array([coors[a, :], coors[b, :]])
#     else:
#         line0 = np.array([coors[b, :], coors[a, :]])
#
#     # Extend line
#     length = np.hypot((line0[0, 0] - line0[1, 0]), (line0[0, 1] - line0[1, 1]))
#     line0 = extend_line(line0, l / length)
#
#     # Thicken line
#     line1 = offset_line(line0, h / 2)
#     line2 = offset_line(line0, -h / 2)
#     end1 = np.array(
#         [np.linspace(line1[0, 0], line2[0, 0], h), np.linspace(line1[0, 1], line2[0, 1], h)]).T
#     end2 = np.array(
#         [np.linspace(line1[1, 0], line2[1, 0], h), np.linspace(line1[1, 1], line2[1, 1], h)]).T
#
#     # Get cross section
#     num_points = l
#     zvals = np.zeros([h, l])
#     for section in range(h):
#         xvalues = np.linspace(end1[section, 0], end2[section, 0], num_points)
#         yvalues = np.linspace(end1[section, 1], end2[section, 1], num_points)
#         zvals[section, :] = map_coordinates(img.T, [xvalues, yvalues])
#
#     # Mirror
#     zvals = np.fliplr(zvals)
#
#     return zvals


def gaus(x, centre, width):
    """
    Create Gaussian curve with centre and width specified

    """
    return np.exp(-((x - centre) ** 2) / (2 * width ** 2))


def error_func(x, centre, width):
    """
    Create error function with centre and width specified

    """

    return erf((x - centre) / width)
