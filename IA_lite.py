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
import os
import shutil
import glob

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
    coors              coordinates defining cortex. Can use output from def_ROI function

    Parameters:
    thickness          thickness of cross section over which to perform segmentation
    rol_ave            width of rolling average
    mem_sigma          width of membrane gaussian
    itp                interpolates profiles before fitting, allows sub-pixel alignment
    freedom            from 0 - 0.5, specifies how much free alignment is permitted
    end_fix            this many pixels from each end of the profile will be fixed to the background curve
    parallel           TRUE = perform fitting in parallel using all available cores
    destination        location to save txt and tif files, must be created beforehand

    """

    def __init__(self, img, cytbg, coors=None, thickness=50, rol_ave=20, mem_sigma=3, itp=10, freedom=0.3, end_fix=10,
                 parallel=True, destination=None):

        self.img = img * 0.0001
        self.coors_init = coors
        self.coors = coors
        self.periodic = True
        self.thickness = thickness
        self.itp = itp
        self.thickness_itp = int(self.itp * self.thickness)
        self.cytbg = cytbg
        self.membg = gaus(np.arange(thickness * 2), thickness, mem_sigma)
        self.cytbg_itp = interp_1d_array(self.cytbg, 2 * self.thickness_itp)
        self.membg_itp = interp_1d_array(self.membg, 2 * self.thickness_itp)
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.parallel = parallel
        self.end_fix = int(self.itp * end_fix)
        self.rotate = False
        self.destination = destination

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

    """
    Fitting
    
    """

    def fit(self):

        # Filter/smoothen/interpolate straight image
        self.straight = straighten(self.img, self.coors, self.thickness)
        self.straight_filtered = rolling_ave_2d(self.straight, self.rol_ave, self.periodic)
        straight = interp_2d_array(self.straight_filtered, self.thickness_itp, method='cubic')

        if self.parallel:
            results = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self._fit_profile)(straight[:, x], self.cytbg_itp, self.membg_itp)
                for x in range(len(straight[0, :]))))
            self.offsets = interp_1d_array(results[:, 0], len(self.coors[:, 0]))
            self.mems = interp_1d_array(results[:, 1], len(self.coors[:, 0]))
        else:
            for x in range(len(straight[0, :])):
                results = self._fit_profile(straight[:, x], self.cytbg_itp, self.membg_itp)
                self.offsets[x] = results[0]
                self.mems[x] = results[1]

        self.sim_images()

    def _fit_profile(self, profile, cytbg, membg):
        bounds = (
            ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
            (0, 2 * max(profile)))
        res = differential_evolution(self._mse, bounds=bounds, args=(profile, cytbg, membg), tol=0.1)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o, res.x[1]

    def _mse(self, l_m, profile, cytbg, membg):
        l, m = l_m
        y = self.total_profile(profile, cytbg, membg, l=l, m=m)
        return np.mean((profile - y) ** 2)

    def total_profile(self, profile, cytbg, membg, l, m):
        cyt = cytbg[int(l):int(l) + len(profile)]
        mem = membg[int(l):int(l) + len(profile)]
        p = np.polyfit([np.mean(cyt[:self.end_fix]), np.mean(cyt[-self.end_fix:])],
                       [np.mean(profile[:self.end_fix]), np.mean(profile[-self.end_fix:])], 1)
        return (p[0] * cyt + p[1]) + (m * mem)

    def cyt_profile(self, profile, cytbg, membg, l, m):
        cyt = cytbg[int(l):int(l) + len(profile)]
        p = np.polyfit([np.mean(cyt[:self.end_fix]), np.mean(cyt[-self.end_fix:])],
                       [np.mean(profile[:self.end_fix]), np.mean(profile[-self.end_fix:])], 1)
        return p[0] * cyt + p[1]

    def mem_profile(self, profile, cytbg, membg, l, m):
        mem = membg[int(l):int(l) + len(profile)]
        return m * mem

    """
    Misc

    """

    def sim_images(self):
        """
        Creates simulated images based on fit results

        """
        straight = interp_2d_array(self.straight_filtered, self.thickness_itp, method='cubic')

        for x in range(len(self.coors[:, 0])):
            m = self.mems[x]
            l = int(self.offsets[x] * self.itp + (self.thickness_itp / 2))
            self.straight_cyt[:, x] = interp_1d_array(
                self.cyt_profile(straight[:, x], self.cytbg_itp, self.membg_itp, l=l, m=m), self.thickness)
            self.straight_mem[:, x] = interp_1d_array(
                self.mem_profile(straight[:, x], self.cytbg_itp, self.membg_itp, l=l, m=m), self.thickness)
            self.straight_fit[:, x] = interp_1d_array(
                self.total_profile(straight[:, x], self.cytbg_itp, self.membg_itp, l=l, m=m), self.thickness)
            self.straight_resids[:, x] = self.straight[:, x] - self.straight_fit[:, x]

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
        if self.periodic:
            self.coors = np.vstack(
                (savgol_filter(self.coors[:, 0], 19, 1, mode='wrap'),
                 savgol_filter(self.coors[:, 1], 19, 1, mode='wrap'))).T
        elif not self.periodic:
            self.coors = np.vstack(
                (savgol_filter(self.coors[:, 0], 19, 1, mode='nearest'),
                 savgol_filter(self.coors[:, 1], 19, 1, mode='nearest'))).T

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

    def save(self):
        """
        Save all results in specific directory

        """

        np.savetxt(self.destination + '/mems.txt', self.mems, fmt='%.4f', delimiter='\t')
        np.savetxt(self.destination + '/mems_1000.txt', interp_1d_array(self.mems, 1000), fmt='%.4f', delimiter='\t')
        np.savetxt(self.destination + '/coors.txt', self.coors, fmt='%.4f', delimiter='\t')
        saveimg(self.img, self.destination + '/img.tif')
        saveimg(self.straight, self.destination + '/straight.tif')
        saveimg(self.straight_filtered, self.destination + '/straight_filtered.tif')
        saveimg(self.straight_fit, self.destination + '/straight_fit.tif')
        saveimg(self.straight_mem, self.destination + '/straight_mem.tif')
        saveimg(self.straight_cyt, self.destination + '/straight_cyt.tif')
        saveimg(self.straight_resids, self.destination + '/straight_resids.tif')

    """
    Run
    
    """

    def run(self):
        self.fit()
        self.adjust_coors()
        self.fit()
        self.save()


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


########### AF REMOVAL ###########


def make_mask(shape, coors):
    return cv2.fillPoly(np.zeros(shape) * np.nan, [np.int32(coors)], 1)


def af_correlation(img1, img2, mask, sigma=1, plot=None, c=None):
    """

    Calculates pixel-by-pixel correlation between two channels
    Takes 3d image stacks shape [512, 512, n] or single images shape [512, 512]

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


def af_subtraction(ch1, ch2, m, c):
    """
    Subtract ch2 from ch1
    ch2 is first adjusted to m * ch2 + c

    :param ch1: gfp channel image
    :param ch2: af channel image
    :param m:
    :param c:
    :return:
    """

    af = m * ch2 + c
    signal = ch1 - af
    return signal


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
                     '\nBACKSPACE: Undo'
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

    for section in range(thickness):
        sectioncoors = offset_coordinates(coors, offsets[section])
        a = map_coordinates(img.T, [sectioncoors[:, 0], sectioncoors[:, 1]])
        a[a == 0] = np.mean(a)  # if selection goes outside of the image
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
    :param method:
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
    :param method:
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


def gaus(x, centre, width):
    """
    Create Gaussian curve with centre and width specified

    """
    return np.exp(-((x - centre) ** 2) / (2 * width ** 2))


def direcslist(parent, levels=0, exclude=('!',), include=None):
    """
    Gives a list of directories in a given directory (full path), filtered according to criteria

    :param parent: parent directory to search
    :param levels: goes down this many levels
    :param exclude: exclude directories containing this string
    :param include: exclude directories that don't contain this string
    :return:
    """
    lis = glob.glob('%s/*/' % parent)

    for level in range(levels):
        newlis = []
        for e in lis:
            newlis.extend(glob.glob('%s/*/' % e))
        lis = newlis
        lis = [x[:-1] for x in lis]

    # Filter
    if exclude is not None:
        for i in exclude:
            lis = [x for x in lis if i not in x]

    if include is not None:
        for i in include:
            lis = [x for x in lis if i in x]

    return lis
