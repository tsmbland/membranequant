import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import splprep, splev, CubicSpline, interp1d
from scipy.special import erf
from skimage import io
import cv2
import glob
import copy
import os

"""
os.walk for direcslist

"""


########## IMAGE HANDLING ###########


def load_image(filename):
    """
    Given the filename of a TIFF, creates numpy array with pixel intensities

    :param filename:
    :return:
    """

    # img = np.array(Image.open(filename), dtype=np.float64)
    # img[img == 0] = np.nan
    return io.imread(filename).astype(float)


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


########### IMAGE OPERATIONS ###########


def straighten(img, roi, thickness, interp='cubic'):
    """
    Creates straightened image based on coordinates

    Doesn't work properly for non-periodic rois

    :param img:
    :param roi: Coordinates. Should be 1 pixel length apart in a loop
    :param thickness:
    :return:

    """

    # Calculate gradients
    xcoors = roi[:, 0]
    ycoors = roi[:, 1]
    ydiffs = np.diff(ycoors, prepend=ycoors[-1])
    xdiffs = np.diff(xcoors, prepend=xcoors[-1])
    grad = ydiffs / xdiffs
    tangent_grad = -1 / grad

    # Get interpolation coordinates
    offsets = np.linspace(thickness / 2, -thickness / 2, thickness)
    xchange = ((offsets ** 2)[np.newaxis, :] / (1 + tangent_grad ** 2)[:, np.newaxis]) ** 0.5
    ychange = xchange / abs(grad)[:, np.newaxis]
    gridcoors_x = xcoors[:, np.newaxis] + np.sign(ydiffs)[:, np.newaxis] * np.sign(offsets)[np.newaxis, :] * xchange
    gridcoors_y = ycoors[:, np.newaxis] - np.sign(xdiffs)[:, np.newaxis] * np.sign(offsets)[np.newaxis, :] * ychange

    # Interpolate
    if interp == 'linear':
        straight = map_coordinates(img.T, [gridcoors_x, gridcoors_y], order=1, mode='nearest')
    elif interp == 'cubic':
        straight = map_coordinates(img.T, [gridcoors_x, gridcoors_y], order=3, mode='nearest')
    return straight.astype(np.float64).T


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


def rotated_embryo(img, roi, l=None, h=None):
    """
    Takes an image and rotates according to coordinates so that anterior is on left, posterior on right

    :param img:
    :param roi:
    :return:

    To do:
    - ensure correct orientation (posterior on right)
    - ability to specify height and width of image
    - interpolation type

    """

    # PCA on ROI coordinates
    [latent, coeff] = np.linalg.eig(np.cov(roi.T))

    # Transform ROI
    roi_transformed = np.dot(coeff.T, roi.T)

    # Force long axis orientation
    x_range = (min(roi_transformed[0, :]) - max(roi_transformed[0, :]))
    y_range = (min(roi_transformed[1, :]) - max(roi_transformed[1, :]))
    if x_range > y_range:
        img = img.T
        roi_transformed = np.flipud(roi_transformed)
        coeff = coeff.T

    # Coordinate grid
    centre_x = (min(roi_transformed[0, :]) + max(roi_transformed[0, :])) / 2
    xvals = np.arange(int(centre_x - l / 2), int(centre_x + l / 2))
    centre_y = (min(roi_transformed[1, :]) + max(roi_transformed[1, :])) // 2
    yvals = np.arange(int(centre_y - h / 2), int(centre_y + h / 2))
    xvals_grid = np.tile(xvals, [len(yvals), 1])
    yvals_grid = np.tile(yvals, [len(xvals), 1]).T

    # Transform coordinate grid back
    [xvals_back, yvals_back] = np.dot(coeff, np.array([xvals_grid.flatten(), yvals_grid.flatten()]))
    xvals_back_grid = np.reshape(xvals_back, [len(yvals), len(xvals)])
    yvals_back_grid = np.reshape(yvals_back, [len(yvals), len(xvals)])

    # Map coordinates using linear interpolation
    zvals = map_coordinates(img.T, [xvals_back_grid, yvals_back_grid], order=1)

    # Force posterior on right
    if roi_transformed[0, 0] < roi_transformed[0, roi_transformed.shape[1] // 2]:
        zvals = np.fliplr(zvals)

    return zvals


def bg_subtraction(img, roi, band=(25, 75)):
    a = polycrop(img, roi, band[1]) - polycrop(img, roi, band[0])
    a = [np.nanmean(a[np.nonzero(a)])]
    return img - a


########### ROI OPERATIONS ###########


def offset_coordinates(roi, offsets):
    """
    Reads in coordinates, adjusts according to offsets

    :param roi: two column array containing x and y coordinates. e.g. coors = np.loadtxt(filename)
    :param offsets: array the same length as coors. Direction?
    :return: array in same format as coors containing new coordinates

    To save this in a fiji readable format run:
    np.savetxt(filename, newcoors, fmt='%.4f', delimiter='\t')

    """

    # Calculate gradients
    xcoors = roi[:, 0]
    ycoors = roi[:, 1]
    ydiffs = np.diff(ycoors, prepend=ycoors[-1])
    xdiffs = np.diff(xcoors, prepend=xcoors[-1])
    grad = ydiffs / xdiffs
    tangent_grad = -1 / grad

    # Offset coordinates
    xchange = ((offsets ** 2) / (1 + tangent_grad ** 2)) ** 0.5
    ychange = xchange / abs(grad)
    newxs = xcoors + np.sign(ydiffs) * np.sign(offsets) * xchange
    newys = ycoors - np.sign(xdiffs) * np.sign(offsets) * ychange
    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)
    return newcoors


def interp_roi(roi, periodic=True):
    """
    Interpolates coordinates to one pixel distances (or as close as possible to one pixel)
    Linear interpolation

    :param roi:
    :return:
    """

    if periodic:
        c = np.append(roi, [roi[0, :]], axis=0)
    else:
        c = roi

    # Calculate distance between points in pixel units
    distances = ((np.diff(c[:, 0]) ** 2) + (np.diff(c[:, 1]) ** 2)) ** 0.5
    distances_cumsum = np.r_[0, np.cumsum(distances)]
    total_length = sum(distances)

    # Interpolate
    fx, fy = interp1d(distances_cumsum, c[:, 0], kind='linear'), interp1d(distances_cumsum, c[:, 1], kind='linear')
    positions = np.linspace(0, total_length, int(round(total_length)))
    xcoors, ycoors = fx(positions), fy(positions)
    newpoints = np.c_[xcoors[:-1], ycoors[:-1]]
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


def spline_roi(roi, periodic=True, s=0):
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
    tck, u = splprep([x, y], s=s, per=periodic)

    # Evaluate spline
    xi, yi = splev(np.linspace(0, 1, 10000), tck)

    # Interpolate
    return interp_roi(np.vstack((xi, yi)).T, periodic=periodic)


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


########### ARRAY OPERATIONS ###########


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

    Todo: no loops

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

    """
    if window == 1:
        return array
    if not periodic:
        array_padded = np.r_[array[:int(window / 2)][::-1], array, array[-int(window / 2):][::-1]]
    else:
        array_padded = np.r_[array[-int(window / 2):], array, array[:int(window / 2)]]
    cumsum = np.cumsum(array_padded)
    return (cumsum[window:] - cumsum[:-window]) / window


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
        array_padded = np.c_[array[:, :int(window / 2)][:, :-1], array, array[:, -int(window / 2):][:, :-1]]
    else:
        array_padded = np.c_[array[:, -int(window / 2):], array, array[:, :int(window / 2)]]
    cumsum = np.cumsum(array_padded, axis=1)
    return (cumsum[:, window:] - cumsum[:, :-window]) / window


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


########### REFERENCE PROFILES ###########


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


########### MISC FUNCTIONS ###########


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
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    return (4 / 3) * np.pi * r2 * r2 * r1


def calc_sa(normcoors):
    r1 = (max(normcoors[:, 0]) - min(normcoors[:, 0])) / 2
    r2 = (max(normcoors[:, 1]) - min(normcoors[:, 1])) / 2
    e = (1 - (r2 ** 2) / (r1 ** 2)) ** 0.5
    return 2 * np.pi * r2 * r2 * (1 + (r1 / (r2 * e)) * np.arcsin(e))


def make_mask(shape, roi):
    return cv2.fillPoly(np.zeros(shape) * np.nan, [np.int32(roi)], 1)


def readnd(path):
    """

    :param direc: directory to embryo folder containing nd file
    :return: dictionary containing data from nd file
    """

    nd = {}
    f = open(path, 'r').readlines()
    for line in f[:-1]:
        nd[line.split(', ')[0].replace('"', '')] = line.split(', ')[1].strip().replace('"', '')
    return nd


def organise_by_nd(direc):
    """
    Organises images in a folder using the nd files

    :param direc:
    :return:
    """
    a = glob.glob('%s/*.nd' % direc)
    for b in a:
        name = os.path.basename(os.path.normpath(b))
        if name[0] == '_':
            folder = name[1:-3]
        else:
            folder = name[:-3]
        os.makedirs('%s/%s' % (direc, folder))
        os.rename(b, '%s/%s/%s' % (direc, folder, name))
        for file in glob.glob('%s_*' % b[:-3]):
            os.rename(file, '%s/%s/%s' % (direc, folder, os.path.basename(os.path.normpath(file))))


def direcslist(dest, levels=0, exclude=('!',), exclusive=None):
    """
    Gives a list of directories in a given directory (full path)

    :param dest:
    :param levels:
    :param exclude: exclude directories containing this string
    :param exclusive: exclude directories that don't contain this string
    :return:
    """

    def _direcslist(_dest, _levels=0, _exclude=('!',), _exclusive=None):

        lis = sorted(glob.glob('%s/*/' % _dest))

        for level in range(_levels):
            newlis = []
            for e in lis:
                newlis.extend(sorted(glob.glob('%s/*/' % e)))
            lis = newlis
            lis = [x[:-1] for x in lis]

        # Excluded directories
        lis_new = copy.deepcopy(lis)
        for i in _exclude:
            for x in lis:
                if i in x:
                    lis_new.remove(x)

        # Exclusive directories
        if _exclusive is not None:
            for x in lis:
                if sum([i in x for i in _exclusive]) == 0:
                    lis_new.remove(x)
        return sorted(lis_new)

    if type(dest) is list:
        out = []
        for d in dest:
            out.extend(_direcslist(d, levels, exclude, exclusive))
        return out
    else:
        return _direcslist(dest, levels, exclude, exclusive)


def import_all(direcs, key):
    data = []
    for d in direcs:
        data.extend([np.loadtxt('%s/%s' % (d, key))])
    return np.array(data)