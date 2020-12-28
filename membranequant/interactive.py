import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import glob
from .funcs import spline_roi, load_image, interp_1d_array


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
