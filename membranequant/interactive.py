import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def view_stack(frames, start_frame=0, end_frame=None):
    """
    Interactive stack viewer

    """

    # Detect if single frame or stack
    if type(frames) is list:
        stack = True
        frames_ = frames
    elif len(frames.shape) == 3:
        stack = True
        frames_ = list(frames)
    else:
        stack = False
        frames_ = [frames, ]

    # Set up figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Specify ylim
    vmax = max([np.max(i) for i in frames_])
    vmin = min([np.min(i) for i in frames_])

    # Stack
    if stack:
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        if end_frame is None:
            end_frame = len(frames_)
        sframe = Slider(axframe, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')

        def update(i):
            ax.clear()
            ax.imshow(frames_[int(i)], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        sframe.on_changed(update)
        update(start_frame)

    # Single frame
    else:
        ax.imshow(frames_[0], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.canvas.set_window_title('')
    plt.show(block=True)


def plot_segmentation(frames, rois):
    """
    Plot segmentation results

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Detect if single frame or stack
    if type(frames) is list:
        stack = True
        frames_ = frames
    elif len(frames.shape) == 3:
        stack = True
        frames_ = list(frames)
    else:
        stack = False
        frames_ = [frames, ]

    # Specify ylim
    ylim_top = max([np.max(i) for i in frames_])
    ylim_bottom = min([np.min(i) for i in frames_])

    # Single frame
    if not stack:
        ax.imshow(frames_[0], cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
        ax.plot(rois[:, 0], rois[:, 1], c='lime')
        ax.set_xticks([])
        ax.set_yticks([])

    # Stack
    else:

        # Add frame slider
        plt.subplots_adjust(left=0.25, bottom=0.25)
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', 0, len(frames_), valinit=0, valfmt='%d')

        def update(i):
            ax.clear()
            ax.imshow(frames_[int(i)], cmap='gray', vmin=ylim_bottom, vmax=ylim_top)
            ax.plot(rois[int(i)][:, 0], rois[int(i)][:, 1], c='lime')
            ax.set_xticks([])
            ax.set_yticks([])

        sframe.on_changed(update)
        update(0)

    fig.canvas.set_window_title('Segmentation')
    plt.show(block=True)


def plot_quantification(mems):
    """
    Plot quantification results

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Detect if single frame or stack
    if type(mems) is list:
        stack = True
        mems_ = mems
    elif len(mems.shape) == 2:
        stack = True
        mems_ = list(mems)
    else:
        stack = False
        mems_ = [mems, ]

    # Single frame
    if not stack:
        ax.plot(mems_[0])
        ax.set_xlabel('Position')
        ax.set_ylabel('Membrane concentration')
        ax.set_ylim(bottom=0)

    # Stack
    else:

        # Specify ylim
        ylim_top = max([np.max(m) for m in mems_])
        ylim_bottom = min([np.min(m) for m in mems_])

        # Add frame silder
        plt.subplots_adjust(left=0.25, bottom=0.25)
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', 0, len(mems_), valinit=0, valfmt='%d')

        def update(i):
            ax.clear()
            ax.plot(mems_[int(i)])
            ax.axhline(0, c='k', linestyle='--')
            ax.set_xlabel('Position')
            ax.set_ylabel('Membrane concentration')
            ax.set_ylim(ylim_bottom, ylim_top)

        sframe.on_changed(update)
        update(0)

    fig.canvas.set_window_title('Membrane Quantification')
    plt.show(block=True)


class FitPlotter:
    def __init__(self, target, fit):

        # Detect if single frame or stack
        if type(target) is list:
            self.stack = True
            target_ = target
            fit_ = fit
        elif len(target.shape) == 3:
            self.stack = True
            target_ = list(target)
            fit_ = list(fit)
        else:
            self.stack = False
            target_ = [target, ]
            fit_ = [fit, ]

        # Internal variables
        self.target = target_
        self.fit = fit_
        self.pos = 10

        # Set up figure
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(3, 3)
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.fig.add_subplot(gs[1:, :])

        # Specify ylim
        straight_max = max([np.max(i) for i in self.target])
        straight_min = min([np.min(i) for i in self.target])
        fit_max = max([np.max(i) for i in self.fit])
        fit_min = min([np.min(i) for i in self.fit])
        self.ylim_top = max([straight_max, fit_max])
        self.ylim_bottom = min([straight_min, fit_min])

        # Frame slider
        if self.stack:
            plt.subplots_adjust(bottom=0.25, left=0.25)
            axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
            slider_frame = Slider(axframe, 'Frame', 0, len(self.target), valinit=0, valfmt='%d')
            slider_frame.on_changed(lambda f: self.update_frame(int(f)))

        # Initial plot
        self.update_frame(0)

        # Show
        self.fig.canvas.set_window_title('Local fits')
        plt.show(block=True)

    def update_pos(self, p):
        self.pos = int(p)
        self.ax1_update()
        self.ax2_update()

    def update_frame(self, i):
        self._target = self.target[i]
        self._fit = self.fit[i]

        # Position slider
        self.slider_pos = Slider(self.ax1, '', 0, len(self._target[0, :]), valinit=self.pos, valfmt='%d',
                                 facecolor='none', edgecolor='none')
        self.slider_pos.on_changed(self.update_pos)

        self.ax1_update()
        self.ax2_update()

    def ax1_update(self):
        self.ax1.clear()
        self.ax1.imshow(self._target, cmap='gray', vmin=self.ylim_bottom, vmax=1.1 * self.ylim_top)
        self.ax1.axvline(self.pos, c='r')
        self.ax1.set_yticks([])
        self.ax1.set_xlabel('Position')
        self.ax1.xaxis.set_label_position('top')

    def ax2_update(self):
        self.ax2.clear()
        self.ax2.plot(self._target[:, self.pos], label='Actual')
        self.ax2.plot(self._fit[:, self.pos], label='Fit')
        self.ax2.set_xticks([])
        self.ax2.set_ylabel('Intensity')
        self.ax2.legend(frameon=False, loc='upper left', fontsize='small')
        self.ax2.set_ylim(bottom=self.ylim_bottom, top=self.ylim_top)


def plot_fits(target, fit_total):
    FitPlotter(target, fit_total)
