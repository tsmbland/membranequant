import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from .funcs import spline_roi


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

    def __init__(self, img, spline, start_frame=0, end_frame=None, periodic=True, show_fit=False):

        if type(img) == list:
            self.images = img
        else:
            self.images = [img, ]
        self.img = self.images[0]
        self.spline = spline
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.periodic = periodic
        self.show_fit = show_fit

        # Internal
        self._current_frame = self.start_frame
        self._current_image = 0
        self._point0 = None
        self._points = None
        self._line = None
        self._fitted = False

        # Outputs
        self.xpoints = []
        self.ypoints = []
        self.roi = None

    def run(self):
        # Set up figure
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        # Stack
        if len(self.img.shape) == 3:
            plt.subplots_adjust(left=0.25, bottom=0.25)
            self.axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
            if self.end_frame is None:
                self.end_frame = len(self.img[:, 0, 0]) - 1
            self.sframe = Slider(self.axframe, 'Frame', self.start_frame, self.end_frame, valinit=self.start_frame,
                                 valfmt='%d')
            self.sframe.on_changed(self.draw_frame)
        self.draw_frame(self.start_frame)

        # Show figure
        self.fig.canvas.set_window_title('Specify ROI')
        self.fig.canvas.mpl_connect('close_event', lambda event: self.fig.canvas.stop_event_loop())
        self.fig.canvas.start_event_loop(timeout=-1)

    def draw_frame(self, i):
        self._current_frame = i

        # Calculate intensity ranges
        vmin, vmax = [1, 1.1] * np.percentile(self.img.flatten(),
                                              [0.1, 99.9])  # < move this, slow to calculate every time

        # Plot image
        if len(self.img.shape) == 3:
            self.ax.imshow(self.img[int(i), :, :], cmap='gray', vmin=vmin, vmax=vmax)
        else:
            self.ax.imshow(self.img, cmap='gray', vmin=vmin, vmax=vmax)

        # Finalise figure
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
            if len(self.xpoints) != 0:
                roi = np.vstack((self.xpoints, self.ypoints)).T

                # Spline
                if self.spline:
                    if not self._fitted:
                        self.roi = spline_roi(roi, periodic=self.periodic)
                        self._fitted = True

                        # Display line
                        if self.show_fit:
                            self._line = self.ax.plot(self.roi[:, 0], self.roi[:, 1], c='b')
                            self.fig.canvas.draw()
                        else:
                            plt.close(self.fig)  # comment this out to see spline fit
                    else:
                        plt.close(self.fig)
                else:
                    self.roi = roi
                    plt.close(self.fig)
            else:
                self.roi = []
                plt.close(self.fig)

        if event.key == ',':
            self._current_image = max(0, self._current_image - 1)
            self.img = self.images[self._current_image]
            self.draw_frame(self._current_frame)

        if event.key == '.':
            self._current_image = min(len(self.images) - 1, self._current_image + 1)
            self.img = self.images[self._current_image]
            self.draw_frame(self._current_frame)

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


def def_roi(stack, spline=True, start_frame=0, end_frame=None, periodic=True, show_fit=True):
    r = ROI(stack, spline=spline, start_frame=start_frame, end_frame=end_frame, periodic=periodic, show_fit=show_fit)
    r.run()
    return r.roi
