import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from .funcs import straighten, rolling_ave_2d, interp_1d_array, interp_2d_array, rotate_roi, save_img, \
    offset_coordinates, spline_roi
from .interactive import view_stack, view_stack_jupyter, plot_fits, plot_fits_jupyter, plot_segmentation, \
    plot_segmentation_jupyter, plot_quantification, plot_quantification_jupyter
from scipy.interpolate import interp1d
from tqdm import tqdm
import pickle
import time

"""
To do:
Doesn't throw error if bgcurve are wrong size - could be a problem
Permit wider bgcurves for wiggle room
Switch from linear bgcurve interpolation to cubic spline
Option to have separate curves (and sigma) for each image
Some problem when training cytbg of width 50 but not 60 or 40???
Have a free alignment parameter (i.e. on or off)

"""


class ImageQuant:
    """
    Quantification works by taking cross sections across the membrane, and fitting the resulting profile as the sum of
    a cytoplasmic signal component and a membrane signal component

    Input data:
    img                image

    Background curves:
    cytbg              cytoplasmic background curve, should be as thick as thickness parameter
    membg              membrane background curve, as above
    sigma              if either of above are not specified, assume gaussian/error function with width set by sigma

    ROI:
    roi                coordinates defining cortex. Can use output from def_roi function

    Fitting parameters:
    periodic           True if coordinates form a closed loop
    thickness          thickness of cross section over which to perform quantification
    rol_ave            width of rolling average
    rotate             if True, will automatically rotate ROI so that the first/last points are at the end of the long
                       axis
    nfits              performs this many fits at regular intervals around ROI
    iterations         if >1, adjusts ROI and re-fits
    uni_cyt            globally fit uniform cytoplasm
    uni_mem            globally fit uniform membrane
    bg_subtract        if True, will estimate and subtract background signal prior to quantification

    Saving:
    save_path          destination to save results, will create if it doesn't already exist


    """

    def __init__(self, img, roi, cytbg=None, membg=None, sigma=2, periodic=True, thickness=50,
                 rol_ave=20, rotate=False, nfits=None, iterations=1, bg_subtract=False, uni_cyt=False, uni_mem=False,
                 cyt_only=False, mem_only=False, lr=0.01, descent_steps=2000, adaptive_sigma=False,
                 adaptive_membg=False, adaptive_cytbg=False, align=True, batch_norm=False):

        # Detect if single frame or stack
        if type(img) is list:
            self.stack = True
            self.img = img
        elif len(img.shape) == 3:
            self.stack = True
            self.img = list(img)
        else:
            self.stack = False
            self.img = [img, ]
        self.n = len(self.img)

        # ROI
        if not self.stack:
            self.roi = [roi, ]
        elif type(roi) is list:
            if len(roi) > 1:
                self.roi = roi
            else:
                self.roi = roi * self.n
        else:
            self.roi = [roi] * self.n

        self.periodic = periodic

        # Background subtraction
        self.bg_subtract = bg_subtract

        # Normalisation
        self.batch_norm = batch_norm

        # Fitting mode
        self.uni_cyt = uni_cyt
        self.uni_mem = uni_mem
        self.cyt_only = cyt_only
        self.mem_only = mem_only

        # Fitting parameters
        self.align = align
        self.iterations = iterations
        self.thickness = thickness
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.sigma = sigma
        self.nfits = nfits
        self.lr = lr
        self.descent_steps = descent_steps

        # Background curves
        self.cytbg = cytbg
        self.membg = membg

        # Learning
        self.adaptive_sigma = adaptive_sigma
        self.adaptive_membg = adaptive_membg
        self.adaptive_cytbg = adaptive_cytbg

        # Internal variables

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

    """
    Run

    """

    def run(self):
        t = time.time()

        # Fitting
        for i in range(self.iterations):
            if i > 0:
                self.adjust_roi()
            self.fit()

        print('Time elapsed: %.2f seconds ' % (time.time() - t))

    def preprocess(self, frame, roi):

        # Straighten
        straight = straighten(frame, roi, thickness=self.thickness)

        # Smoothen
        straight_filtered = rolling_ave_2d(straight, window=self.rol_ave, periodic=self.periodic)

        # Background subtract
        if self.bg_subtract:
            straight_filtered -= np.mean(straight_filtered[:5, :])

        # Interpolate
        straight_filtered_itp = interp_2d_array(straight_filtered, self.nfits, ax=0, method='cubic')

        # Normalise
        if not self.batch_norm:
            norm = np.percentile(straight_filtered_itp, 99)
            target = straight_filtered_itp / norm
        else:
            norm = 1
            target = straight_filtered_itp

        return target, norm

    def init_tensors(self):
        nimages = self.target.shape[0]
        nfits = self.target.shape[2]
        self.vars = []

        # Offsets
        self.offsets_t = tf.Variable(np.zeros([nimages, nfits]))
        if self.align:
            self.vars.append(self.offsets_t)

        # Cytoplasmic concentrations
        if self.uni_cyt:
            self.cyts_t = tf.Variable(0 * np.mean(self.target[:, -5:, :], axis=(1, 2)))
        else:
            self.cyts_t = tf.Variable(0 * np.mean(self.target[:, -5:, :], axis=1))
        if self.mem_only:
            self.cyts_t = self.cyts_t * 0
        else:
            self.vars.append(self.cyts_t)

        # Membrane concentrations
        if self.uni_mem:
            self.mems_t = tf.Variable(0 * np.max(self.target, axis=(1, 2)))
        else:
            self.mems_t = tf.Variable(0 * np.max(self.target, axis=1))
        if self.cyt_only:
            self.mems_t = self.mems_t * 0
        else:
            self.vars.append(self.mems_t)

        # Sigma
        if self.sigma is not None:
            self.sigma_t = tf.Variable(self.sigma, dtype=tf.float64)
        if self.adaptive_sigma:
            self.vars.append(self.sigma_t)

        # Cytbg
        if self.cytbg is not None:
            self.cytbg_t = tf.Variable(self.cytbg)
        if self.adaptive_cytbg:
            self.vars.append(self.cytbg_t)

        # Membg
        if self.membg is not None:
            self.membg_t = tf.Variable(self.membg)
        if self.adaptive_membg:
            self.vars.append(self.membg_t)

    def sim_images(self, include_c=True, include_m=True):
        nimages = self.mems_t.shape[0]
        nfits = self.nfits

        # Need to align peak to centre - how?

        # Positions to evaluate mem and cyt curves
        positions_ = np.arange(self.thickness, dtype=np.float64)[tf.newaxis, tf.newaxis, :]
        offsets_ = self.offsets_t[:, :, tf.newaxis]
        positions = tf.reshape(tf.math.add(positions_, offsets_), [-1])

        # Mem curve
        if self.membg is not None:
            # membg_norm = self.membg_t / tf.reduce_max(self.membg_t)
            mem_curve = tfp.math.interp_regular_1d_grid(y_ref=self.membg_t, x_ref_min=0, x_ref_max=1,
                                                        x=positions / self.thickness)
        else:
            mem_curve = tf.math.exp(-((positions - self.thickness / 2) ** 2) / (2 * self.sigma_t ** 2))
        mem_curve = tf.reshape(mem_curve, [nimages, nfits, self.thickness])

        # Cyt curve
        if self.cytbg is not None:
            # cytbg_norm = self.cytbg_t / tf.reduce_max(self.cytbg_t)
            cyt_curve = tfp.math.interp_regular_1d_grid(y_ref=self.cytbg_t, x_ref_min=0, x_ref_max=1,
                                                        x=positions / self.thickness)
        else:
            cyt_curve = (1 + tf.math.erf((positions - self.thickness / 2) / self.sigma_t)) / 2
        cyt_curve = tf.reshape(cyt_curve, [nimages, nfits, self.thickness])

        # Calculate output
        if self.uni_mem:
            mem_total = mem_curve * tf.expand_dims(tf.expand_dims(self.mems_t, axis=-1), axis=-1)
        else:
            mem_total = mem_curve * tf.expand_dims(self.mems_t, axis=-1)
        if self.uni_cyt:
            cyt_total = cyt_curve * tf.expand_dims(tf.expand_dims(self.cyts_t, axis=-1), axis=-1)
        else:
            cyt_total = cyt_curve * tf.expand_dims(self.cyts_t, axis=-1)

        # Sum outputs
        if include_c and include_m:
            return tf.transpose(tf.math.add(mem_total, cyt_total), [0, 2, 1])
        elif include_c:
            return tf.transpose(cyt_total, [0, 2, 1])
        elif include_m:
            return tf.transpose(mem_total, [0, 2, 1])

    def losses_full(self):
        return tf.math.reduce_mean((self.sim_images() - self.target) ** 2, axis=[1, 2])

    def fit(self):

        # Specify number of fits
        if self.nfits is None:
            self.nfits = len(self.roi[:, 0])

        # Preprocess
        target, norms = zip(*[self.preprocess(frame, roi) for frame, roi in zip(self.img, self.roi)])
        self.target = np.array(target)
        self.norms = np.array(norms)

        # Batch normalise
        if self.batch_norm:
            norm = np.percentile(self.target, 99)
            self.target /= norm
            self.norms = norm * np.ones(self.target.shape[0])

        # Init tensors
        self.init_tensors()

        # Run optimisation
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.losses = np.zeros([len(self.img), self.descent_steps])
        for i in tqdm(range(self.descent_steps)):
            with tf.GradientTape() as tape:
                losses_full = self.losses_full()
                self.losses[:, i] = losses_full
                loss = tf.reduce_mean(losses_full)
                grads = tape.gradient(loss, self.vars)
                opt.apply_gradients(list(zip(grads, self.vars)))

        # Save and rescale sim images (rescaled)
        self.sim_both = self.sim_images().numpy() * self.norms[:, np.newaxis, np.newaxis]
        self.sim_cyt = self.sim_images(include_m=False).numpy() * self.norms[:, np.newaxis, np.newaxis]
        self.sim_mem = self.sim_images(include_c=False).numpy() * self.norms[:, np.newaxis, np.newaxis]
        self.target = self.target * self.norms[:, np.newaxis, np.newaxis]

        # Save and rescale results
        if self.uni_mem:
            self.mems = np.tile((self.mems_t.numpy() * self.norms)[:, np.newaxis], [1, self.nfits])
        else:
            self.mems = self.mems_t.numpy() * self.norms[:, np.newaxis]
        if self.uni_cyt:
            self.cyts = np.tile((self.cyts_t.numpy() * self.norms)[:, np.newaxis], [1, self.nfits])
        else:
            self.cyts = self.cyts_t.numpy() * self.norms[:, np.newaxis]
        self.offsets = self.offsets_t.numpy()

        # Interpolated results
        self.offsets_full = [interp_1d_array(offsets, len(roi[:, 0]), method='linear') for offsets, roi in
                             zip(self.offsets, self.roi)]
        self.cyts_full = [interp_1d_array(cyts, len(roi[:, 0]), method='linear') for cyts, roi in
                          zip(self.cyts, self.roi)]
        self.mems_full = [interp_1d_array(mems, len(roi[:, 0]), method='linear') for mems, roi in
                          zip(self.mems, self.roi)]

        # Interpolated sim images
        self.sim_both_full = [interp1d(np.arange(self.nfits), sim_both, axis=-1)(
            np.linspace(0, self.nfits - 1, len(roi[:, 0]))) for roi, sim_both in
            zip(self.roi, self.sim_both)]
        self.sim_cyt_full = [
            interp1d(np.arange(self.nfits), sim_cyt, axis=-1)(np.linspace(0, self.nfits - 1, len(roi[:, 0]))) for
            roi, sim_cyt in zip(self.roi, self.sim_cyt)]
        self.sim_mem_full = [interp1d(np.arange(self.nfits), sim_mem, axis=-1)(
            np.linspace(0, self.nfits - 1, len(roi[:, 0]))) for roi, sim_mem in
            zip(self.roi, self.sim_mem)]
        self.target_full = [interp1d(np.arange(self.nfits), target, axis=-1)(
            np.linspace(0, self.nfits - 1, len(roi[:, 0]))) for roi, target in zip(self.roi, self.target)]
        self.resids_full = [i - j for i, j in zip(self.target_full, self.sim_both_full)]

        # Save adaptable params
        if self.sigma is not None:
            self.sigma = self.sigma_t.numpy()
        if self.cytbg is not None:
            self.cytbg = self.cytbg_t.numpy()
        if self.membg is not None:
            self.membg = self.membg_t.numpy()

    """
    Misc

    """

    def adjust_roi(self):
        """
        Can do after a preliminary fit to refine coordinates
        Must refit after doing this

        """

        # Offset coordinates
        self.roi = [offset_coordinates(roi, offsets_full) for roi, offsets_full in zip(self.roi, self.offsets_full)]

        # Filter
        self.roi = [spline_roi(roi=roi, periodic=self.periodic, s=100) for roi in self.roi]

        # Rotate
        if self.periodic:
            if self.rotate:
                self.roi = [rotate_roi(roi) for roi in self.roi]

    def save(self, save_path, i=None):
        """
        Save all results to save_path

        """

        if not self.stack:
            i = 0

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        np.savetxt(save_path + '/offsets.txt', self.offsets[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/cyts.txt', self.cyts[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/mems.txt', self.mems[i], fmt='%.4f', delimiter='\t')
        np.savetxt(save_path + '/roi.txt', self.roi[i], fmt='%.4f', delimiter='\t')
        save_img(self.img[i], save_path + '/img.tif')
        save_img(self.target_full[i], save_path + '/target.tif')
        save_img(self.sim_both_full[i], save_path + '/fit.tif')
        save_img(self.sim_cyt_full[i], save_path + '/fit_cyt.tif')
        save_img(self.sim_mem_full[i], save_path + '/fit_mem.tif')
        save_img(self.resids_full[i], save_path + '/resids.tif')
        save_img(np.clip(self.resids_full[i], 0, None), save_path + '/resids_pos.tif')
        save_img(abs(np.clip(self.resids_full[i], None, 0)), save_path + '/resids_neg.tif')

    def compile_res(self):
        # Create empty dataframe
        df = pd.DataFrame({'Frame': [],
                           'Position': [],
                           'Membrane concentration': [],
                           'Cytoplasmic concentration': []})

        # Fill with data
        for i in range(len(self.img)):
            df = df.append(pd.DataFrame({'Frame': i,
                                         'Position': range(self.nfits),
                                         'Membrane signal': self.mems[i],
                                         'Cytoplasmic signal': self.cyts[i]}))

        df = df.reindex(columns=['Frame', 'Position', 'Membrane signal', 'Cytoplasmic signal'])
        df = df.astype({'Frame': int, 'Position': int})
        return df

    def pickle(self, filename):
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    """
    Interactive
    
    """

    def view_frames(self, jupyter=False):
        if not jupyter:
            if self.stack:
                view_stack(self.img)
            else:
                view_stack(self.img[0])
        else:
            if self.stack:
                view_stack_jupyter(self.img)
            else:
                view_stack_jupyter(self.img[0])

    def plot_quantification(self, jupyter=False):
        if not jupyter:
            if self.stack:
                plot_quantification(self.mems_full)
            else:
                plot_quantification(self.mems_full[0])
        else:
            if self.stack:
                plot_quantification_jupyter(self.mems_full)
            else:
                plot_quantification_jupyter(self.mems_full[0])

    def plot_fits(self, jupyter=False):
        if not jupyter:
            if self.stack:
                plot_fits(self.target_full, self.sim_both_full)
            else:
                plot_fits(self.target_full[0], self.sim_both_full[0])
        else:
            if self.stack:
                plot_fits_jupyter(self.target_full, self.sim_both_full)
            else:
                plot_fits_jupyter(self.target_full[0], self.sim_both_full[0])

    def plot_segmentation(self, jupyter=False):
        if not jupyter:
            if self.stack:
                plot_segmentation(self.img, self.roi)
            else:
                plot_segmentation(self.img[0], self.roi[0])
        else:
            if self.stack:
                plot_segmentation_jupyter(self.img, self.roi)
            else:
                plot_segmentation_jupyter(self.img[0], self.roi[0])

    # def def_roi(self):
    #     r = ROI(self.img, spline=True)
    #     self.roi = r.roi
