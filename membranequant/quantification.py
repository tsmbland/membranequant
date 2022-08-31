import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from .funcs import straighten, rolling_ave_2d, interp_1d_array, interp_2d_array, rotate_roi, save_img
from .roi import offset_coordinates, interp_roi
from .interactive import view_stack, view_stack_jupyter, plot_fits, plot_fits_jupyter, plot_segmentation, \
    plot_segmentation_jupyter, plot_quantification, plot_quantification_jupyter
from scipy.interpolate import interp1d
from tqdm import tqdm
import pickle
import time
from scipy.special import erf
from .tgf_interpolate import interpolate
import matplotlib.pyplot as plt

"""
To do:
Doesn't throw error if bgcurve are wrong size - could be a problem
Permit wider bgcurves for wiggle room
- Make sure bgcurves are centred no matter how big
Write def_roi function

Normalise bgcurves?

"""


class ImageQuant:
    """
    Quantification works by taking cross sections across the membrane, and fitting the resulting profile as the sum of
    a cytoplasmic signal component and a membrane signal component

    Input data:
    img                image
    roi                initial coordinates defining cortex, which can be quite rough. Can use output from def_roi
                       function

    Background curves:
    cytbg              cytoplasmic background curve, should be as thick as thickness parameter
    membg              membrane background curve, as above
    sigma              if either of above are not specified, assume gaussian/error function with width set by sigma

    ROI:
    roi_knots          number of knots in cubic-spline fit ROI
    freedom            amount by which the roi can move with each iteration

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
    position_weights   can assign a weight to each position in the loss function. A way of restricting training to a
                       certain part of the cell
    batch_norm         if True, images will be globally, rather than internally, normalised. Shouldn't affect
                       quantification but is recommended for model optimisation
    zerocap            if True, will restrict concentrations to positive values
    interp_type        interpolation type: 'cubic' or 'linear'
    fit_outer          if True, will fit the outer portion of each profile to a nonzero value

    Gradient descent:
    lr                 learning rate
    descent_steps      number of gradient descent steps
    loss               loss function: 'mse' or 'mae'

    Model optimisation:
    adaptive_sigma     if True, sigma will be trained by gradient descent
    adaptive_membg     if True, membg will be trained by gradient descent
    adaptive_cytbg     if True, cytbg will be trained by gradient descent

    """

    def __init__(self, img, roi, cytbg=None, membg=None, sigma=2, periodic=True, thickness=50,
                 rol_ave=10, rotate=False, nfits=100, iterations=2, uni_cyt=False, uni_mem=False,
                 cyt_only=False, mem_only=False, lr=0.01, descent_steps=500, adaptive_sigma=False,
                 adaptive_membg=False, adaptive_cytbg=False, batch_norm=False, position_weights=None,
                 freedom=10, zerocap=False, roi_knots=20, loss='mse', interp_type='cubic', fit_outer=False,
                 save_training=False, save_sims=False):

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
        self.roi_knots = roi_knots

        # Normalisation
        self.batch_norm = batch_norm

        # Fitting mode
        self.uni_cyt = uni_cyt
        self.uni_mem = uni_mem
        self.cyt_only = cyt_only
        self.mem_only = mem_only

        # Fitting parameters
        self.iterations = iterations
        self.thickness = thickness
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.sigma = sigma
        self.nfits = nfits
        self.lr = lr
        self.descent_steps = descent_steps
        self.position_weights = position_weights
        self.freedom = freedom
        self.zerocap = zerocap
        self.loss_mode = loss
        self.interp_type = interp_type
        self.fit_outer = fit_outer
        self.save_training = save_training
        self.save_sims = save_sims

        # Background curves
        self.cytbg = cytbg
        self.membg = membg

        if cytbg is None and adaptive_cytbg is True:
            self.cytbg = (1 + erf((np.arange(self.thickness) - self.thickness / 2) / self.sigma)) / 2
        if membg is None and adaptive_membg is True:
            self.membg = np.exp(-((np.arange(self.thickness) - self.thickness / 2) ** 2) / (2 * self.sigma ** 2))

        # Learning
        self.adaptive_sigma = adaptive_sigma
        self.adaptive_membg = adaptive_membg
        self.adaptive_cytbg = adaptive_cytbg

        # Results containers
        self.offsets = None
        self.cyts = None
        self.mems = None
        self.offsets_full = None
        self.cyts_full = None
        self.mems_full = None

        # Tensors
        self.cyts_t = None
        self.mems_t = None
        self.offsets_t = None

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
        straight = straighten(frame, roi, thickness=self.thickness, interp='cubic', periodic=self.periodic)

        # Smoothen
        straight_filtered = rolling_ave_2d(straight, window=self.rol_ave, periodic=self.periodic)

        # Interpolate
        straight_filtered_itp = interp_2d_array(straight_filtered, self.nfits, ax=1, method='cubic')

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
        self.vars = {}

        # Offsets
        self.offsets_t = tf.Variable(np.zeros([nimages, self.roi_knots]), name='Offsets')
        if not self.freedom == 0:
            self.vars['offsets'] = self.offsets_t

        # Cytoplasmic concentrations
        if self.uni_cyt:
            self.cyts_t = tf.Variable(0 * np.mean(self.target[:, -5:, :], axis=(1, 2)))
        else:
            self.cyts_t = tf.Variable(0 * np.mean(self.target[:, -5:, :], axis=1))
        if self.mem_only:
            self.cyts_t = self.cyts_t * 0
        else:
            self.vars['cyts'] = self.cyts_t

        # Membrane concentrations
        if self.uni_mem:
            self.mems_t = tf.Variable(0 * np.max(self.target, axis=(1, 2)))
        else:
            self.mems_t = tf.Variable(0 * np.max(self.target, axis=1))
        if self.cyt_only:
            self.mems_t = self.mems_t * 0
        else:
            self.vars['mems'] = self.mems_t

        # Outers
        if self.fit_outer:
            self.outers_t = tf.Variable(0 * np.mean(self.target[:, :5, :], axis=1))
            self.vars['outers'] = self.outers_t

        # Sigma
        if self.sigma is not None:
            self.sigma_t = tf.Variable(self.sigma, dtype=tf.float64)
        if self.adaptive_sigma:
            self.vars['sigma'] = self.sigma_t

        # Cytbg
        if self.cytbg is not None:
            self.cytbg_t = tf.Variable(self.cytbg)
        if self.adaptive_cytbg:
            self.vars['cytbg'] = self.cytbg_t

        # Membg
        if self.membg is not None:
            self.membg_t = tf.Variable(self.membg)
        if self.adaptive_membg:
            self.vars['membg'] = self.membg_t

    def sim_images(self, include_c=True, include_m=True):
        nimages = self.mems_t.shape[0]
        nfits = self.nfits

        # Constrain concentrations
        if self.zerocap:
            mems = tf.math.maximum(self.mems_t, 0)
            cyts = tf.math.maximum(self.cyts_t, 0)
        else:
            mems = self.mems_t
            cyts = self.cyts_t

        # Fit spline to offsets
        if self.periodic:
            x = np.tile(np.expand_dims(np.arange(-1., self.roi_knots + 2), 0), (nimages, 1))
            y = tf.concat((self.offsets_t[:, -1:], self.offsets_t, self.offsets_t[:, :2]), axis=1)
            knots = tf.stack((x, y))
            positions = tf.expand_dims(tf.cast(tf.linspace(start=0.0, stop=self.roi_knots,
                                                           num=self.nfits + 1)[:-1], dtype=tf.float64), axis=-1)
        else:
            x = np.tile(np.expand_dims(np.arange(-1., self.roi_knots + 1), 0), (nimages, 1))
            y = tf.concat((self.offsets_t[:, :1], self.offsets_t, self.offsets_t[:, -1:]), axis=1)
            knots = tf.stack((x, y))
            positions = tf.expand_dims(tf.cast(tf.linspace(start=0.0, stop=self.roi_knots - 1.000001,
                                                           num=self.nfits), dtype=tf.float64), axis=-1)
        spline = interpolate(knots, positions, degree=3, cyclical=False)
        spline = tf.squeeze(spline, axis=1)
        offsets_spline = tf.transpose(spline[:, 1, :])

        # Constrain offsets
        offsets = self.freedom * tf.math.tanh(offsets_spline)

        # Positions to evaluate mem and cyt curves
        positions_ = np.arange(self.thickness, dtype=np.float64)[tf.newaxis, tf.newaxis, :]
        offsets_ = offsets[:, :, tf.newaxis]
        positions = tf.reshape(tf.math.add(positions_, offsets_), [-1])

        # Cap positions off edge
        positions = tf.minimum(positions, self.thickness - 1.000001)
        positions = tf.maximum(positions, 0)

        # Mask
        mask = 1 - (tf.cast(tf.math.less(positions, 0), tf.float64) + tf.cast(
            tf.math.greater(positions, self.thickness), tf.float64))
        mask_ = tf.reshape(mask, [nimages, nfits, self.thickness])

        # Mem curve
        if self.membg is not None:
            # membg_norm = self.membg_t / tf.reduce_max(self.membg_t)
            if self.interp_type == 'cubic':
                x = np.arange(-1., self.thickness + 1)
                y = tf.concat(([self.membg_t[0]], self.membg_t, [self.membg_t[-1]]), axis=0)
                knots = tf.stack((x, y))
                spline = interpolate(knots, tf.expand_dims(positions, -1), degree=3, cyclical=False)
                mem_curve = spline[:, 0, 1]
            elif self.interp_type == 'linear':
                mem_curve = tfp.math.interp_regular_1d_grid(y_ref=self.membg_t, x_ref_min=0, x_ref_max=1,
                                                            x=positions / self.thickness)

        else:
            mem_curve = tf.math.exp(-((positions - self.thickness / 2) ** 2) / (2 * self.sigma_t ** 2))

        mem_curve = tf.reshape(mem_curve, [nimages, nfits, self.thickness])

        # Cyt curve
        if self.cytbg is not None:
            # cytbg_norm = self.cytbg_t / tf.reduce_max(self.cytbg_t)
            if self.interp_type == 'cubic':
                x = np.arange(-1., self.thickness + 1)
                y = tf.concat(([self.cytbg_t[0]], self.cytbg_t, [self.cytbg_t[-1]]), axis=0)
                knots = tf.stack((x, y))
                spline = interpolate(knots, tf.expand_dims(positions, -1), degree=3, cyclical=False)
                cyt_curve = spline[:, 0, 1]
            elif self.interp_type == 'linear':
                cyt_curve = tfp.math.interp_regular_1d_grid(y_ref=self.cytbg_t, x_ref_min=0, x_ref_max=1,
                                                            x=positions / self.thickness)
        else:
            cyt_curve = (1 + tf.math.erf((positions - self.thickness / 2) / self.sigma_t)) / 2
        cyt_curve = tf.reshape(cyt_curve, [nimages, nfits, self.thickness])

        # Calculate output
        if self.uni_mem:
            mem_total = mem_curve * tf.expand_dims(tf.expand_dims(mems, axis=-1), axis=-1)
        else:
            mem_total = mem_curve * tf.expand_dims(mems, axis=-1)
        if self.uni_cyt:
            cyt_total = cyt_curve * tf.expand_dims(tf.expand_dims(cyts, axis=-1), axis=-1)
        else:
            if not self.fit_outer:
                cyt_total = cyt_curve * tf.expand_dims(cyts, axis=-1)
            else:
                cyt_total = tf.expand_dims(self.outers_t, axis=-1) + cyt_curve * tf.expand_dims((cyts - self.outers_t),
                                                                                                axis=-1)

        # Sum outputs
        if include_c and include_m:
            return tf.transpose(tf.math.add(mem_total, cyt_total), [0, 2, 1]), tf.transpose(mask_, [0, 2, 1])
        elif include_c:
            return tf.transpose(cyt_total, [0, 2, 1]), tf.transpose(mask_, [0, 2, 1])
        elif include_m:
            return tf.transpose(mem_total, [0, 2, 1]), tf.transpose(mask_, [0, 2, 1])

    def losses_full(self):
        self.sim, mask = self.sim_images()

        if self.loss_mode == 'mse':
            sq_errors = (self.sim - self.target) ** 2

            # Position weights
            if self.position_weights is not None:
                sq_errors *= tf.expand_dims(tf.expand_dims(self.position_weights, axis=0), axis=0) / tf.reduce_mean(
                    self.position_weights)

            # Masked average
            mse = tf.reduce_sum(sq_errors * mask, axis=[1, 2]) / tf.reduce_sum(mask, axis=[1, 2])
            return mse

        elif self.loss_mode == 'mae':
            errors = tf.math.abs(self.sim - self.target)

            # Position weights
            if self.position_weights is not None:
                errors *= tf.expand_dims(tf.expand_dims(self.position_weights, axis=0), axis=0) / tf.reduce_mean(
                    self.position_weights)

            # Masked average
            mse = tf.reduce_sum(errors * mask, axis=[1, 2]) / tf.reduce_sum(mask, axis=[1, 2])
            return mse

        else:
            return None

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
        self.saved_vars = []
        self.saved_sims = []
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.losses = np.zeros([len(self.img), self.descent_steps])
        for i in tqdm(range(self.descent_steps)):
            with tf.GradientTape() as tape:
                losses_full = self.losses_full()
                self.losses[:, i] = losses_full
                loss = tf.reduce_mean(losses_full)
                grads = tape.gradient(loss, self.vars.values())
                opt.apply_gradients(list(zip(grads, self.vars.values())))

            # Save trained variables
            if self.save_training:
                newdict = {key: value.numpy() for key, value in self.vars.items()}
                self.saved_vars.append(newdict)

            # Save interim simulations
            if self.save_sims:
                self.saved_sims.append(self.sim.numpy() * self.norms[:, np.newaxis, np.newaxis])

        # Save and rescale sim images (rescaled)
        self.sim_both = self.sim_images()[0].numpy() * self.norms[:, np.newaxis, np.newaxis]
        self.sim_cyt = self.sim_images(include_m=False)[0].numpy() * self.norms[:, np.newaxis, np.newaxis]
        self.sim_mem = self.sim_images(include_c=False)[0].numpy() * self.norms[:, np.newaxis, np.newaxis]
        self.target = self.target * self.norms[:, np.newaxis, np.newaxis]

        # Save and rescale results
        if self.zerocap:
            mems = tf.math.maximum(self.mems_t, 0)
            cyts = tf.math.maximum(self.cyts_t, 0)
        else:
            mems = self.mems_t
            cyts = self.cyts_t
        if self.uni_mem:
            self.mems = np.tile((mems.numpy() * self.norms)[:, np.newaxis], [1, self.nfits])
        else:
            self.mems = mems.numpy() * self.norms[:, np.newaxis]
        if self.uni_cyt:
            self.cyts = np.tile((cyts.numpy() * self.norms)[:, np.newaxis], [1, self.nfits])
        else:
            self.cyts = cyts.numpy() * self.norms[:, np.newaxis]

        # Offsets
        if self.periodic:
            x = np.tile(np.expand_dims(np.arange(-1., self.roi_knots + 2), 0), (self.mems_t.shape[0], 1))
            y = tf.concat((self.offsets_t[:, -1:], self.offsets_t, self.offsets_t[:, :2]), axis=1)
            knots = tf.stack((x, y))
            positions = tf.expand_dims(tf.cast(tf.linspace(start=0.0, stop=self.roi_knots, num=self.nfits + 1)[:-1],
                                               dtype=tf.float64), axis=-1)

        else:
            x = np.tile(np.expand_dims(np.arange(-1., self.roi_knots + 1), 0), (self.mems_t.shape[0], 1))
            y = tf.concat((self.offsets_t[:, :1], self.offsets_t, self.offsets_t[:, -1:]), axis=1)
            knots = tf.stack((x, y))
            positions = tf.expand_dims(tf.cast(tf.linspace(start=0.0, stop=self.roi_knots - 1.000001,
                                                           num=self.nfits), dtype=tf.float64), axis=-1)
        spline = interpolate(knots, positions, degree=3, cyclical=False)
        spline = tf.squeeze(spline, axis=1)
        offsets_spline = tf.transpose(spline[:, 1, :])
        self.offsets = self.freedom * tf.math.tanh(offsets_spline).numpy()

        # Interpolated results
        self.offsets_full = [interp_1d_array(offsets, len(roi[:, 0]), method='cubic') for offsets, roi in
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
        self.roi = [interp_roi(offset_coordinates(roi, offsets_full), periodic=self.periodic) for roi, offsets_full in
                    zip(self.roi, self.offsets_full)]

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
                           'Membrane signal': [],
                           'Cytoplasmic signal': []})

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
                fig, ax = view_stack(self.img)
            else:
                fig, ax = view_stack(self.img[0])
        else:
            if self.stack:
                fig, ax = view_stack_jupyter(self.img)
            else:
                fig, ax = view_stack_jupyter(self.img[0])
        return fig, ax

    def plot_quantification(self, jupyter=False):
        if not jupyter:
            if self.stack:
                fig, ax = plot_quantification(self.mems_full)
            else:
                fig, ax = plot_quantification(self.mems_full[0])
        else:
            if self.stack:
                fig, ax = plot_quantification_jupyter(self.mems_full)
            else:
                fig, ax = plot_quantification_jupyter(self.mems_full[0])
        return fig, ax

    def plot_fits(self, jupyter=False):
        if not jupyter:
            if self.stack:
                fig, ax = plot_fits(self.target_full, self.sim_both_full)
            else:
                fig, ax = plot_fits(self.target_full[0], self.sim_both_full[0])
        else:
            if self.stack:
                fig, ax = plot_fits_jupyter(self.target_full, self.sim_both_full)
            else:
                fig, ax = plot_fits_jupyter(self.target_full[0], self.sim_both_full[0])
        return fig, ax

    def plot_segmentation(self, jupyter=False):
        if not jupyter:
            if self.stack:
                fig, ax = plot_segmentation(self.img, self.roi)
            else:
                fig, ax = plot_segmentation(self.img[0], self.roi[0])
        else:
            if self.stack:
                fig, ax = plot_segmentation_jupyter(self.img, self.roi)
            else:
                fig, ax = plot_segmentation_jupyter(self.img[0], self.roi[0])
        return fig, ax

    def plot_losses(self, log=False):
        fig, ax = plt.subplots()
        if not log:
            ax.plot(self.losses.T)
            ax.set_xlabel('Descent step')
            if self.loss_mode == 'mae':
                ax.set_ylabel('Mean absolute error')
            elif self.loss_mode == 'mse':
                ax.set_ylabel('Mean square error')

        else:
            ax.plot(np.log10(self.losses.T))
            ax.set_xlabel('Descent step')
            if self.loss_mode == 'mae':
                ax.set_ylabel('log10(Mean absolute error)')
            elif self.loss_mode == 'mse':
                ax.set_ylabel('log10(Mean square error)')

        return fig, ax

    # def def_roi(self):
    #     r = ROI(self.img, spline=True)
    #     self.roi = r.roi

