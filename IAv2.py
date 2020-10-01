from IA import *
import tensorflow_probability as tfp


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
    rol_ave            width of rolling average
    rotate             if True, will automatically rotate ROI so that the first/last points are at the end of the long
                       axis
    zerocap            if True, prevents negative membane and cytoplasm values
    nfits              performs this many fits at regular intervals around ROI
    iterations         if >1, adjusts ROI and re-fits
    interp             interpolation type (linear or cubic)
    uni_cyt            globally fit uniform cytoplasm
    uni_mem            globally fit uniform membrane
    bg_subtract        if True, will estimate and subtract background signal prior to quantification

    Saving:
    save_path          destination to save results, will create if it doesn't already exist


    """

    def __init__(self, img, cytbg=None, membg=None, sigma=None, roi=None, freedom=0.5, periodic=True, thickness=50,
                 rol_ave=10, rotate=False, zerocap=True, nfits=None, iterations=1, interp='cubic', save_path=None,
                 bg_subtract=False, uni_cyt=False, uni_mem=False, lr=0.01):

        # Image / stack
        self.img = img

        # ROI
        self.roi_init = roi
        self.roi = roi
        self.periodic = periodic

        # Background subtraction
        self.bg_subtract = bg_subtract

        # Fitting mode
        self.uni_cyt = uni_cyt
        self.uni_mem = uni_mem

        # Fitting parameters
        self.iterations = iterations
        self.thickness = thickness
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.rotate = rotate
        self.zerocap = zerocap
        self.sigma = sigma
        self.nfits = nfits
        self.interp = interp
        self.lr = lr

        # Saving
        self.save_path = save_path

        # # Background curves
        self.cytbg = cytbg
        self.membg = membg

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
                self.reset_res()
                self.fit()

        # Simulate images
        self.sim_images()

        # Save
        if self.save_path is not None:
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
        straight_filtered_itp = interp_2d_array(self.straight_filtered, self.nfits, ax=0, method=self.interp)

        # Normalise
        self.norm = np.max(straight_filtered_itp)
        self.y = straight_filtered_itp / self.norm

        # Fit
        self._fit()

        # Interpolate
        self.offsets_full = interp_1d_array(self.offsets, len(self.roi[:, 0]), method=self.interp)
        self.cyts_full = interp_1d_array(self.cyts, len(self.roi[:, 0]), method=self.interp)
        self.mems_full = interp_1d_array(self.mems, len(self.roi[:, 0]), method=self.interp)

    """
    Fitting

    """

    def _fit(self):

        # Create tensors
        offsets = tf.Variable(self.offsets)
        if self.uni_cyt:
            cyts = tf.Variable(self.cyts[0])
        else:
            cyts = tf.Variable(self.cyts)
        if self.uni_mem:
            mems = tf.Variable(self.mems[0])
        else:
            mems = tf.Variable(self.mems)
        var_list = [offsets, cyts, mems]

        # Run optimisation
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        # losses = np.zeros(1000)
        for i in range(1000):
            # losses[i] = calc_loss()
            opt.minimize(lambda: self._loss_function(cyts, mems, offsets), var_list=var_list)

        # Concentration bounds
        if self.zerocap:
            mems = tf.math.maximum(mems, 0)
            cyts = tf.math.maximum(cyts, 0)

        self.mems[:] = mems.numpy() * self.norm
        self.cyts[:] = cyts.numpy() * self.norm
        self.offsets[:] = offsets.numpy()

    def _sim_img(self, cyts, mems, offsets):
        # Concentration bounds
        if self.zerocap:
            mems_ = tf.math.maximum(mems, 0)
            cyts_ = tf.math.maximum(cyts, 0)
        else:
            mems_ = mems
            cyts_ = cyts

        # Offset bounds
        offsets_ = tf.math.maximum(offsets, -self.freedom * self.thickness / 2)
        offsets_ = tf.math.minimum(offsets_, self.freedom * self.thickness / 2)

        # Alignment
        positions = tf.reshape(tf.reshape(tf.tile(np.arange(self.thickness, dtype=np.float64), [self.nfits]),
                                          [self.nfits, self.thickness]) + tf.expand_dims(offsets_, -1), [-1])

        # Mem curve
        if self.membg is None:
            # Default
            mem_curve = tf.math.exp(-((positions - self.thickness / 2) ** 2) / (2 * self.sigma ** 2))
        else:
            # Custom
            mem_curve = tfp.math.interp_regular_1d_grid(y_ref=self.membg, x_ref_min=0, x_ref_max=1,
                                                        x=(positions + self.thickness / 2) / (self.thickness * 2))

        # Cyt curve:
        if self.membg is None:
            # Default
            cyt_curve = (1 + tf.math.erf((positions - self.thickness / 2) / self.sigma)) / 2
        else:
            # Custom
            cyt_curve = tfp.math.interp_regular_1d_grid(y_ref=self.cytbg, x_ref_min=0, x_ref_max=1,
                                                        x=(positions + self.thickness / 2) / (self.thickness * 2))

        # Reshape
        cyt_curve_ = tf.reshape(cyt_curve, [self.nfits, self.thickness])
        mem_curve_ = tf.reshape(mem_curve, [self.nfits, self.thickness])

        # Calculate y^
        mem_total = mem_curve_ * tf.expand_dims(mems_, axis=-1)
        cyt_total = cyt_curve_ * tf.expand_dims(cyts_, axis=-1)
        yhat = tf.math.add(mem_total, cyt_total)
        return yhat

    def _loss_function(self, cyts, mems, offsets):
        yhat = self._sim_img(cyts, mems, offsets)
        loss = tf.math.reduce_mean((yhat - self.y.T) ** 2)
        return loss

    """
    Misc

    """

    def sim_images(self):
        """
        Creates simulated images based on fit results

        Todo: do without a loop

        """
        for x in range(len(self.roi[:, 0])):
            c = self.cyts_full[x]
            m = self.mems_full[x]
            o = self.offsets_full[x]

            itp_pos = np.linspace(o, o + self.thickness, self.thickness)

            if self.cytbg is None:
                self.straight_cyt[:, x] = c * (1 + erf((itp_pos - self.thickness / 2) / self.sigma)) / 2
            else:
                self.straight_cyt[:, x] = c * np.squeeze(
                    tfp.math.interp_regular_1d_grid(y_ref=self.cytbg, x_ref_min=0, x_ref_max=1,
                                                    x=(itp_pos + self.thickness / 2) / (
                                                            self.thickness * 2)))
            if self.membg is None:
                self.straight_mem[:, x] = m * np.exp(-((itp_pos - self.thickness / 2) ** 2) / (2 * self.sigma ** 2))
            else:
                self.straight_mem[:, x] = m * np.squeeze(
                    tfp.math.interp_regular_1d_grid(y_ref=self.membg, x_ref_min=0, x_ref_max=1,
                                                    x=(itp_pos + self.thickness / 2) / (
                                                            self.thickness * 2)))

        self.straight_fit = self.straight_cyt + self.straight_mem
        self.straight_resids = self.straight - self.straight_fit
        self.straight_resids_pos = np.clip(self.straight_resids, a_min=0, a_max=None)
        self.straight_resids_neg = abs(np.clip(self.straight_resids, a_min=None, a_max=0))

    def adjust_roi(self):
        """
        Can do after a preliminary fit to refine coordinates
        Must refit after doing this

        """

        # Offset coordinates
        self.roi = offset_coordinates(self.roi, self.offsets_full)

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
