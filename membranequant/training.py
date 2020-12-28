import numpy as np
import tensorflow as tf
from .funcs import error_func, gaus, interp_1d_array, interp_2d_array, straighten, rolling_ave_2d
from .quantification import ImageQuant


class ExtractProfiles:
    """
    Gradient descent fitting of cytbg and membg to images of polarised cells
    Assumes uniform cytoplasmic component and spatially varying membrane component

    This version:
    - fit with spatially varying cytoplasm
    - use to approximate global conc
    - fit curves allowing cyt to change

    """

    def __init__(self, images, rois, thickness=100, iterations=3, sigma=3, learning_rate=0.001, nfits=100, rol_ave=10):

        # Input data (lists)
        self.images = images
        self.rois = rois

        # Parameters
        self.thickness = int(thickness)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss = np.zeros([1000 * self.iterations])
        self.nfits = nfits
        self.rol_ave = rol_ave
        self.itp = 10
        self.interp = 'cubic'

        # Normalise images
        self.images_norm = [x / np.mean(x.flatten()[np.argsort(x.flatten())[-100:]]) for x in self.images]

        # Internal
        self.y = np.zeros([len(self.images), 1000, self.thickness])
        self.mems = np.zeros([len(self.images), 1000])
        self.cyts = np.zeros([len(self.images)])
        self.it = 0

        # Output
        self.cyt_curve = (1 + error_func(np.arange(thickness), thickness / 2, sigma)) / 2
        self.mem_curve = gaus(np.arange(thickness), thickness / 2, sigma)

    def fit(self, i):
        # Set up quantifier
        m = ImageQuant(img=self.images_norm[i], membg=self.mem_curve, cytbg=self.cyt_curve, roi=self.rois[i],
                       nfits=self.nfits, freedom=0.2, periodic=True, thickness=int(self.thickness / 2),
                       rol_ave=self.rol_ave, uni_cyt=False, uni_mem=False,
                       zerocap=True, iterations=1, interp=self.interp, save_path='.caches/')

        # Fit
        m.run()
        self.mems[i, :] = interp_1d_array(m.mems, 1000, method='linear')
        self.cyts[i] = np.mean(m.cyts)

        # Adjust roi
        m.adjust_roi()
        self.rois[i] = m.roi

        # Re-straighten with new ROI
        y = straighten(m.img, self.rois[i], self.thickness)
        y = rolling_ave_2d(y, m.rol_ave, m.periodic)
        self.y[i, :, :] = interp_2d_array(y, 1000, ax=0, method=m.interp).T

    def optimise(self):
        # Create tensors
        mems = tf.Variable(self.mems)
        cyts = tf.Variable(self.cyts)
        mem_curve = tf.Variable(self.mem_curve)
        cyt_curve = tf.Variable(self.cyt_curve)

        # Run optimisation
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        for i in range(500):
            self.loss[i + (self.it * 500)] = self.calc_loss(mems, cyts, mem_curve,
                                                            cyt_curve)  # inefficient: calculating loss twice
            opt.minimize(lambda: self.calc_loss(mems, cyts, mem_curve, cyt_curve),
                         var_list=[mem_curve, cyt_curve, cyts])

        # Normalise
        self.mem_curve = mem_curve.numpy()
        self.cyt_curve = cyt_curve.numpy()
        self.mem_curve /= max(self.mem_curve)
        self.cyt_curve /= max(self.cyt_curve)

    def sim_images(self, mems, cyts, mem_curve, cyt_curve):
        mem_curve_norm = mem_curve  # / tf.reduce_max(mem_curve)
        cyt_curve_norm = cyt_curve / tf.reduce_max(cyt_curve)
        mem_total = tf.tensordot(mems, mem_curve_norm, axes=0)
        cyt_total = tf.tensordot(tf.reshape(tf.tile(cyts, [1000]), [tf.shape(cyts)[0], 1000]),
                                 cyt_curve_norm, axes=0)
        yhat = tf.math.add(mem_total, cyt_total)
        return yhat

    def calc_loss(self, mems, cyts, mem_curve, cyt_curve):
        yhat = self.sim_images(mems, cyts, mem_curve, cyt_curve)
        loss = tf.math.reduce_mean((yhat - self.y) ** 2)
        return loss

    def run(self):
        for it in range(self.iterations):
            print(it)
            self.it = it

            # Fit each image independently
            for i in range(len(self.images)):
                self.fit(i)

            # Refine cyt/mem curves
            self.optimise()


class GenerateProfile:
    """
    Class for getting cytoplasmic or membrane profiles from images expressing only cytoplasmic or membrane protein

    To do:
    - test
    - option for periodic

    """

    def __init__(self, img, roi, thickness=100, iterations=3, sigma=3, nfits=100, rol_ave=10, profile_type='cyt',
                 periodic=True, bg_subtract=False):
        # Input data (lists)
        self.img = img
        self.roi = roi

        # Parameters
        self.thickness = int(thickness)
        self.iterations = iterations
        self.loss = np.zeros([1000 * self.iterations])
        self.nfits = nfits
        self.rol_ave = rol_ave
        self.itp = 10
        self.interp = 'cubic'
        self.sigma = sigma
        self.profile_type = profile_type
        self.periodic = periodic
        self.bg_subtract = bg_subtract

        # Output
        self.profile = None

    def fit(self):
        if self.profile_type == 'cyt':
            cyt = self.profile
            mem = None
        elif self.profile_type == 'mem':
            cyt = None
            mem = self.profile

        # Set up quantifier
        m = ImageQuant(img=self.img, sigma=self.sigma, cytbg=cyt, membg=mem, roi=self.roi,
                       nfits=self.nfits, freedom=0.2, periodic=self.periodic, thickness=int(self.thickness / 2),
                       rol_ave=self.rol_ave, uni_cyt=False, uni_mem=False,
                       zerocap=True, iterations=1, interp=self.interp, bg_subtract=self.bg_subtract)

        # Fit
        m.run()

        # Adjust roi
        m.adjust_roi()
        self.roi = m.roi

        # Re-straighten with new ROI
        y = straighten(m.img, self.roi, self.thickness)

        # Background subtract
        if self.bg_subtract:
            y -= np.mean(y[:5, :])

        # Average
        self.profile = np.mean(y, axis=1)

    def run(self):
        for it in range(self.iterations):
            self.fit()


class ExtractMembraneProfile:
    """
    Gradient descent fitting of cytbg and membg to images of polarised cells
    Assumes uniform cytoplasmic component and spatially varying membrane component

    """

    def __init__(self, images, rois, cyt_curve, thickness=100, iterations=3, sigma=3, learning_rate=0.001, nfits=100,
                 rol_ave=10):

        # Input data (lists)
        self.images = images
        self.rois = rois
        self.cyt_curve = cyt_curve

        # Parameters
        self.thickness = int(thickness)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss = np.zeros([1000 * self.iterations])
        self.nfits = nfits
        self.rol_ave = rol_ave
        self.itp = 10
        self.interp = 'cubic'

        # Normalise images
        self.images_norm = [x / np.mean(x.flatten()[np.argsort(x.flatten())[-100:]]) for x in self.images]

        # Internal
        self.y = np.zeros([len(self.images), 1000, self.thickness])
        self.mems = np.zeros([len(self.images), 1000])
        self.cyts = np.zeros([len(self.images)])
        self.it = 0

        # Output
        self.mem_curve = gaus(np.arange(thickness), thickness / 2, sigma)

    def fit(self, i):
        # Set up quantifier
        m = ImageQuant(img=self.images_norm[i], membg=self.mem_curve, cytbg=self.cyt_curve, roi=self.rois[i],
                       nfits=self.nfits, freedom=0.2, periodic=True, thickness=int(self.thickness / 2),
                       rol_ave=self.rol_ave, uni_cyt=False, uni_mem=False,
                       zerocap=True, iterations=1, interp=self.interp, save_path='.caches/')

        # Fit
        m.run()
        self.mems[i, :] = interp_1d_array(m.mems, 1000, method='linear')
        self.cyts[i] = np.mean(m.cyts)

        # Adjust roi
        m.adjust_roi()
        self.rois[i] = m.roi

        # Re-straighten with new ROI
        y = straighten(m.img, self.rois[i], self.thickness)
        y = rolling_ave_2d(y, m.rol_ave, m.periodic)
        self.y[i, :, :] = interp_2d_array(y, 1000, ax=0, method=m.interp).T

    def optimise(self):
        # Create tensors
        mems = tf.Variable(self.mems)
        cyts = tf.Variable(self.cyts)
        mem_curve = tf.Variable(self.mem_curve)
        cyt_curve = tf.constant(self.cyt_curve)

        # Run optimisation
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        for i in range(500):
            self.loss[i + (self.it * 500)] = self.calc_loss(mems, cyts, mem_curve,
                                                            cyt_curve)  # inefficient: calculating loss twice
            opt.minimize(lambda: self.calc_loss(mems, cyts, mem_curve, cyt_curve),
                         var_list=[mem_curve, cyts, mems])

        # Normalise
        self.mem_curve = tf.math.maximum(mem_curve, 0)
        self.mem_curve = self.mem_curve.numpy()
        self.mem_curve /= max(self.mem_curve)

    def sim_images(self, mems, cyts, mem_curve, cyt_curve):

        # Cap at zero
        mems = tf.math.maximum(mems, 0)
        cyts = tf.math.maximum(cyts, 0)
        mem_curve = tf.math.maximum(mem_curve, 0)

        # Normalise curve
        mem_curve_norm = mem_curve / tf.reduce_max(mem_curve)

        # Simulate image
        mem_total = tf.tensordot(mems, mem_curve_norm, axes=0)
        cyt_total = tf.tensordot(tf.reshape(tf.tile(cyts, [1000]), [tf.shape(cyts)[0], 1000]), cyt_curve, axes=0)
        yhat = tf.math.add(mem_total, cyt_total)
        return yhat

    def calc_loss(self, mems, cyts, mem_curve, cyt_curve):
        yhat = self.sim_images(mems, cyts, mem_curve, cyt_curve)
        loss = tf.math.reduce_mean((yhat - self.y) ** 2)
        return loss

    def run(self):
        for it in range(self.iterations):
            print(it)
            self.it = it

            # Fit each image independently
            for i in range(len(self.images)):
                self.fit(i)

            # Refine cyt/mem curves
            self.optimise()
