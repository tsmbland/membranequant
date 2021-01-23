import numpy as np
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
import multiprocessing
import os
from .funcs import straighten, rolling_ave_2d, interp_roi, interp_1d_array, interp_2d_array, rotate_roi, save_img, \
    offset_coordinates, error_func, gaus, spline_roi


class ImageQuant2:
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
            self.cytbg_itp = interp_1d_array(self.cytbg, 2 * self.thickness_itp, method=self.interp)
        if membg is None:
            self.membg = gaus(np.arange(thickness * 2), thickness, self.sigma)
            self.membg_itp = gaus(np.arange(2 * self.thickness_itp), self.thickness_itp, self.sigma * self.itp)
        else:
            self.membg = membg
            self.membg_itp = interp_1d_array(self.membg, 2 * self.thickness_itp, method=self.interp)

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
        for i in range(self.iterations):
            if i > 0:
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
            bg_intensity = np.mean(self.straight[:5, :])
            self.straight -= bg_intensity
            self.img -= bg_intensity

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
        self.offsets_full = interp_1d_array(self.offsets, len(self.roi[:, 0]), method='linear')
        self.cyts_full = interp_1d_array(self.cyts, len(self.roi[:, 0]), method='linear')
        self.mems_full = interp_1d_array(self.mems, len(self.roi[:, 0]), method='linear')

    """
    METHOD 0: Non-uniform cytoplasm

    """

    def _fit_profile(self, profile):
        if self.zerocap:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (0, max(2 * max(profile), 0)), (0, max(2 * max(profile), 0)))
        else:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (-0.2 * max(profile), 2 * max(profile)), (-0.2 * max(profile), 2 * max(profile)))
        res = differential_evolution(self._mse, bounds=bounds, args=(profile,), tol=0.2)
        o = (res.x[0] - self.thickness_itp / 2) / self.itp
        return o, res.x[1], res.x[2]

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
            bounds = (0, max(2 * np.percentile(straight, 95), 0))
        else:
            bounds = (-0.2 * np.percentile(straight, 95), 2 * np.percentile(straight, 95))

        res = self.iterative_opt(self._fit_profile_1_func, p_range=bounds, args=(straight,), N=5, iterations=7)

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
                (0, max(2 * max(profile), 0)))
        else:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (-0.2 * max(profile), 2 * max(profile)))
        res = self.iterative_opt_2d(self._fit_profile_2_func, p1_range=bounds[0], p2_range=bounds[1], N=5, iterations=7,
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
                (0, max(2 * max(profile), 0)))
        else:
            bounds = (
                ((self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                (-0.2 * max(profile), 2 * max(profile)))
        res = self.iterative_opt_2d(self._fit_profile_2_func, p1_range=bounds[0], p2_range=bounds[1], N=5, iterations=7,
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
            bounds = (0, max(2 * np.percentile(straight, 95), 0))
        else:
            bounds = (-0.2 * np.percentile(straight, 95), 2 * np.percentile(straight, 95))
        res = self.iterative_opt_2d(self._fit_profile_ucum_func, p1_range=bounds,
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

        res, fun = self.iterative_opt(self._fit_profile_ucum_2_func, p_range=(
            (self.thickness_itp / 2) * (1 - self.freedom), (self.thickness_itp / 2) * (1 + self.freedom)),
                                      args=(profile, c, m), N=5, iterations=3)

        o = (res - self.thickness_itp / 2) / self.itp
        return o

    def _fit_profile_ucum_2b(self, profile, c, m):
        """
        Fitting local offsets, returns error

        """

        res, fun = self.iterative_opt(self._fit_profile_ucum_2_func, p_range=(
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

        # Offset coordinates
        self.roi = offset_coordinates(self.roi, self.offsets_full)

        # Fit spline
        self.roi = spline_roi(roi=self.roi, periodic=self.periodic, s=100)

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

    def iterative_opt(self, func, p_range, N, iterations, args=()):
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

    def iterative_opt_2d(self, func, p1_range, p2_range, N, iterations, args=()):
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