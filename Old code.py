class Profile1:
    """
    Parameters
    l = offset
    a = gaussian height
    w = gaussian width
    o = cytbg offset

    """

    def __init__(self, itp, thickness, end_region):
        self.itp = itp
        self.thickness = thickness
        self.end_region = end_region

    def total_profile(self, profile, cytbg, l, a, w, o):
        cytbg_offset = (self.itp / self.thickness) * o
        bgcurve_seg = cytbg[int(l + cytbg_offset):int(l + cytbg_offset) + self.itp]

        line = np.polyfit(
            [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
             np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])],
            [np.mean(profile[:int(len(profile) * self.end_region)]),
             np.mean(profile[int(len(profile) * (1 - self.end_region)):])], 1)

        p0 = line[0]
        p1 = line[1]

        g0 = a * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / w)
        g1 = a * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / w))
        g = np.append(g0, g1)
        y = p0 * (bgcurve_seg + g) + p1
        return y

    def cyt_profile(self, profile, cytbg, l, o, a=None, w=None):
        cytbg_offset = (self.itp / self.thickness) * o
        bgcurve_seg = cytbg[int(l + cytbg_offset):int(l + cytbg_offset) + self.itp]

        line = np.polyfit(
            [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
             np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])],
            [np.mean(profile[:int(len(profile) * self.end_region)]),
             np.mean(profile[int(len(profile) * (1 - self.end_region)):])], 1)

        p0 = line[0]
        p1 = line[1]
        y = p0 * bgcurve_seg + p1
        return y

    def mem_profile(self, profile, cytbg, l, a, w, o):
        cytbg_offset = (self.itp / self.thickness) * o
        bgcurve_seg = cytbg[int(l + cytbg_offset):int(l + cytbg_offset) + self.itp]

        line = np.polyfit(
            [np.mean(bgcurve_seg[:int(len(bgcurve_seg) * self.end_region)]),
             np.mean(bgcurve_seg[int(len(bgcurve_seg) * (1 - self.end_region)):])],
            [np.mean(profile[:int(len(profile) * self.end_region)]),
             np.mean(profile[int(len(profile) * (1 - self.end_region)):])], 1)

        p0 = line[0]
        p1 = line[1]

        g0 = a * np.e ** ((np.array(range(0, (self.itp - int(l)))) - (self.itp - l)) / w)
        g1 = a * np.e ** (-((np.array(range((self.itp - int(l)), self.itp)) - (self.itp - l)) / w))
        g = np.append(g0, g1)
        y = p0 * g
        return y


# Fit type 1

class Segmenter1aSingle(SegmenterParent, Profile1):
    """

    Single channel segmentation, based on background cytoplasmic curves

    Cytbg offset fixed
    Gaussian width free

    Input data:
    img             image
    bgcurve         background curve. Must be 2* wider than the eventual profile for img
    coors           original coordinates

    Parameters:
    freedom         0 = no freedom, 1 = max freedom
    it              Interpolation of profiles/bgcurves
    thickness       thickness of straightened images
    rol_ave         sets the degree of image smoothening
    end_region      for end fitting

    """

    def __init__(self, img, cytbg, coors=None, mag=1, iterations=3, parallel=False, resolution=5, freedom=0.3,
                 periodic=True, thickness=50, itp=1000, rol_ave=50, cytbg_offset=0, end_region=0.1):
        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel,
                                 resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg = cytbg
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.error_check()

    def error_check(self):
        if self.cytbg is not None:
            if len(self.cytbg) != 2 * self.thickness:
                raise Exception('bgcurve must be twice as wide as thickness')

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight[:, x * int(self.mag * self.resolution)], cytbg)
                for x in range(len(straight[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight[:, x * int(self.mag * self.resolution)], cytbg)
        return offsets

    def calc_offset(self, profile, cytbg):
        try:
            params = self.fit_profile_a(profile, cytbg)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)):
                o = np.nan
            elif math.isclose(params[0], (self.itp / 2) * (1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0, 20],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf, 200]),
                               p0=[self.itp / 2, 0, 100])
        return popt

    def fit_profile_a_func(self, x, l, a, w):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=l, a=a, w=w, o=self.cytbg_offset)

        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50), (50, 100)),
                                     args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, l_a_w, profile, cytbg):
        l, a, w = l_a_w
        y = self.total_profile(profile, cytbg, l=l, a=a, w=w, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Segmenter1aDouble(SegmenterParent, Profile1):
    """

    Two-channel segmentation, using two background curves

    Input data:
    img_g           green channel image
    img_r           red channel image
    bgcurve_g       green background curve. Must be 2* wider than the eventual profile for img_r
    bgcurve_r       red background curve. Must be 2* wider than the eventual profile for img_r
    coors           original coordinates

    Input parameters:
    freedom         0 = no freedom, 1 = max freedom
    it              Interpolation of profiles/bgcurves
    thickness       thickness of straightened images
    rol_ave         sets the degree of image smoothening
    end_region      for end fitting

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, coors=None, mag=1, iterations=3, parallel=False, resolution=5,
                 freedom=0.3, periodic=True, thickness=50, itp=1000, rol_ave=50, end_region=0.1, cytbg_offset=0):

        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag, iterations=iterations,
                                 periodic=periodic,
                                 parallel=parallel, resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg_g = cytbg_g
        self.cytbg_r = cytbg_r
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.error_check()

    def error_check(self):
        if self.cytbg_g is not None:
            if len(self.cytbg_g) != 2 * self.thickness:
                raise Exception('bg_g must be twice as wide as thickness')
        if self.cytbg_r is not None:
            if len(self.cytbg_r) != 2 * self.thickness:
                raise Exception('bg_r must be twice as wide as thickness')

    def calc_offsets(self):
        """

        """
        # Straighten
        straight_g = straighten(self.img_g, self.coors, int(self.thickness * self.mag))
        straight_r = straighten(self.img_r, self.coors, int(self.thickness * self.mag))

        # Smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.itp), int(self.rol_ave * self.mag), self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg_g = interp_1d_array(self.cytbg_g, 2 * self.itp)
        cytbg_r = interp_1d_array(self.cytbg_r, 2 * self.itp)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight_g[:, x * int(self.mag * self.resolution)],
                                          straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                          cytbg_r)
                for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight_g[:, x * int(self.mag * self.resolution)],
                                              straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                              cytbg_r)
        return offsets

    def calc_offset(self, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """

        try:
            params = self.fit_profiles_de(profile_g, profile_r, cytbg_g, cytbg_r)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)) or math.isclose(params[0],
                                                                                            (self.itp / 2) * (
                                                                                                        1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profiles_curve_fit(self, profile_g, profile_r, cytbg_g, cytbg_r):
        x = np.stack((np.append(cytbg_g, cytbg_r), np.append((np.hstack((profile_g, np.zeros([self.itp])))),
                                                             (np.hstack((profile_r, np.zeros([self.itp])))))), axis=0)
        popt, pcov = curve_fit(self._func, x, np.append(profile_g, profile_r),
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0, 20, 0, 20],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf, 200, np.inf, 200]),
                               p0=[self.itp / 2, 0, 100, 0, 100])
        return popt

    def _func(self, x, l, a_g, w_g, a_r, w_r):
        profile_g = x[1, :self.itp]
        bgcurve_g = x[0, :2 * self.itp]
        profile_r = x[1, 2 * self.itp:3 * self.itp]
        bgcurve_r = x[0, 2 * self.itp:]
        y = np.zeros([2 * self.itp])
        y[:self.itp] = self.total_profile(profile_g, bgcurve_g, l=l, a=a_g, w=w_g, o=self.cytbg_offset)
        y[self.itp:2 * self.itp] = self.total_profile(profile_r, bgcurve_r, l=l, a=a_r, w=w_r, o=self.cytbg_offset)
        return y

    def fit_profiles_de(self, profile_g, profile_r, cytbg_g, cytbg_r):
        res = differential_evolution(self._mse, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50), (0, 50), (50, 100),
            (50, 100)), args=(profile_g, profile_r, cytbg_g, cytbg_r))
        return res.x

    def _mse(self, l_ag_ar_wg_wr, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """
        l, ag, ar, wg, wr = l_ag_ar_wg_wr
        yg = self.total_profile(profile_g, cytbg_g, l=l, a=ag, w=wg, o=self.cytbg_offset)
        yr = self.total_profile(profile_r, cytbg_r, l=l, a=ar, w=wr, o=self.cytbg_offset)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


class Segmenter1bSingle(SegmenterParent, Profile1):
    """
    Cytbg offset fixed
    Gaussian width fixed

    """

    def __init__(self, img, cytbg=None, coors=None, mag=1, iterations=3, parallel=False, resolution=5, freedom=0.3,
                 periodic=True, thickness=50, itp=1000, rol_ave=50, cytbg_offset=0, end_region=0.1, gwidth=70):
        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel,
                                 resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg = cytbg
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.gwidth = gwidth
        self.error_check()

    def error_check(self):
        if self.cytbg is not None:
            if len(self.cytbg) != 2 * self.thickness:
                raise Exception('bgcurve must be twice as wide as thickness')

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight[:, x * int(self.mag * self.resolution)], cytbg)
                for x in range(len(straight[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight[:, x * int(self.mag * self.resolution)], cytbg)
        return offsets

    def calc_offset(self, profile, cytbg):
        try:
            params = self.fit_profile_a(profile, cytbg)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)):
                o = np.nan
            elif math.isclose(params[0], (self.itp / 2) * (1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf]),
                               p0=[self.itp / 2, 0])
        return popt

    def fit_profile_a_func(self, x, l, a):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=l, a=a, w=self.gwidth, o=self.cytbg_offset)

        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50)),
                                     args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, l_a, profile, cytbg):
        l, a = l_a
        y = self.total_profile(profile, cytbg, l=l, a=a, w=self.gwidth, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Segmenter1bDouble(SegmenterParent, Profile1):
    """
    Cytbg offset fixed
    Gaussuan width fixed

    """

    def __init__(self, img_g, img_r, cytbg_g, cytbg_r, coors=None, mag=1, iterations=3, parallel=False, resolution=5,
                 freedom=0.3, periodic=True, thickness=50, itp=1000, rol_ave=50, cytbg_offset=0, end_region=0.1,
                 gwidth=70):
        SegmenterParent.__init__(self, img_g=img_g, img_r=img_r, coors=coors, mag=mag, iterations=iterations,
                                 periodic=periodic,
                                 parallel=parallel, resolution=resolution, thickness=thickness)
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg_g = cytbg_g
        self.cytbg_r = cytbg_r
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.gwidth = gwidth
        self.error_check()

    def error_check(self):
        if self.cytbg_g is not None:
            if len(self.cytbg_g) != 2 * self.thickness:
                raise Exception('bg_g must be twice as wide as thickness')
        if self.cytbg_r is not None:
            if len(self.cytbg_r) != 2 * self.thickness:
                raise Exception('bg_r must be twice as wide as thickness')

    def calc_offsets(self):
        """

        """
        # Straighten
        straight_g = straighten(self.img_g, self.coors, int(self.thickness * self.mag))
        straight_r = straighten(self.img_r, self.coors, int(self.thickness * self.mag))

        # Smoothen/interpolate images
        straight_g = rolling_ave_2d(interp_2d_array(straight_g, self.itp), int(self.rol_ave * self.mag), self.periodic)
        straight_r = rolling_ave_2d(interp_2d_array(straight_r, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg_g = interp_1d_array(self.cytbg_g, 2 * self.itp)
        cytbg_r = interp_1d_array(self.cytbg_r, 2 * self.itp)

        # Calculate offsets
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.calc_offset)(straight_g[:, x * int(self.mag * self.resolution)],
                                          straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                          cytbg_r)
                for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight_g[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight_g[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.calc_offset(straight_g[:, x * int(self.mag * self.resolution)],
                                              straight_r[:, x * int(self.mag * self.resolution)], cytbg_g,
                                              cytbg_r)
        return offsets

    def calc_offset(self, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """

        try:
            params = self.fit_profiles_curve_fit(profile_g, profile_r, cytbg_g, cytbg_r)
            if math.isclose(params[0], (self.itp / 2) * (1 - self.freedom)) or math.isclose(params[0],
                                                                                            (self.itp / 2) * (
                                                                                                        1 + self.freedom)):
                o = np.nan
            else:
                o = (params[0] - self.itp / 2) / (self.itp / self.thickness)
        except RuntimeError:
            o = np.nan

        return o

    def fit_profiles_curve_fit(self, profile_g, profile_r, cytbg_g, cytbg_r):
        x = np.stack((np.append(cytbg_g, cytbg_r), np.append((np.hstack((profile_g, np.zeros([self.itp])))),
                                                             (np.hstack((profile_r, np.zeros([self.itp])))))), axis=0)
        popt, pcov = curve_fit(self._func, x, np.append(profile_g, profile_r),
                               bounds=([(self.itp / 2) * (1 - self.freedom), 0, 0],
                                       [(self.itp / 2) * (1 + self.freedom), np.inf, np.inf]),
                               p0=[self.itp / 2, 0, 0])
        return popt

    def _func(self, x, l, a_g, a_r):
        profile_g = x[1, :self.itp]
        bgcurve_g = x[0, :2 * self.itp]
        profile_r = x[1, 2 * self.itp:3 * self.itp]
        bgcurve_r = x[0, 2 * self.itp:]
        y = np.zeros([2 * self.itp])
        y[:self.itp] = self.total_profile(profile_g, bgcurve_g, l=l, a=a_g, w=self.gwidth, o=self.cytbg_offset)
        y[self.itp:2 * self.itp] = self.total_profile(profile_r, bgcurve_r, l=l, a=a_r, w=self.gwidth,
                                                      o=self.cytbg_offset)
        return y

    def fit_profiles_de(self, profile_g, profile_r, cytbg_g, cytbg_r):
        res = differential_evolution(self._mse, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (0, 50), (0, 50)),
                                     args=(profile_g, profile_r, cytbg_g, cytbg_r))
        return res.x

    def _mse(self, l_ag_ar, profile_g, profile_r, cytbg_g, cytbg_r):
        """

        """
        l, ag, ar = l_ag_ar
        yg = self.total_profile(profile_g, cytbg_g, l=l, a=ag, w=self.gwidth, o=self.cytbg_offset)
        yr = self.total_profile(profile_r, cytbg_r, l=l, a=ar, w=self.gwidth, o=self.cytbg_offset)
        return np.mean([np.mean((profile_g - yg) ** 2), np.mean((profile_r - yr) ** 2)])


# Fit type 1

class Quantifier1a(Profile1):
    """
    Cytbg offset fixed
    Gaussian width free

    Seems to give exactly the same answer as Quantifier1c, suggests that gaussian width isn't important

    """

    def __init__(self, img, coors, mag, cytbg, thickness=50, itp=1000, cytbg_offset=0, end_region=0.1, rol_ave=10,
                 periodic=True, psize=0.255):

        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)

        self.img = img
        self.coors = coors
        self.mag = mag
        self.thickness = thickness
        self.cytbg = cytbg
        self.periodic = periodic
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.psize = psize

        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if self.cytbg is not None:
            if len(self.cytbg) != 2 * self.thickness:
                raise Exception('cytbg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            pro = interp_1d_array(profile, int(self.thickness * self.mag))
            fbc = interp_1d_array(
                self.cyt_profile(profile, cytbg, l=int(self.itp / 2), o=self.cytbg_offset),
                int(self.thickness * self.mag))
            self.sigs[x] = np.trapz(pro - fbc)

        # Convert to um units
        self.sigs *= self.mag / self.psize

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([0, 20], [np.inf, 200]), p0=[0, 100])
        return popt

    def fit_profile_a_func(self, x, a, w):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=int(self.itp / 2), a=a, w=w, o=self.cytbg_offset)
        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=((0, 50), (50, 100)), args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, a_w, profile, cytbg):
        a, w = a_w
        y = self.total_profile(profile, cytbg, l=int(self.itp / 2), a=a, w=w, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)


class Quantifier1b(Profile1):
    """
    Cytbg offset free
    Gaussian width free
    Used for callibration of gaussian width and cytbg offset

    """

    def __init__(self, img, coors, mag, cytbg, thickness=50, itp=1000, end_region=0.1, rol_ave=10,
                 periodic=True, psize=0.255):
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)

        self.img = img
        self.coors = coors
        self.mag = mag
        self.thickness = thickness
        self.cytbg = cytbg
        self.periodic = periodic
        self.rol_ave = rol_ave
        self.psize = psize

        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if self.cytbg is not None:
            if len(self.cytbg) != 2 * self.thickness:
                raise Exception('cytbg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            params = self.fit_profile_b(profile, cytbg)
            pro = interp_1d_array(profile, int(self.thickness * self.mag))
            fbc = interp_1d_array(
                self.cyt_profile(profile, cytbg, l=int(self.itp / 2), o=params[2]),
                int(self.thickness * self.mag))
            self.sigs[x] = np.trapz(pro - fbc)

        # Convert to um units
        self.sigs *= self.mag / self.psize

    # def fit_profile_a(self, profile, cytbg):
    #     x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
    #     popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
    #                            bounds=([0, 20, -5], [np.inf, 200, 5]), p0=[0, 100, 0])
    #     return popt
    #
    # def fit_profile_a_func(self, x, a, w, o):
    #     profile = x[1, :self.itp]
    #     bgcurve = x[0, :]
    #     y = self.total_profile(profile, bgcurve, l=int(self.itp / 2), a=a, w=w, o=o)
    #     return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=((0, 50), (50, 100), (-5, 5)),
                                     args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, a_w_o, profile, cytbg):
        a, w, o = a_w_o
        y = self.total_profile(profile, cytbg, l=int(self.itp / 2), a=a, w=w, o=o)
        return np.mean((profile - y) ** 2)


class Quantifier1c(Profile1):
    """
    Cytbg offset fixed
    Gaussian width fixed

    Main method for quantification

    """

    def __init__(self, img, coors, mag, cytbg, thickness=50, itp=1000, cytbg_offset=0, end_region=0.1, rol_ave=10,
                 periodic=True, gwidth=70, psize=0.255):
        Profile1.__init__(self, itp=itp, thickness=thickness, end_region=end_region)

        self.img = img
        self.coors = coors
        self.mag = mag
        self.thickness = thickness
        self.cytbg = cytbg
        self.periodic = periodic
        self.gwidth = gwidth
        self.rol_ave = rol_ave
        self.cytbg_offset = cytbg_offset
        self.psize = psize

        self.sigs = np.zeros([len(coors[:, 0])])
        self.error_check()

    def error_check(self):
        if self.cytbg is not None:
            if len(self.cytbg) != 2 * self.thickness:
                raise Exception('cytbg must be twice as wide as thickness')

    def run(self):
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)

        # Get cortical signals
        for x in range(len(straight[0, :])):
            profile = straight[:, x]
            pro = interp_1d_array(profile, int(self.thickness * self.mag))
            fbc = interp_1d_array(
                self.cyt_profile(profile, cytbg, l=int(self.itp / 2), o=self.cytbg_offset),
                int(self.thickness * self.mag))
            self.sigs[x] = np.trapz(pro - fbc)

        # Convert to um units
        self.sigs *= self.mag / self.psize

    def fit_profile_a(self, profile, cytbg):
        x = np.stack((cytbg, (np.hstack((profile, np.zeros([self.itp]))))), axis=0)
        popt, pcov = curve_fit(self.fit_profile_a_func, x, profile,
                               bounds=([0, 20]), p0=[0])
        return popt

    def fit_profile_a_func(self, x, a):
        profile = x[1, :self.itp]
        bgcurve = x[0, :]
        y = self.total_profile(profile, bgcurve, l=int(self.itp / 2), a=a, w=self.gwidth, o=self.cytbg_offset)
        return y

    def fit_profile_b(self, profile, cytbg):
        res = differential_evolution(self.fit_profile_b_func, bounds=((0, 50), (50, 100)), args=(profile, cytbg))
        return res.x

    def fit_profile_b_func(self, a, profile, cytbg):
        y = self.total_profile(profile, cytbg, l=int(self.itp / 2), a=a, w=self.gwidth, o=self.cytbg_offset)
        return np.mean((profile - y) ** 2)




        # class Quantifier2b(Profile2):
        #     """
        #     Cytbg offset free
        #
        #     For calibration of cytbg offset
        #
        #     """
        #
        #     def __init__(self, img, coors, mag, cytbg, membg, thickness=50, itp=1000, rol_ave=10, periodic=True,
        #                  end_region=0.1, psize=0.255):
        #         Profile2.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        #         self.img = img
        #         self.coors = coors
        #         self.mag = mag
        #         self.cytbg = cytbg
        #         self.membg = membg
        #         self.rol_ave = rol_ave
        #         self.periodic = periodic
        #         self.psize = psize
        #
        #         self.sigs = np.zeros([len(coors[:, 0])])
        #         self.cyts = np.zeros([len(coors[:, 0])])
        #         self.straight_mem = np.zeros([self.thickness, len(self.coors[:, 0])])
        #         self.straight_cyt = np.zeros([self.thickness, len(self.coors[:, 0])])
        #
        #         self.error_check()
        #
        #     def error_check(self):
        #         if self.cytbg is not None:
        #             if len(self.cytbg) != 2 * self.thickness:
        #                 raise Exception('cytbg must be twice as wide as thickness')
        #         if self.membg is not None:
        #             if len(self.membg) != 2 * self.thickness:
        #                 raise Exception('membg must be twice as wide as thickness')
        #
        #     def run(self):
        #         straight = straighten(self.img, self.coors, int(self.thickness * self.mag))
        #         straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)
        #         cytbg = interp_1d_array(self.cytbg, 2 * self.itp)
        #         membg = interp_1d_array(self.membg, 2 * self.itp)
        #
        #         # Get cortical/cytoplasmic signals
        #         for x in range(len(straight[0, :])):
        #             profile = straight[:, x]
        #             res = self.fit_profile_a(profile, cytbg, membg)
        #             l = int(self.itp / 2)
        #             j = (res[1] * self.itp) / self.thickness
        #             m1, c1, c2 = self.fix_ends(profile, cytbg[int(l + j):int(l + j) + self.itp],
        #                                        membg[int(l):int(l) + self.itp], res[0])
        #
        #             self.sigs[x] = m1 * res[0]
        #             self.cyts[x] = m1
        #             self.straight_cyt[:, x] = interp_1d_array(self.cyt_profile(profile, cytbg, membg, l=l, a=res[0], o=res[1]),
        #                                                       self.thickness)
        #             self.straight_mem[:, x] = interp_1d_array(self.total_profile(profile, cytbg, membg, l=l, a=res[0],
        #                                                                          o=res[1]), self.thickness) - self.straight_cyt[
        #                                                                                                       :, x]
        #
        #         # Deconvolve membrane signals
        #         self.sigs *= 2 * np.trapz(self.membg[:int(len(self.membg / 2))])
        #
        #         # Convert to um units
        #         self.sigs *= self.mag / self.psize
        #         self.cyts *= (self.mag / self.psize) ** 2
        #
        #     def fit_profile_a(self, profile, cytbg, membg):
        #         res = differential_evolution(self.fit_profile_a_func, bounds=((-5, 50), (-5, 5)), args=(profile, cytbg, membg))
        #         return res.x
        #
        #     def fit_profile_a_func(self, a_o, profile, cytbg, membg):
        #         a, o = a_o
        #         y = self.total_profile(profile, cytbg, membg, l=int(self.itp / 2), a=a, o=o)
        #         return np.mean((profile - y) ** 2)

class SegmenterSingleB(SegmenterParent, Profile):
    """
    Fit profiles to cytoplasmic background + membrane background

    Cytbg offset free

    """

    def __init__(self, img, cytbg, membg, coors=None, mag=1, iterations=2, parallel=False, resolution=5, freedom=0.3,
                 periodic=True, thickness=50, itp=1000, rol_ave=50, end_region=0.1):

        SegmenterParent.__init__(self, img=img, coors=coors, mag=mag, iterations=iterations, periodic=periodic,
                                 parallel=parallel,
                                 resolution=resolution, thickness=thickness)

        Profile.__init__(self, itp=itp, thickness=thickness, end_region=end_region)
        self.cytbg = cytbg
        self.membg = membg
        self.freedom = freedom
        self.rol_ave = rol_ave
        self.error_check()

    def error_check(self):
        if self.cytbg is not None:
            if len(self.cytbg) != 2 * self.thickness:
                raise Exception('cytbg must be twice as wide as thickness')
        if self.membg is not None:
            if len(self.membg) != 2 * self.thickness:
                raise Exception('membg must be twice as wide as thickness')

    def calc_offsets(self):
        # Straighten
        straight = straighten(self.img, self.coors, int(self.thickness * self.mag))

        # Filter/smoothen/interpolate images
        straight = rolling_ave_2d(interp_2d_array(straight, self.itp), int(self.rol_ave * self.mag), self.periodic)

        # Interpolate bgcurves
        cytbg = interp_1d_array(self.cytbg, 2 * self.itp)
        membg = interp_1d_array(self.membg, 2 * self.itp)

        # Fit
        if self.parallel:
            offsets = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.fit_profile)(straight[:, x * int(self.mag * self.resolution)], cytbg, membg)
                for x in range(len(straight[0, :]) // int(self.mag * self.resolution))))
        else:
            offsets = np.zeros(len(straight[0, :]) // int(self.mag * self.resolution))
            for x in range(len(straight[0, :]) // int(self.mag * self.resolution)):
                offsets[x] = self.fit_profile(straight[:, x * int(self.mag * self.resolution)], cytbg, membg)

        return offsets

    def fit_profile(self, profile, cytbg, membg):
        res = differential_evolution(self.mse, bounds=(
            ((self.itp / 2) * (1 - self.freedom), (self.itp / 2) * (1 + self.freedom)), (-5, 50), (-5, 5)),
                                     args=(profile, cytbg, membg))
        offset = (res.x[0] - self.itp / 2) / (self.itp / self.thickness)

        return offset

    def mse(self, l_a_o, profile, cytbg, membg):
        l, a, o = l_a_o
        y = self.total_profile(profile, cytbg, membg, l=l, a=a, o=o)
        return np.mean((profile - y) ** 2)



def fix_ends(curve1, curve2):
    """
    Used for background subtraction. Returns fitted bgcurve which can then be subtracted from the signal curve
    Bg fitted by fixing ends

    Fixes ends of curve 2 to ends of curve 1

    :param curve1:
    :param curve2:
    :return:
    """

    # Fix ends
    line = np.polyfit(
        [np.mean(curve2[:int(len(curve2) * 0.2)]), np.mean(curve2[int(len(curve2) * 0.8):])],
        [np.mean(curve1[:int(len(curve1) * 0.2)]), np.mean(curve1[int(len(curve1) * 0.8):])], 1)

    # Create new bgcurve
    curve2 = curve2 * line[0] + line[1]

    return curve2


def offset_line(line, offset):
    """
    Moves a straight line of coordinates perpendicular to itself

    :param line:
    :param offset:
    :return:
    """

    xcoors = line[:, 0]
    ycoors = line[:, 1]

    # Create coordinates
    rise = ycoors[1] - ycoors[0]
    run = xcoors[1] - xcoors[0]
    bisectorgrad = rise / run
    tangentgrad = -1 / bisectorgrad

    xchange = ((offset ** 2) / (1 + tangentgrad ** 2)) ** 0.5
    ychange = xchange / abs(bisectorgrad)
    newxs = xcoors + np.sign(rise) * np.sign(offset) * xchange
    newys = ycoors - np.sign(run) * np.sign(offset) * ychange

    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)
    return newcoors


def extend_line(line, extend):
    """
    Extends a straight line of coordinates along itself

    Should adjust to allow shrinking
    :param line:
    :param extend: e.g. 1.1 = 10% longer
    :return:
    """

    xcoors = line[:, 0]
    ycoors = line[:, 1]

    len = np.hypot((xcoors[0] - xcoors[1]), (ycoors[0] - ycoors[1]))
    extension = (extend - 1) * len * 0.5

    rise = ycoors[1] - ycoors[0]
    run = xcoors[1] - xcoors[0]
    bisectorgrad = rise / run
    tangentgrad = -1 / bisectorgrad

    xchange = ((extension ** 2) / (1 + bisectorgrad ** 2)) ** 0.5
    ychange = xchange / abs(tangentgrad)
    newxs = xcoors - np.sign(rise) * np.sign(tangentgrad) * xchange * np.array([-1, 1])
    newys = ycoors - np.sign(run) * np.sign(tangentgrad) * ychange * np.array([-1, 1])
    newcoors = np.swapaxes(np.vstack([newxs, newys]), 0, 1)
    return newcoors


