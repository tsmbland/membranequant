from ImageAnalysis.IA import *


def animation1(img):
    """
    Animation explaining how offset calculation works

    :param img:
    :return:
    """

    plt.clf()
    fig = plt.figure()
    ax = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((4, 6), (0, 2), colspan=4)
    ax3 = plt.subplot2grid((4, 6), (1, 2), colspan=4)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    frames = np.linspace(0, len(img[0, :]), 100).astype(int)

    offsets = calc_offsets2(img)

    img2 = interp(img, 1000)
    img3 = rolling_ave(img2, 5)
    img4 = savitsky_golay(img3, 251, 5)

    def update_anim(x):
        ax.clear()
        ax2.clear()
        ax3.clear()
        ax.plot(img4[:, x])
        ax.axvline(500, c='k', linestyle='--')
        ax.axvline(500 - offsets[x] * 20, c='k')
        ax.set_ylim([img4.min(), img4.max()])
        ax2.plot(offsets[:x])
        ax2.set_ylim([-10, 10])
        ax2.set_xlim([0, len(img4[0, :])])
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax.set_xlabel('y')
        ax.set_ylabel('Intensity')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y offset')
        ax2.axhline(0, c='k', linestyle='--')
        ax3.imshow(img, cmap='gray')
        ax3.axvline(x, c='r', linestyle='--')
        ax2.axvline(x, c='r', linestyle='--')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.tight_layout()

    paranim = animation.FuncAnimation(fig, update_anim, frames=iter(frames), save_count=len(frames))
    writer = animation.writers['ffmpeg']
    writer = writer(fps=20, bitrate=2000)
    paranim.save('../../../../Desktop/animation.mp4', writer=writer)


def fit_coordinates(data, settings, func):
    """
    Old function for fitting coordinates, works in conjunction with ImageJ

    :param data:
    :param settings:
    :param func:
    :return:
    """

    img = af_subtraction(data.GFP_straight, data.AF_straight, settings)
    newcoors = offset_coordinates(data.ROI_orig, func(img))
    # newcoors = rotate_coors(newcoors)
    np.savetxt('%s/ROI_fitted.txt' % data.direc, newcoors, fmt='%.4f', delimiter='\t')


def align(img, offsets):
    """
    Aligns the columns in img according to offsets
    Array stays same size as original, unfilled 'pixels' given value nan

    :param img:
    :param offsets:
    :return:
    """

    aligned = np.zeros([len(img[:, 0]), len(img[0, :])])
    aligned[:, :] = np.nan
    for x in range(len(img[0, :])):
        if offsets[x] == 0:
            aligned[:, x] = img[:, x]
        if offsets[x] > 0:
            aligned[:-int(offsets[x]), x] = img[int(offsets[x]):, x]
        if offsets[x] < 0:
            aligned[int(abs(offsets[x])):, x] = img[:-int(abs(offsets[x])), x]

    return aligned


def batch1(direc):
    """
    To perform batch operations on embryos
    Use as a template

    :param direc: directory containing embryo folders
    :return:
    """

    embryos = dirslist(direc)
    for embryo in range(1):
        print('%s' % embryo)
        data = Data('%s/%s' % (direc, embryos[embryo]))
        settings = Settings(m=2.06832924528)
        # fit_coordinates(data, settings, calc_offsets2)
        # animation1(af_subtraction5(data.GFP_straight, data.AF_straight, settings))
        # animation1(data.GFP_straight)

        # img1 = loadimage('%s/%s/_%s_w2488 SP 535-50 Nelio_straight.tif' % (
        #     direc, embryos[embryo], embryos[embryo]))
        # img2 = loadimage('%s/%s/_%s_w3488 SP 630-75 Nelio-AF_straight.tif' % (
        #     direc, embryos[embryo], embryos[embryo]))
        # img3 = af_subtraction(img1, img2, m=2.10464091196, c=0)
        # img4 = loadimage('%s/%s/_%s_w4561 SP 630-75 Nelio_straight.tif' % (
        #     direc, embryos[embryo], embryos[embryo]))
        #
        # coors = np.loadtxt('%s/%s/ROI_orig.txt' % (direc, embryos[embryo]))
        #
        # offsets = calc_offsets2(img3)
        # newcoors = offset_coordinates(coors, offsets)
        # np.savetxt('%s/%s/ROI_fitted.txt' % (direc, embryos[embryo]), newcoors, fmt='%.4f',
        #            delimiter='\t')


def batch2(direc):
    """
    To perform batch operations on embryos
    Use as a template

    :param direc: directory containing conditions folders (which contain embryo folders)
    :return:
    """

    conditions = dirslist(direc)
    for condition in conditions:
        embryos = dirslist('%s/%s' % (direc, condition))
        for embryo in range(len(embryos)):
            print('%s/%s' % (condition, embryo))

            img1 = loadimage('%s/%s/%s/_%s_w2488 SP 535-50 Nelio_straight.tif' % (
                direc, condition, embryos[embryo], embryos[embryo]))
            img2 = loadimage('%s/%s/%s/_%s_w3488 SP 630-75 Nelio-AF_straight.tif' % (
                direc, condition, embryos[embryo], embryos[embryo]))
            img3 = af_subtraction(img1, img2, m=2.10464091196, c=0)

            coors = np.loadtxt('%s/%s/%s/ROI_fitted.txt' % (direc, condition, embryos[embryo]))

            offsets = calc_offsets2(img3)
            newcoors = offset_coordinates(coors, offsets)
            np.savetxt('%s/%s/%s/ROI_fitted.txt' % (direc, condition, embryos[embryo]), newcoors, fmt='%.4f',
                       delimiter='\t')


def n2_analysis2(n2direcs, mchdirecs):
    """
    Cytoplasmic mean correlation between the two channels for N2 embryos
    When something is in the mCherry channel

    :param direc:
    :return:
    """

    xdata = []
    ydata = []

    for direc in n2direcs:
        embryos = dirslist(direc)

        # Cytoplasmic means
        for embryo in range(len(embryos)):
            img1 = loadimage(
                glob.glob('%s/%s/_%s_w?488 SP 535-50 Nelio.TIF' % (direc, embryos[embryo], embryos[embryo]))[0])
            img2 = loadimage(
                glob.glob('%s/%s/_%s_w?488 SP 630-75 Nelio-AF.TIF' % (direc, embryos[embryo], embryos[embryo]))[0])
            coors = np.loadtxt('%s/%s/ROI_orig.txt' % (direc, embryos[embryo]))
            xdata.extend([cytoconc(img2, coors)])
            ydata.extend([cytoconc(img1, coors)])

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    plt.scatter(xdata, ydata, c='b')

    def func1(x, m, g):
        y = m * x + g
        return y

    popt, pcov = curve_fit(func1, xdata, ydata)
    # a = np.mean(ydata / xdata)
    x = np.array([0.9 * min(xdata.flatten()), 1.1 * max(xdata.flatten())])
    y = (popt[0] * x + popt[1])
    plt.plot(x, y, c='b')
    print(popt)
    print(np.mean(xdata))

    xdata = []
    ydata = []

    for direc in mchdirecs:
        embryos = dirslist(direc)

        # Cytoplasmic means
        for embryo in range(len(embryos)):
            img1 = loadimage(
                glob.glob('%s/%s/_%s_w?488 SP 535-50 Nelio.TIF' % (direc, embryos[embryo], embryos[embryo]))[0])
            img2 = loadimage(
                glob.glob('%s/%s/_%s_w?488 SP 630-75 Nelio-AF.TIF' % (direc, embryos[embryo], embryos[embryo]))[0])
            coors = np.loadtxt('%s/%s/ROI_orig.txt' % (direc, embryos[embryo]))
            plt.scatter(cytoconc(img2, coors), cytoconc(img1, coors))
            xdata.extend([cytoconc(img2, coors)])
            ydata.extend([cytoconc(img1, coors)])

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    plt.scatter(xdata, ydata, c='r')

    def func1(x, m, g):
        y = m * x + g
        return y

    popt, pcov = curve_fit(func1, xdata, ydata)
    x = np.array([0.9 * min(xdata.flatten()), 1.1 * max(xdata.flatten())])
    y = (popt[0] * x) + popt[1]
    plt.plot(x, y, c='r')
    print([popt])

    plt.xlabel('AF channel mean cytoplasmic intensity')
    plt.ylabel('GFP channel mean cytoplasmic intensity')

    plt.xlim([3000, 5500])
    plt.ylim([6000, 10000])

    print(np.mean(xdata))


def dataframe(direc, name):
    """
    Use as a template to make a more specific function

    :param direc:
    :param name:
    :return:
    """

    conditions = dirslist(direc)

    df = pd.DataFrame(columns=['condition', 'ch1', 'ch2', 'signal'])

    for condition in conditions:
        embryos = dirslist('%s/%s' % (direc, condition))
        for embryo in embryos:
            ch1 = cytoconc(
                loadimage('%s/%s/%s/_%s_w1488 SP 535-50 Nelio.TIF_cell.tif' % (direc, condition, embryo, embryo)))
            ch2 = cytoconc(
                loadimage('%s/%s/%s/_%s_w2488 SP 630-75 Nelio-AF.TIF_cell.tif' % (direc, condition, embryo, embryo)))
            signal = af_subtraction(ch1, ch2, m=m, c=c)

            row = pd.DataFrame([[condition, ch1, ch2, signal]],
                               columns=['condition', 'ch1', 'ch2', 'signal'])
            df = df.append(row)

    df.to_csv(name)


def fit_bgcurve(curve, bgcurve):
    # Fix ends
    line = np.polyfit(
        [np.nanmean(bgcurve[:int(len(bgcurve) * 0.2)]), np.nanmean(bgcurve[int(len(bgcurve) * 0.8):])],
        [np.nanmean(curve[:int(len(curve) * 0.2)]), np.nanmean(curve[int(len(curve) * 0.8):])], 1)
    bgcurve = line[0] * bgcurve + line[1]

    # Interpolate bgcurve
    bgcurve = np.interp(np.linspace(0, len(bgcurve), len(curve)), range(len(bgcurve)), bgcurve)

    return bgcurve


def gaussian(x, a, b, c):
    """
    Generic function for a gaussian curve

    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    """

    y = a * np.e ** (- ((x - b) ** 2) / (2 * (c ** 2)))
    return y


def fit_gaussian(signal):
    """
    Fits a gaussian curve to a baseline-subtracted signal curve

    :param signal:
    :return:
    """

    popt, pcov = curve_fit(gaussian, range(len(signal)), signal, bounds=([0, 40, 0], [np.inf, 60, np.inf]))
    gauss = gaussian(np.array(range(len(signal))), *popt)

    return gauss


def fit_coordinates_alg(data, settings, func, iterations):
    """
    Algorithm that takes confocal images and an initial manual outline and fits outline to the edge of the cell
    Returns coordinates of fitted outline

    :param data:
    :param settings:
    :param func:
    :param iterations:
    :return:
    """

    img = af_subtraction(data.GFP, data.AF, settings)
    coors = data.ROI_orig

    for i in range(iterations):
        straight = straighten(img, coors, 20)
        coors = offset_coordinates(coors, func(straight))
        coors = np.vstack(
            (savgol_filter(coors[:, 0], 19, 2, mode='wrap'), savgol_filter(coors[:, 1], 19, 2, mode='wrap'))).T

    # coors = rotate_coors(coors)
    return coors


def subtract_baseline(curve, baseline):
    """
    Fixes baseline curve to end of signal curve, and subtracts to return membrane signal curve

    :param curve:
    :param baseline:
    :return:
    """

    # Fix ends
    line = np.polyfit(
        [np.nanmean(baseline[:int(len(baseline) * 0.2)]), np.nanmean(baseline[int(len(baseline) * 0.8):])],
        [np.nanmean(curve[:int(len(curve) * 0.2)]), np.nanmean(curve[int(len(curve) * 0.8):])], 1)
    baseline = line[0] * baseline + line[1]

    # Interpolate baseline
    baseline = np.interp(np.linspace(0, len(baseline), len(curve)), range(len(baseline)), baseline)

    # Subtract baseline
    signal = curve - baseline

    return signal


def total_signal(data, settings):
    """
    Estimate of the total GFP level in the embryo.
    Expands cross section and takes average pixel intensity

    :param data:
    :param settings:
    :return:
    """

    corrected = af_subtraction2(data.GFP, data.AF, m=settings.m, c=settings.c, x=settings.x)
    img = polycrop(corrected, data.ROI_fitted, 10)
    cyt = np.mean(img[np.nonzero(img)])
    # cyt = float(np.size(img[np.nonzero(img)]))
    return cyt


def total_signal2(cyt, cort, ratio):
    return cyt + ratio * cort


def calc_offsets(img):
    """
    Calculates an offset value for each x, to align the peak to the centre

    :param img: straightened image (af corrected)
    :return:
    """
    img2 = interp(img, 1000)
    img3 = rolling_ave(img2, 20)
    img4 = savitsky_golay(img3, 251, 5)

    offsets = np.zeros(len(img[0, :]))

    for x in range(len(img[0, :])):
        offsets[x] = (len(img[:, 0]) / 2) - np.argmax(img4[:, x]) * (len(img[:, 0]) / 1000)

    return offsets


def af_subtraction(ch1, ch2, m, c):
    af = m * ch2 + c
    signal = ch1 - af
    return signal


def af_subtraction2(ch1, ch2, m, c, x):
    """
    AF subtraction that accounts for leak of GFP into AF channel

    :param ch1: GFP channel
    :param ch2: AF channel
    :param m: N2 relationship: y = mx + c (y=ch1, x=ch2)
    :param c: ""
    :param x: leakage of GFP into AF channel
    :return:
    """

    signal = (ch1 - m * ch2 - c) / (1 - m * x)
    return signal


def af_subtraction3(ch1, ch2, ch3, m, x, y):
    """

    :param ch1: 488,535-50
    :param ch2: 488,630-75
    :param ch3: 561,535-50
    :param m: N2 relationship, 488,535-50 / 488,630-75
    :param x: GFP emission bleed through (630-75 / 535-50)
    :param y: mCherry excitation bleed through (488 / 561) = 0.120219
    :return:
    """

    signal = (ch1 - m * ch2 + y * m * ch3) / (1 - x * m)
    return signal


def af_subtraction4(ch1, ch2, ch3, settings):
    signal = (ch1 - settings.m * ch2 + settings.y * settings.m * ch3 - settings.c) / (1 - settings.x * settings.m)
    return signal


class Results:
    def __init__(self):
        self.cyts = np.array([])
        self.corts = np.array([])
        self.totals = np.array([])

        self.cyts_norm = None
        self.corts_norm = None
        self.totals_norm = None

    def extend(self, entry):
        self.cyts = np.append(self.cyts, [entry.cyt], axis=0)
        self.corts = np.append(self.corts, [entry.cort], axis=0)
        self.totals = np.append(self.totals, [entry.total], axis=0)

    def normalise(self, factor):
        try:
            self.cyts_norm = self.cyts / np.mean(factor.cyts)
        except:
            pass

        try:
            self.corts_norm = self.corts / np.mean(factor.corts)
        except:
            pass

        try:
            self.totals_norm = self.totals / np.mean(factor.totals)
        except:
            pass


class Data:
    """
    Structure to hold all imported data for an embryo

    """

    def __init__(self, direc):

        # Directory
        self.direc = direc
        self.cond_direc = os.path.dirname(direc)

        # nd file
        try:
            self.nd = readnd(direc)
        except IndexError:
            self.nd = None

        # Conditions
        try:
            self.conds = read_conditions(direc)
        except:
            self.conds = None

        # EmbryoID
        self.emID = os.path.basename(direc)

        # DIC
        try:
            self.DIC = loadimage(sorted(glob.glob('%s/*DIC SP Camera*' % direc), key=len)[0])
        except IndexError:
            self.DIC = None

        # GFP
        try:
            self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 535-50*' % direc), key=len)[0])
        except IndexError:
            self.GFP = None

        # AF
        try:
            self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
        except IndexError:
            self.AF = None

        # RFP
        try:
            self.RFP = loadimage(sorted(glob.glob('%s/*561 SP 630-75*' % direc), key=len)[0])
        except IndexError:
            self.RFP = None

        # GFP straight
        try:
            self.GFP_straight = loadimage(glob.glob('%s/*488 SP 535-50*straight*' % direc)[0])
        except IndexError:
            self.GFP_straight = None

        # AF straight
        try:
            self.AF_straight = loadimage(glob.glob('%s/*488 SP 630-75*straight*' % direc)[0])
        except IndexError:
            self.AF_straight = None

        # RFP straight
        try:
            self.RFP_straight = loadimage(glob.glob('%s/*561 SP 630-75*straight*' % direc)[0])
        except IndexError:
            self.RFP_straight = None

        # ROI orig
        try:
            self.ROI_orig = np.loadtxt('%s/ROI_orig.txt' % direc)
        except FileNotFoundError:
            self.ROI_orig = None

        # ROI fitted
        try:
            self.ROI_fitted = np.loadtxt('%s/ROI_fitted.txt' % direc)
        except FileNotFoundError:
            try:
                self.ROI_fitted = np.loadtxt('%s/ROI_orig.txt' % direc)
            except FileNotFoundError:
                self.ROI_fitted = None

        # Surface area / Volume
        try:
            [self.sa, self.vol] = geometry(self.ROI_fitted)
        except:
            self.sa = None
            self.vol = None

        # Res
        try:
            self.res = pklload(direc)
        except:
            self.res = None


def slider(img):
    """
    Plots profiles across y dimension of image for user controlled x values

    :param img:
    :return:
    """

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
    sframe = Slider(axframe, '', 0, len(img[0, :]), valinit=0, valfmt='%d')

    def update_slider(val):
        ax.clear()
        x = int(sframe.val)
        ax.plot(img[:, x])
        ax.set_ylim([0, img.max()])

    sframe.on_changed(update_slider)
    plt.show()


def calc_offsets2(img):
    """
    Calculates an offset value for each x (where profile crosses midpoint?)

    :param img: straightened image
    :return:
    """
    img2 = interp(img, 1000)
    img3 = rolling_ave(img2, 50)
    img4 = savitsky_golay(img3, 251, 5)

    offsets = np.zeros(len(img[0, :]))

    for x in range(len(img[0, :])):
        offsets[x] = (len(img[:, 0]) / 2) - np.argmin(
            np.absolute(img4[:, x] - np.mean([np.mean(img4[900:, x]), np.mean(img4[:100, x])]))) * (
                                                len(img[:, 0]) / 1000)

    return offsets


def fit_coordinates_alg2(img, coors, func, iterations):
    """

    :param img:
    :param coors:
    :param func:
    :param iterations:
    :return:

    np.savetxt('%s/ROI_fitted.txt' % data.direc, newcoors, fmt='%.4f', delimiter='\t')

    """

    for i in range(iterations):
        straight = straighten(img, coors, 20)
        coors = offset_coordinates(coors, func(straight))
        coors = np.vstack(
            (savgol_filter(coors[:, 0], 19, 2, mode='wrap'), savgol_filter(coors[:, 1], 19, 2, mode='wrap'))).T

    coors = rotate_coors(coors)
    return coors
