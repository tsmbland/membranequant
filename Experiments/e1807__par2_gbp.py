import BgCurves as b
import AFSettings as s
from IA import *

"""
PAR-2 GBP

"""

# Done, checked


#####################################################################################


# INPUT DATA

conds_list_total = ['180730/180730_nwg0151_wt_tom4,15,30',
                    '180730/180730_nwg0151_wt_tom4,15,30+bleach',
                    '180730/180730_nwg0151xnwg0143_tom4,15,30',
                    '180730/180730_nwg0151xnwg0143_tom4,15,30+bleach',
                    '180726/180726_nwg0151_wt_tom4,15,30',
                    '180726/180726_nwg0151_wt_tom4,15,30+bleach',
                    '180804/180804_nwg0151dr466_wt_tom4,15,30',
                    '180804/180804_nwg0151dr466_wt_tom4,15,30+bleach']

embryos_list_total = embryos_direcslist(conds_list_total)
# embryos_list_total = embryos_direcslist(['180804/180804_nwg0151dr466_wt_tom4,15,30+bleach'])


settings = s.N2s10
bgcurve = b.bgG4
d = Data


#####################################################################################


# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    coors = fit_coordinates_alg3(img, data.ROI_orig, bgcurve, 2)
    np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)


#####################################################################################


# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.8, 0.2))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def cytoplasmic_signal_RFP(data, mag=1):
    img = data.RFP

    # Subtract background
    bg = straighten(img, offset_coordinates(data.ROI_fitted, 50 * mag), int(50 * mag))
    mean1 = np.nanmean(bg[np.nonzero(bg)])
    af = settings.m2 * mean1 + settings.c2

    # Get cytoplasmic signal
    img2 = polycrop(img, data.ROI_fitted, -20 * mag)
    mean2 = np.nanmean(img2[np.nonzero(img2)])  # mean, excluding zeros

    return mean2 - af


def func3(embryo):
    data = d(embryo)
    sig = cortical_signal_RFP(data, b.bgC2, bounds=(0.8, 0.2))
    cyt = cytoplasmic_signal_RFP(data)
    total = cyt + (data.sa / data.vol) * cortical_signal_RFP(data, b.bgC2, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res2')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func3)(embryo) for embryo in embryos_list_total)


# GFP SPATIAL QUANTIFICATION

def func4(embryo):
    data = d(embryo)
    sigs = spatial_signal_GFP(data, bgcurve, settings)
    pklsave(data.direc, sigs, 'res1_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func4)(embryo) for embryo in embryos_list_total)


# RFP SPATIAL QUANTIFICATION

def func5(embryo):
    data = d(embryo)
    sigs = spatial_signal_RFP(data, b.bgC2)
    pklsave(data.direc, sigs, 'res2_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func5)(embryo) for embryo in embryos_list_total)

#####################################################################################


# LOAD DATA

nwg51_gbp = Results(np.array(conds_list_total)[[2]])
nwg51_gbp_bleach = Results(np.array(conds_list_total)[[3]])
nwg51_wt = Results(np.array(conds_list_total)[[0, 4]])
nwg51_wt_bleach = Results(np.array(conds_list_total)[[1, 5]])
nwg51_dr466 = Results(np.array(conds_list_total)[[6]])
nwg51_dr466_bleach = Results(np.array(conds_list_total)[[7]])



#####################################################################################

# CHECK SEGMENTATION

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#
#     plt.imshow(af_subtraction(data.GFP, data.AF, s.N2s2), cmap='gray')
#     plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     plt.show()
#
#     plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50), cmap='gray')
#     plt.show()
#
#     plt.imshow(straighten(data.RFP, data.ROI_fitted, 50), cmap='gray')
#     plt.show()
