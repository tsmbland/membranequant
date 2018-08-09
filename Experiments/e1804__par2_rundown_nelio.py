import BgCurves as b
import AFSettings as s
from IA import *

"""
Nelio's PAR-2 rundown data in NWG76 (March-April 2018)

Manually segmented by Nelio

"""

# Done, checked

#####################################################################################


# INPUT DATA

conds_list_total = [
    'PAR2_Nelio/NR020418_P2rundown_NWG0076',
    'PAR2_Nelio/NR020418_P2rundown_NWG0076_WT',
    'PAR2_Nelio/NR030418 - P2 rundown - NWG0076',
    'PAR2_Nelio/NR030418 - P2 rundown - NWG0076_WT',
    'PAR2_Nelio/NR040418 - P2 rundown - NWG0076',
    'PAR2_Nelio/NR040418 - P2 rundown - NWG0076_WT',
    'PAR2_Nelio/NR290318 - P2 rundown - NWG0076',
    'PAR2_Nelio/NR290318 - P2 rundown - NWG0076_WT',
    'PAR2_Nelio/NR300318 - P2 rundown - NWG0076',
    'PAR2_Nelio/NR300318 - P2 rundown - NWG0076_WT']

embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s4
bgcurve = b.bgG4
d = Data


#####################################################################################

# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    try:
        coors = fit_coordinates_alg3(composite(data, settings, 2), data.ROI_orig, bgcurve, 2, mag=2)
        np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')
    except np.linalg.linalg.LinAlgError:
        print(data.direc)


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)


#####################################################################################

# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.9, 0.1), mag=2)
    cyt = cytoplasmic_signal_GFP(data, settings, mag=2)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1), mag=2)
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def func3(embryo):
    data = d(embryo)
    sig = cortical_signal_RFP(data, bgcurve, bounds=(0.9, 0.1), mag=2)
    cyt = cytoplasmic_signal_RFP(data, mag=2)
    total = cyt + (data.sa / data.vol) * cortical_signal_RFP(data, bgcurve, bounds=(0, 1), mag=2)
    pklsave(data.direc, Res(cyt, sig, total), 'res2')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func3)(embryo) for embryo in embryos_list_total)


# GFP SPATIAL QUANTIFICATION

def func4(embryo):
    data = d(embryo)
    sigs = spatial_signal_GFP(data, bgcurve, settings, mag=2)
    pklsave(data.direc, sigs, 'res1_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func4)(embryo) for embryo in embryos_list_total)


# RFP SPATIAL QUANTIFICATION

def func5(embryo):
    data = d(embryo)
    sigs = spatial_signal_RFP(data, bgcurve, mag=2)
    pklsave(data.direc, sigs, 'res2_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func5)(embryo) for embryo in embryos_list_total)

#####################################################################################

# LOAD DATA

nwg76_wt = Results(np.array(conds_list_total)[[1, 3, 5, 7, 9]])
nwg76_rd = Results(np.array(conds_list_total)[[0, 2, 4, 6, 9]])

#####################################################################################

# CHECK SEGMENTATION <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#
#     plt.imshow(data.RFP, cmap='gray')
#     plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     plt.show()
#
#     # plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 100), cmap='gray')
#     # plt.show()
#     #
#     # plt.imshow(straighten(data.RFP, data.ROI_fitted, 100), cmap='gray')
#     # plt.show()


# CHECK RFP BG <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#
#     plt.imshow(straighten(data.RFP, offset_coordinates(data.ROI_fitted, 100), 100))
#     plt.show()


# Check bg fitting

# for embryo in embryos_list_total:
#     data = Data(embryo)
#
#     mag = 2
#     bounds = (0.9, 0.1)
#
#     # Correct autofluorescence
#     img = af_subtraction(data.GFP, data.AF, settings=settings)
#
#     # Straighten
#     img = straighten(img, data.ROI_fitted, int(50 * mag))
#
#     # Average
#     if bounds[0] < bounds[1]:
#         profile = np.nanmean(img[:, int(len(img[0, :]) * bounds[0]): int(len(img[0, :]) * bounds[1] + 1)], 1)
#     else:
#         profile = np.nanmean(
#             np.hstack((img[:, :int(len(img[0, :]) * bounds[1] + 1)], img[:, int(len(img[0, :]) * bounds[0]):])), 1)
#
#     # Adjust for magnification (e.g. if 2x multiplier is used)
#     profile = np.interp(np.linspace(0, len(profile), 50), range(len(profile)), profile)
#
#     # Get cortical signal
#     a, bg = fit_background_v2_2(profile, bgcurve[25:75])
#     signal = profile - bg
#     plt.plot(profile)
#     plt.plot(bg)
#     plt.plot(gaussian_plus2(b.bgG4[25:75], *a))
#     plt.show()
