from IA import *
import BgCurves as b
import AFSettings as s

"""
PH-rundown control experiment (April 2018)


"""

#################################################################

# INPUT DATA

conds_list_total = [
    '180420/180420_nwg1_0920xfp_tom4,5,30',
    '180420/180420_nwg1_1100xfp_tom4,5,30',
    '180420/180420_nwg1_1300xfp_tom4,5,30',
    '180420/180420_nwg1_1600(180419)xfp_tom4,5,30',
    '180420/180420_nwg1_wt_tom4,5,30']

embryos_list_total = embryos_direcslist(conds_list_total)

bgcurve = b.bgG2
settings = s.OD70s1
d = Data


#################################################################

# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    coors = fit_coordinates_alg3(data.RFP, data.ROI_orig, bgcurve, 2)
    np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)


#################################################################

# GFP QUANTIFICATION

def func2(embryo):
    data = Data(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.9, 0.1))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def func3(embryo):
    data = Data(embryo)
    sig = cortical_signal_RFP(data, bgcurve, bounds=(0, 1))
    cyt = cytoplasmic_signal_RFP(data)
    total = cyt + (data.sa / data.vol) * cortical_signal_RFP(data, bgcurve, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res2')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func3)(embryo) for embryo in embryos_list_total)


# GFP SPATIAL QUANTIFICATION

def func4(embryo):
    data = Data(embryo)
    sigs = spatial_signal_GFP(data, bgcurve, settings)
    pklsave(data.direc, sigs, 'res1_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func4)(embryo) for embryo in embryos_list_total)


# RFP SPATIAL QUANTIFICATION

def func5(embryo):
    data = Data(embryo)
    sigs = spatial_signal_RFP(data, bgcurve)
    pklsave(data.direc, sigs, 'res2_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func5)(embryo) for embryo in embryos_list_total)


# CROSS SECTION GFP

def func6(embryo):
    data = Data(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    sec = cross_section(img=img, coors=data.ROI_fitted, thickness=10, extend=1.5)
    pklsave(data.direc, sec, 'res1_csection')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func6)(embryo) for embryo in embryos_list_total)


# CROSS SECTION RFP

def func7(embryo):
    data = Data(embryo)
    img = data.RFP
    mag = 1
    bg = straighten(img, offset_coordinates(data.ROI_fitted, 50 * mag), 50 * mag)
    sec = cross_section(img=img, coors=data.ROI_fitted, thickness=10, extend=1.5) - np.nanmean(bg[np.nonzero(bg)])
    pklsave(data.direc, sec, 'res2_csection')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func7)(embryo) for embryo in embryos_list_total)

#################################################################

# Load data


ph_rd = Results(np.array(conds_list_total)[[0, 1, 3]])
ph_wt = Results(np.array(conds_list_total)[[2, 4]])  # <- fudging this a bit, 2 is not strictly wt (but only embryos 6 and 8 appear affected by RNAi)


#################################################################

# Check segmentation <- ok but not great, I wonder why...

# for embryo in embryos_list_total:
#     data = d(embryo)
#
#     print(data.direc)
#
#     # plt.imshow(data.RFP)
#     # plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     # plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     # plt.show()
#
#     plt.imshow(straighten(data.RFP, data.ROI_fitted, 50))
#     plt.show()


# CHECK RFP BG <- considerable glow from the cell visible

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#     plt.imshow(straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50), 50))
#     plt.show()


# # Check bg fitting <- looks a bit off
# for embryo in embryos_list_total:
#     data = Data(embryo)
#
#     print(data.direc)
#
#     img = af_subtraction(data.GFP, data.AF, settings=s.N2s2)
#     img_straight = straighten(img, data.ROI_fitted, 50)
#     profile = np.nanmean(np.hstack(
#         (img_straight[:, :int(len(img_straight[0, :]) * 0.1)], img_straight[:, int(len(img_straight[0, :]) * 0.9):])),
#         1)
#     a, bg = fit_background_v2_2(profile, b.bgG4[25:75])
#     signal = profile - bg
#     plt.plot(profile)
#     plt.plot(bg)
#     plt.plot(gaussian_plus2(b.bgG4[25:75], *a))
#     plt.show()
