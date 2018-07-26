from IA import *
import BgCurves as b
import AFSettings as s

"""
Comparision of cytoplasmic affinity of different PAR-2 mutants in PKC+/- conditions (June 2018)
On SD after move to new room

"""

#####################################################################################

# INPUT DATA

conds_list_total = [
    '180622/180622_jh1799_48hrpkc_tom3,15',
    '180622/180622_jh1799_wt_tom3,15',
    '180622/180622_jh2817_48hrpkc_tom3,15',
    '180622/180622_jh2817_wt_tom3,15',
    '180622/180622_kk1273_48hrpkc_tom3,15',
    '180622/180622_kk1273_wt_tom3,15',
    '180622/180622_th129_48hrpkc_tom3,15',
    '180622/180622_th129_wt_tom3,15']

embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s2
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


# QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.9, 0.1))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# SPATIAL QUANTIFICATION

def func3(embryo):
    data = d(embryo)
    sigs = spatial_signal_GFP(data, bgcurve, settings)
    pklsave(data.direc, sigs, 'res1_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func3)(embryo) for embryo in embryos_list_total)


# CROSS SECTION GFP

def func6(embryo):
    data = d(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    sec = cross_section(img=img, coors=data.ROI_fitted, thickness=10, extend=1.5)
    pklsave(data.direc, sec, 'res1_csection')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func6)(embryo) for embryo in embryos_list_total)

#####################################################################################

# Load data

kk1273_wt = Results(np.array(conds_list_total)[[5]])
kk1273_pkc = Results(np.array(conds_list_total)[[4]])

jh1799_wt = Results(np.array(conds_list_total)[[1]])
jh1799_pkc = Results(np.array(conds_list_total)[[0]])

jh2817_wt = Results(np.array(conds_list_total)[[3]])
jh2817_pkc = Results(np.array(conds_list_total)[[2]])

th129_wt = Results(np.array(conds_list_total)[[7]])
th129_pkc = Results(np.array(conds_list_total)[[6]])

#####################################################################################

# Check segmentation <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#
#     print(data.direc)
#
#     # plt.imshow(af_subtraction(data.GFP, data.AF, s.N2s2))
#     # plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     # plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     # plt.show()
#
#     plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50))
#     plt.show()


# # Check bg fitting
# for embryo in embryos_list_total:
#     data = Experiments(embryo)
#
#     print(data.direc)
#
#     img = af_subtraction5(data.GFP, data.AF, settings=s.N2s2)
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
