import BgCurves as b
import AFSettings as s
from IA import *

"""
Comparision of cytoplasmic affinity of different PAR-2 mutants in PKC+/- conditions (March-June 2018)


"""

# Done, checked

#####################################################################################

# INPUT DATA

conds_list_total = [
    '180302/180302_nwg0123_24hr0par2,par3_tom3,15,pfsout',
    '180322/180322_nwg0123_par3_tom4,15,30,pfsout',
    '180501/180501_kk1273_wt_tom3,15+ph',
    '180509/180509_nwg0062_wt_tom3,15,pfsout',
    '180525/180525_jh1799_wt_tom3,15',
    '180525/180525_kk1273_wt_tom3,15',
    '180606/180606_jh1799_48hrpar6_tom3,15',
    '180606/180606_jh1799_wt_tom3,15',
    '180606/180606_jh2882_wt_tom3,15',
    '180611/180611_jh1799_48hrctrlrnai_tom3,15',
    '180611/180611_jh1799_wt_tom3,15',
    '180618/180618_jh2817_48hrpar6_tom3,15',
    '180618/180618_jh2817_wt_tom3,15',
    '180618/180618_kk1273_48hrpar6_tom3,15',
    '180618/180618_nwg62_48hrpar6_tom3,15',
    '180618/180618_th129_48hrpar6_tom3,15',
    '180618/180618_th129_wt_tom3,15',
    '180606/180606_jh2882_48hrpar6_tom3,15',
    '180804/180804_nwg0123_wt_tom4,15,30+bleach']


embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s2
bgcurve = b.bgG4
d = Data

# embryos_list_total = embryos_direcslist(['180804/180804_nwg0123_wt_tom4,15,30+bleach'])


#####################################################################################


# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    coors = fit_coordinates_alg3(img, data.ROI_orig, bgcurve, 2)
    np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)


#####################################################################################


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

kk1273_wt = Results(np.array(conds_list_total)[[2, 5]])
kk1273_par6 = Results(np.array(conds_list_total)[[13]])
nwg0123_wt = Results(np.array(conds_list_total)[[0, 1]])
nwg0062_wt = Results(np.array(conds_list_total)[[3]])
nwg0062_par6 = Results(np.array(conds_list_total)[[14]])
jh1799_wt = Results(np.array(conds_list_total)[[4, 7, 10]])
jh1799_par6 = Results(np.array(conds_list_total)[[6]])
jh1799_ctrl = Results(np.array(conds_list_total)[[9]])
jh2882_wt = Results(np.array(conds_list_total)[[8]])
jh2882_par6 = Results(np.array(conds_list_total)[[17]])
jh2817_wt = Results(np.array(conds_list_total)[[12]])
jh2817_par6 = Results(np.array(conds_list_total)[[11]])
th129_wt = Results(np.array(conds_list_total)[[16]])
th129_par6 = Results(np.array(conds_list_total)[[15]])
nwg123_wt_bleach = Results(np.array(conds_list_total)[[18]])



#####################################################################################

# Check segmentation <- good
# for embryo in embryos_list_total:
#     data = d(embryo)
#
#     print(data.direc)
#
#     plt.imshow(af_subtraction(data.GFP, data.AF, s.N2s2))
#     plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     plt.show()
#
#     # plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50))
#     # plt.show()


# Check bg fitting <- good
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
