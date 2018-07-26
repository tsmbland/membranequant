from IA import *
import BgCurves as b
import AFSettings as s

"""


"""

#####################################################################################


# INPUT DATA

conds_list_total = direcslist('180712/180712_sv2061_wt_tom4,15,30,100msecG')

embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s0
bgcurve = b.bgG4
d = Data


#####################################################################################


# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    coors = fit_coordinates_alg3(data.RFP, data.ROI_orig, bgcurve, 2)
    np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)


# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def func3(embryo):
    data = d(embryo)
    sig = cortical_signal_RFP(data, bgcurve, bounds=(0, 1))
    cyt = cytoplasmic_signal_RFP(data)
    total = cyt + (data.sa / data.vol) * cortical_signal_RFP(data, bgcurve, bounds=(0, 1))
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
    sigs = spatial_signal_RFP(data, bgcurve)
    pklsave(data.direc, sigs, 'res2_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func5)(embryo) for embryo in embryos_list_total)


#####################################################################################


# LOAD DATA

e2 = Results(np.array(conds_list_total)[[0]])
e3 = Results(np.array(conds_list_total)[[1]])
e4 = Results(np.array(conds_list_total)[[2]])
e5 = Results(np.array(conds_list_total)[[3]])
e6 = Results(np.array(conds_list_total)[[4]])


#####################################################################################

# CHECK SEGMENTATION <- mostly good but not perfect. Also clearly visible polar body in most cells

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#
#     # plt.imshow(af_subtraction(data.GFP, data.AF, s.N2s2), cmap='gray')
#     # plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     # plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     # plt.show()
#
#     plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50), cmap='gray')
#     plt.show()
#
#     # plt.imshow(straighten(data.RFP, data.ROI_fitted, 50), cmap='gray')
#     # plt.show()


# CHECK RFP BG <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#     plt.imshow(straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50), 50))
#     plt.show()
