import BgCurves as b
import AFSettings as s
from IA import *

"""
Looking at nwg0158 line, aiming to understand if PAR-2 homodimerises


"""

#####################################################################################


# INPUT DATA

conds_list_total = ['180803/180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e1',
                    '180803/180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e2',
                    '180803/180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e3',
                    '180803/180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e4',
                    '180803/180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e5',
                    '180803/MP',
                    '180726/180726_nwg0158_wt_tom4,15,30']

conds_list_2 = ['180730/180730_nwg0151_wt_tom4,15,30',
                '180726/180726_nwg0151_wt_tom4,15,30']

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


#####################################################################################


# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.9, 0.1))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def func3(embryo):
    data = d(embryo)
    sig = cortical_signal_RFP(data, bgcurve, bounds=(0.9, 0.1))
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


# CROSS SECTION GFP

def func6(embryo):
    data = d(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    sec = cross_section(img=img, coors=data.ROI_fitted, thickness=50, extend=1.2)
    pklsave(data.direc, sec, 'res1_csection')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func6)(embryo) for embryo in embryos_list_total)


# CROSS SECTION RFP

def func7(embryo):
    data = d(embryo)
    img = data.RFP
    mag = 1
    bg = straighten(img, offset_coordinates(data.ROI_fitted, 50 * mag), 50 * mag)
    sec = cross_section(img=img, coors=data.ROI_fitted, thickness=50, extend=1.2) - np.nanmean(bg[np.nonzero(bg)])
    pklsave(data.direc, sec, 'res2_csection')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func7)(embryo) for embryo in embryos_list_total)

#####################################################################################


# LOAD DATA
e1 = Results(np.array(conds_list_total)[[0]])
e2 = Results(np.array(conds_list_total)[[1]])
e3 = Results(np.array(conds_list_total)[[2]])
e4 = Results(np.array(conds_list_total)[[3]])
e5 = Results(np.array(conds_list_total)[[4]])
nwg158_pma_mp = Results(np.array(conds_list_total)[[5]])
nwg158_mp = Results(np.array(conds_list_total)[[6]])
nwg151_wt = Results(np.array(conds_list_2)[[0, 1]])



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
