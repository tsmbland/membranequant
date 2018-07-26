from IA import *
import BgCurves as b
import AFSettings as s

"""
PAR-2 rundown experiment (Feb-March 2018)

Not suitable for computational segmentation (no guide in strong RNAi), all segmented manually
PFS in for all embryos so not directly comparable with other experiments

"""

#####################################################################################

# INPUT DATA

conds_list_total = [
    '180309/180309_nwg0123_par3_tom3,15,pfsin',
    '180309/180309_nwg0123_par3,0945par2_tom3,15,pfsin',
    '180309/180309_nwg0123_par3,1100par2_tom3,15,pfsin',
    '180309/180309_nwg0123_par3,1200par2_tom3,15,pfsin',
    '180309/180309_nwg0123_par3,1300par2_tom3,15,pfsin',
    '180309/180309_nwg0123_par3,1400par2_tom,15,pfsin',
    '180223/180223_nwg0123_24hr0par2,par3_tom3,15,pfsin',
    '180223/180223_nwg0123_24hr10par2,par3_tom3,15,pfsin',
    '180223/180223_nwg0123_24hr50par2,par3_tom3,15,pfsin',
    '180223/180223_nwg0123_24hr100par2,par3_tom3,15,pfsin',
    '180223/180223_kk1273_wt_tom3,15,pfsin']

embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s1
bgcurve = b.bgG4
d = Data



#####################################################################################

# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.9, 0.1))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# GFP SPATIAL QUANTIFICATION

def func4(embryo):
    data = d(embryo)
    sigs = spatial_signal_GFP(data, bgcurve, settings)
    pklsave(data.direc, sigs, 'res1_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func4)(embryo) for embryo in embryos_list_total)


# CROSS SECTION GFP

def func6(embryo):
    data = d(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    sec = cross_section(img=img, coors=data.ROI_fitted, thickness=50, extend=1.2)
    pklsave(data.direc, sec, 'res1_csection')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func6)(embryo) for embryo in embryos_list_total)


#####################################################################################

# LOAD DATA

nwg0123_wt = Results(np.array(conds_list_total)[[0, 6]])
nwg0123_rd = Results(np.array(conds_list_total)[[1, 2, 3, 4, 5, 7, 8, 9]])
kk1273_wt = Results(np.array(conds_list_total)[[10]])


#####################################################################################


# CHECK SEGMENTATION (MANUAL) <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#
#     plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50), cmap='gray')
#     plt.show()


