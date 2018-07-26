from IA import *
import AFSettings as s
import BgCurves as b

"""
Nelio's PAR-6 rundown data with NWG26 line (Jan-Feb 2018)

Manually segmented by Nelio


"""

#####################################################################################


# INPUT DATA

conds_list_total = [
    'PAR6_Nelio/NR150218 - P6 rundown on NWG0026 - AM',
    'PAR6_Nelio/NR150218 - P6 rundown on NWG0026 - AM_WT',
    'PAR6_Nelio/NR150218 - P6 rundown on NWG0026 - PM',
    'PAR6_Nelio/NR270118 - NWG0026 rundown',
    'PAR6_Nelio/NR270118 - NWG0026 rundown_WT',
    'PAR6_Nelio/NR280118 - NWG0026 rundown',
    'PAR6_Nelio/NR280118 - NWG0026 rundown_WT',
    'PAR6_Nelio/NR290118 - NWG0026 rundown',
    'PAR6_Nelio/NR290118 - NWG0026 rundown_WT']

embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s5
bgcurve = b.bgG4
d = Data


#####################################################################################


# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    try:
        coors = fit_coordinates_alg3(composite(data, settings, 3), data.ROI_orig, bgcurve, 2)
        np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')
    except np.linalg.linalg.LinAlgError:
        print(data.direc)


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)



#####################################################################################


# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.4, 0.6))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def func3(embryo):
    data = d(embryo)
    sig = cortical_signal_RFP(data, bgcurve, bounds=(0.4, 0.6))
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

nwg26_wt = Results(np.array(conds_list_total)[[1, 4, 6, 8]])
nwg26_rd = Results(np.array(conds_list_total)[[0, 2, 3, 5, 7]])

#####################################################################################

# CHECK SEGMENTATION (MANUAL) <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#
#     # # plt.imshow(af_subtraction(data.GFP, data.AF, s.N2s2), cmap='gray')
#     # plt.imshow(data.RFP, cmap='gray')
#     # plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     # plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     # plt.show()
#
#     # plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50), cmap='gray')
#     # plt.show()
#
#     # plt.imshow(straighten(data.RFP, data.ROI_fitted, 50), cmap='gray')
#     # plt.show()


# CHECK RFP BG <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#     plt.imshow(straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50), 50))
#     plt.show()
