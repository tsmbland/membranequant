from IA import *
import BgCurves as b
import AFSettings as s


"""
PAR-2 rundown, looking at PAR-1 and PAR-2 to assess recruitment (March 2018)


"""


# INPUT DATA

conds_list_total = [
    '180322/180322_nwg0132_par3_tom4,15,30,pfsout',
    '180322/180322_nwg0132_par3,0945par2_tom4,15,30,pfsout',
    '180322/180322_nwg0132_par3,1115par2_tom4,15,30,pfsout',
    '180322/180322_nwg0132_par3,1255par2_tom4,15,30,pfsout',
    '180316/180316_nwg42_wt_tom4,15,30,pfsout',
    '180322/180322_nwg42_wt_tom4,15,30,pfsout',
    '180501/180501_nwg0132_1320par2_tom4,15,30',
    '180501/180501_nwg0132_1630par2_tom4,15,30']

embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s2
bgcurve = b.bgG4
d = Data



# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    coors = fit_coordinates_alg3(img, data.ROI_orig, bgcurve, 2)
    np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)


# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0.9, 0.2))
    cyt = cytoplasmic_signal_GFP(data, settings)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1))
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def func3(embryo):
    data = d(embryo)
    sig = cortical_signal_RFP(data, bgcurve, bounds=(0.9, 0.2))
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


################# Load data

nwg42_wt = Results(np.array(conds_list_total)[[4, 5]])
nwg0132_wt = Results(np.array(conds_list_total)[[0]])
nwg0132_rd = Results(np.array(conds_list_total)[[1, 2, 3, 6, 7]])


############ Checking

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


# CHECK RFP BG <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#     plt.imshow(straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50), 50))
#     plt.show()
