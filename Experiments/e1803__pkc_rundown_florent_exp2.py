from IA import *
import BgCurves as b
import AFSettings as s

"""
Example script to be used as a template for analysis of other experiments

Things that need to be personalised for the particular experiment:
- list of directories containing embryo folders
- af settings
- background curve(s)
- method of segmentation (e.g. which channel is being used as the guide)
- which quantifications are suitable
- bounds for averaging cortical fluorescence over
- magnification of images


"""

#####################################################################################


# INPUT DATA

conds_list_total = ['PKC_rundown_Florent_MP/180221/180221_NWG91_0PKC',
                    'PKC_rundown_Florent_MP/180221/180221_NWG91_25PKC',
                    'PKC_rundown_Florent_MP/180221/180221_NWG91_50PKC',
                    'PKC_rundown_Florent_MP/180221/180221_NWG91_100PKC',
                    'PKC_rundown_Florent_MP/180221/180221_NWG93_0PKC',
                    'PKC_rundown_Florent_MP/180221/180221_NWG93_25PKC',
                    'PKC_rundown_Florent_MP/180221/180221_NWG93_50PKC',
                    'PKC_rundown_Florent_MP/180221/180221_NWG93_100PKC']

embryos_list_total = embryos_direcslist(conds_list_total)

settings = s.N2s8
bgcurve = b.bgG4
d = Data2


#####################################################################################


# SEGMENTATION

def func1(embryo):
    data = d(embryo)
    try:
        coors = fit_coordinates_alg3(composite(data, settings, 0.3), data.ROI_orig, bgcurve, 2, mag=5/3)
        np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')
    except np.linalg.linalg.LinAlgError:
        print(data.direc)


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func1)(embryo) for embryo in embryos_list_total)


# GFP QUANTIFICATION

def func2(embryo):
    data = d(embryo)
    sig = cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1), mag=5/3)
    cyt = cytoplasmic_signal_GFP(data, settings, mag=5/3)
    total = cyt + (data.sa / data.vol) * cortical_signal_GFP(data, bgcurve, settings, bounds=(0, 1), mag=5/3)
    pklsave(data.direc, Res(cyt, sig, total), 'res1')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func2)(embryo) for embryo in embryos_list_total)


# RFP QUANTIFICATION

def func3(embryo):
    data = d(embryo)
    sig = cortical_signal_RFP(data, bgcurve, bounds=(0, 1), mag=5/3)
    cyt = cytoplasmic_signal_RFP(data, mag=5/3)
    total = cyt + (data.sa / data.vol) * cortical_signal_RFP(data, bgcurve, bounds=(0, 1), mag=5/3)
    pklsave(data.direc, Res(cyt, sig, total), 'res2')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func3)(embryo) for embryo in embryos_list_total)


# GFP SPATIAL QUANTIFICATION

def func4(embryo):
    data = d(embryo)
    sigs = spatial_signal_GFP(data, bgcurve, settings, mag=5/3)
    pklsave(data.direc, sigs, 'res1_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func4)(embryo) for embryo in embryos_list_total)


# RFP SPATIAL QUANTIFICATION

def func5(embryo):
    data = d(embryo)
    sigs = spatial_signal_RFP(data, bgcurve, mag=5/3)
    pklsave(data.direc, sigs, 'res2_spatial')


# Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(delayed(func5)(embryo) for embryo in embryos_list_total)




#####################################################################################


# LOAD DATA

nwg91_wt = Results(np.array(conds_list_total)[[0]])
nwg91_rd = Results(np.array(conds_list_total)[[1, 2, 3]])
nwg93_wt = Results(np.array(conds_list_total)[[4]])
nwg93_rd = Results(np.array(conds_list_total)[[5, 6, 7]])



#####################################################################################

# CHECK SEGMENTATION <- good

# for embryo in embryos_list_total:
#     data = d(embryo)
#     print(data.direc)
#
#     # plt.imshow(af_subtraction(data.GFP, data.AF, s.N2s2), cmap='gray')
#     # plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     # plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     # plt.show()
#     #
#     # plt.imshow(data.RFP, cmap='gray')
#     # plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
#     # plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
#     # plt.show()
#     #
#     # plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50), cmap='gray')
#     # plt.show()
#
#     plt.imshow(straighten(data.RFP, data.ROI_fitted, 50), cmap='gray')
#     plt.show()
