import BgCurves as b
import AFSettings as s
from IA import *

############################# INPUT / SETTINGS ################################

# Input data
conds_list_total = [
    '180302_nwg0123_24hr0par2,par3_tom3,15,pfsout',
    '180322_nwg0123_par3_tom4,15,30,pfsout',
    '180501_kk1273_wt_tom3,15+ph',
    '180509_nwg0062_wt_tom3,15,pfsout',
    '180525_jh1799_wt_tom3,15',
    '180525_kk1273_wt_tom3,15',
    '180606_jh1799_48hrpar6_tom3,15',
    '180606_jh1799_wt_tom3,15',
    '180606_jh2882_wt_tom3,15',
    '180611_jh1799_48hrctrlrnai_tom3,15',
    '180611_jh1799_wt_tom3,15',
    '180618_jh2817_48hrpar6_tom3,15',
    '180618_jh2817_wt_tom3,15',
    '180618_kk1273_48hrpar6_tom3,15',
    '180618_nwg62_48hrpar6_tom3,15',
    '180618_th129_48hrpar6_tom3,15',
    '180618_th129_wt_tom3,15',
    '180606_jh2882_48hrpar6_tom3,15',
    '180804_nwg0123_wt_tom4,15,30+bleach']

# Global variables
settings = s.N2s2
bgcurve = b.bgG4
adirec = '../Analysis/%s' % os.path.basename(__file__)[:-3]
mag = 1


################################ DATA IMPORT #################################

class Data:
    def __init__(self, direc):
        self.direc = direc
        self.DIC = loadimage(sorted(glob.glob('%s/*DIC SP Camera*' % direc), key=len)[0])
        self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 535-50*' % direc), key=len)[0])
        self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
        self.ROI = np.loadtxt('%s/ROI.txt' % direc)


############################### SEGMENTATION #################################


def segment(direc):
    try:
        data = Data(direc)
        img = af_subtraction(data.GFP, data.AF, settings=settings)
        coors = fit_coordinates_alg(img, data.ROI, bgcurve, 2, mag=mag)
        np.savetxt('%s/ROI_fitted.txt' % direc, coors, fmt='%.4f', delimiter='\t')
    except np.linalg.linalg.LinAlgError:
        print(direc)


################################ ANALYSIS ####################################


class Res:
    def __init__(self):
        self.g_mem = []
        self.g_cyt = []
        self.g_tot = []
        self.g_spa = []
        self.g_cse = []


class Analysis:
    def __init__(self):
        self.res = Res()

    def g_mem(self, data, coors):
        self.res.g_mem = [cortical_signal_g(data, coors, bgcurve, settings, bounds=(0.9, 0.1), mag=mag)]

    def g_cyt(self, data, coors):
        self.res.g_cyt = [cytoplasmic_signal_g(data, coors, settings, mag=mag)]

    def g_tot(self, data, coors):
        cyt = cytoplasmic_signal_g(data, coors, settings, mag=mag)
        self.res.g_tot = [cyt + (geometry(coors)[0] / geometry(coors)[1]) * cortical_signal_g(data, coors, bgcurve,
                                                                                              settings, bounds=(0, 1),
                                                                                              mag=mag)]

    def g_spa(self, data, coors):
        self.res.g_spa = spatial_signal_g(data, coors, bgcurve, settings, mag=mag)

    def g_cse(self, data, coors):
        self.res.g_cse = cross_section(img=af_subtraction(data.GFP, data.AF, settings=settings), coors=coors,
                                       thickness=10, extend=1.5)


################################## SETUP #####################################

# if os.path.exists(adirec):
#     shutil.rmtree(adirec)
# for cond in conds_list_total:
#     shutil.copytree(cond, '%s/%s' % (adirec, cond))

################################ RUN #########################################

embryos_list_total = embryos_direcslist(direcslist(adirec))
# Parallel(n_jobs=4, verbose=50)(delayed(segment)(embryo) for embryo in embryos_list_total)
# Parallel(n_jobs=4, verbose=50)(delayed(run_analysis)(embryo, Data, Res, Analysis) for embryo in embryos_list_total)

################################ IMPORT ######################################


kk1273_wt = batch_import(adirec, np.array(conds_list_total)[[2, 5]], Res)
kk1273_par6 = batch_import(adirec, np.array(conds_list_total)[[13]], Res)
nwg0123_wt = batch_import(adirec, np.array(conds_list_total)[[0, 1]], Res)
nwg0062_wt = batch_import(adirec, np.array(conds_list_total)[[3]], Res)
nwg0062_par6 = batch_import(adirec, np.array(conds_list_total)[[14]], Res)
jh1799_wt = batch_import(adirec, np.array(conds_list_total)[[4, 7, 10]], Res)
jh1799_par6 = batch_import(adirec, np.array(conds_list_total)[[6]], Res)
jh1799_ctrl = batch_import(adirec, np.array(conds_list_total)[[9]], Res)
jh2882_wt = batch_import(adirec, np.array(conds_list_total)[[8]], Res)
jh2882_par6 = batch_import(adirec, np.array(conds_list_total)[[17]], Res)
jh2817_wt = batch_import(adirec, np.array(conds_list_total)[[12]], Res)
jh2817_par6 = batch_import(adirec, np.array(conds_list_total)[[11]], Res)
th129_wt = batch_import(adirec, np.array(conds_list_total)[[16]], Res)
th129_par6 = batch_import(adirec, np.array(conds_list_total)[[15]], Res)
nwg123_wt_bleach = batch_import(adirec, np.array(conds_list_total)[[18]], Res)
