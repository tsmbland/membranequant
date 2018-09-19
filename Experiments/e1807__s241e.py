import BgCurves as b
import AFSettings as s
from IA import *

############################# INPUT / SETTINGS ################################

# Input data
conds_list_total = ['180726_kk1273_wt_tom4,15,30',
                    '180726_nw62_wt_tom4,15,30',
                    '180726_nwg0126_wt_tom4,15,30',
                    '180727_nwg0126_wt_tom4,15,30',
                    '180726_kk1273_wt_tom4,15,30+bleach',
                    '180726_nw62_wt_tom4,15,30+bleach',
                    '180726_nwg0126_wt_tom4,15,30+bleach',
                    '180727_nwg0126_wt_tom4,15,30+bleach']
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
    data = Data(direc)
    img = af_subtraction(data.GFP, data.AF, settings=settings)
    coors = fit_coordinates_alg(img, data.ROI, bgcurve, 2, mag=mag)
    np.savetxt('%s/ROI_fitted.txt' % direc, coors, fmt='%.4f', delimiter='\t')


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

kk1273_wt = batch_import(adirec, np.array(conds_list_total)[[0]], Res)
nwg62_wt = batch_import(adirec, np.array(conds_list_total)[[1]], Res)
nwg126_wt = batch_import(adirec, np.array(conds_list_total)[[2, 3]], Res)
kk1273_wt_bleach = batch_import(adirec, np.array(conds_list_total)[[4]], Res)
nwg62_wt_bleach = batch_import(adirec, np.array(conds_list_total)[[5]], Res)
nwg126_wt_bleach = batch_import(adirec, np.array(conds_list_total)[[6, 7]], Res)
