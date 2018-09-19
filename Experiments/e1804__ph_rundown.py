import BgCurves as b
import AFSettings as s
from IA import *

"""
PH-rundown control experiment (April 2018)


"""

############################# INPUT / SETTINGS ################################

# Input data
conds_list_total = [
    '180420_nwg1_0920xfp_tom4,5,30',
    '180420_nwg1_1100xfp_tom4,5,30',
    '180420_nwg1_1300xfp_tom4,5,30',
    '180420_nwg1_1600(180419)xfp_tom4,5,30',
    '180420_nwg1_wt_tom4,5,30']

# Global variables
bgcurve = b.bgG2
settings = s.OD70s1
adirec = '../Analysis/%s' % os.path.basename(__file__)[:-3]
mag = 1


################################ DATA IMPORT #################################

class Data:
    def __init__(self, direc):
        self.direc = direc
        self.DIC = loadimage(sorted(glob.glob('%s/*DIC SP Camera*' % direc), key=len)[0])
        self.GFP = loadimage(sorted(glob.glob('%s/*488 SP 535-50*' % direc), key=len)[0])
        self.AF = loadimage(sorted(glob.glob('%s/*488 SP 630-75*' % direc), key=len)[0])
        self.RFP = loadimage(sorted(glob.glob('%s/*561 SP 630-75*' % direc), key=len)[0])
        self.ROI = np.loadtxt('%s/ROI.txt' % direc)


############################### SEGMENTATION #################################


def segment(direc):
    try:
        data = Data(direc)
        coors = fit_coordinates_alg(data.RFP, data.ROI, bgcurve, 2, mag=mag)
        np.savetxt('%s/ROI_fitted.txt' % direc, coors, fmt='%.4f', delimiter='\t')
    except np.linalg.linalg.LinAlgError:
        print(direc)


################################ ANALYSIS ####################################


class Res:
    def __init__(self):
        self.g_mem = []
        self.r_mem = []
        self.g_cyt = []
        self.r_cyt = []
        self.g_tot = []
        self.r_tot = []
        self.g_spa = []
        self.r_spa = []
        self.g_cse = []
        self.r_cse = []


class Analysis:
    def __init__(self):
        self.res = Res()

    def g_mem(self, data, coors):
        self.res.g_mem = [cortical_signal_g(data, coors, bgcurve, settings, bounds=(0.9, 0.1), mag=mag)]

    def r_mem(self, data, coors):
        self.res.r_mem = [cortical_signal_r(data, coors, bgcurve, bounds=(0, 1), mag=mag)]

    def g_cyt(self, data, coors):
        self.res.g_cyt = [cytoplasmic_signal_g(data, coors, settings, mag=mag)]

    def r_cyt(self, data, coors):
        self.res.r_cyt = [cytoplasmic_signal_r(data, coors, mag=mag)]

    def g_tot(self, data, coors):
        cyt = cytoplasmic_signal_g(data, coors, settings, mag=mag)
        self.res.g_tot = [cyt + (geometry(coors)[0] / geometry(coors)[1]) * cortical_signal_g(data, coors, bgcurve,
                                                                                              settings, bounds=(0, 1),
                                                                                              mag=mag)]

    def r_tot(self, data, coors):
        cyt = cytoplasmic_signal_r(data, coors, mag=mag)
        self.res.r_tot = [cyt + (geometry(coors)[0] / geometry(coors)[1]) * cortical_signal_r(data, coors, bgcurve,
                                                                                              bounds=(0, 1),
                                                                                              mag=mag)]

    def g_spa(self, data, coors):
        self.res.g_spa = spatial_signal_g(data, coors, bgcurve, settings, mag=mag)

    def r_spa(self, data, coors):
        self.res.r_spa = spatial_signal_r(data, coors, bgcurve, mag=mag)

    def g_cse(self, data, coors):
        self.res.g_cse = cross_section(img=af_subtraction(data.GFP, data.AF, settings=settings), coors=coors,
                                       thickness=10, extend=1.5)

    def r_cse(self, data, coors):
        bg = straighten(data.RFP, offset_coordinates(coors, 50 * mag), 50 * mag)
        self.res.r_cse = cross_section(img=data.RFP, coors=coors, thickness=10, extend=1.5) - np.nanmean(
            bg[np.nonzero(bg)])




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

ph_wt = batch_import(adirec, np.array(conds_list_total)[[0, 1, 2, 3]], Res)
ph_rd = batch_import(adirec, np.array(conds_list_total)[[4]], Res)
