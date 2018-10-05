from IA import *
import AFSettings as s

"""

NWG0129, NWG42 and KK1273 are the best


"""

dest = adirec + '/BgCurves/'


class BatchAnalysis:
    def __init__(self, direcs, dest, mag, dclass, settings=None, bounds=(0, 1), setup=False,
                 segment=None, funcs=None, parallel=True):
        self.direcs = direcs
        self.dest = dest
        self.mag = mag
        self.dclass = dclass
        self.setup = False
        self.settings = settings
        self.bounds = bounds
        self.segment = None
        self.funcs = ['pro']
        self.parallel = parallel

    def run(self):

        # Setup
        if self.setup:
            copy_data(self.direcs, self.dest)
        embryos_list_total = embryos_direcslist(append_batch(self.dest, self.direcs))

        # Segmentation
        for e in embryos_list_total:
            self.f_segmentation(e)
            print(e)

        # Analysis
        if self.parallel:
            Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(
                delayed(self.f_analysis)(e) for e in embryos_list_total)

        else:
            for e in embryos_list_total:
                self.f_analysis(e)

    def f_segmentation(self, e):
        data = self.dclass(e)
        if self.segment == '0':
            """

            """

            img = af_subtraction(data.GFP, data.AF, self.settings)
            Segmenter3(img, data.ROI, plot=False, save=True, direc=e).run()

        elif self.segment == '1':
            """

            """

            Segmenter3(data.RFP, data.ROI, plot=False, save=True, direc=e).run()

        else:
            pass

    def f_analysis(self, e):
        data = self.dclass(e)
        coors = np.loadtxt('%s/ROI_fitted.txt' % e, delimiter='\t')
        if data.GFP is not None:
            Analyser(img=data.GFP, coors=coors, funcs=self.funcs, bounds=self.bounds,
                     direc=data.direc,
                     name='g', thickness=100).run()
            Analyser(img=data.AF, coors=coors, funcs=self.funcs, bounds=self.bounds,
                     direc=data.direc,
                     name='a', thickness=100).run()
            Analyser(img=af_subtraction(data.GFP, data.AF, self.settings), coors=coors,
                     funcs=self.funcs, bounds=self.bounds, direc=data.direc, name='c', thickness=100).run()

        if data.RFP is not None:
            Analyser(img=data.RFP, coors=coors, funcs=self.funcs, bounds=self.bounds,
                     direc=data.direc,
                     name='r', thickness=100).run()
            Analyser(img=bg_subtraction(data.RFP, coors, mag=self.mag), coors=coors,
                     funcs=self.funcs,
                     bounds=self.bounds, direc=data.direc, name='b', thickness=100).run()

# """
# PAR-2 GFP, anterior
#
# kk1273_g
#
# """
#
# BatchAnalysis(direcs=['180223_kk1273_wt_tom3,15,pfsin'],
#               dest=dest + 'kk1273/',
#               mag=1,
#               dclass=Importers.Data3,
#               settings=s.N2s1,
#               bounds=[0.3, 0.7],
#               segment='0').run()
#
# """
# PAR-2 mCherry, PAR-1 GFP, anterior
#
# nwg42_g
# nwg42_r
#
# """
#
# BatchAnalysis(direcs=['180322_nwg42_wt_tom4,15,30,pfsout'],
#               dest=dest + 'nwg42/',
#               mag=1,
#               dclass=Importers.Data0,
#               settings=s.N2s2,
#               bounds=[0.3, 0.7],
#               segment='0').run()
#
# """
# PKC GFP, all over
#
# nwg129_g
#
# """
#
# BatchAnalysis(direcs=['180316_nwg0129_par3_tom3,15,pfsout'],
#               dest=dest + 'nwg129/',
#               mag=1,
#               dclass=Importers.Data3,
#               settings=s.N2s2,
#               bounds=[0, 1],
#               segment='0').run()
#
# """
# PAR-2 GFP, PH mCherry, anterior
#
# nwg106_g
# nwg106_r
#
# """
# BatchAnalysis(direcs=['180416_nwg0106_wt_tom4,15,30,pfsout'],
#               dest=dest + 'nwg106/',
#               mag=1,
#               dclass=Importers.Data0,
#               settings=s.N2s2,
#               bounds=[0.3, 0.7],
#               segment='0').run()
#
# """
# PAR-2 GFP, PAR-2 mCherry, anterior
# Problem: some embryos have PAR-2 patch in anterior
#
# nwg151_g
# nwg151_r
#
# """
# BatchAnalysis(direcs=['180730_nwg0151_wt_tom4,15,30',
#                       '180726_nwg0151_wt_tom4,15,30'],
#               dest=dest + 'nwg0151/',
#               mag=1,
#               dclass=Importers.Data0,
#               settings=s.N2s2,
#               bounds=[0.3, 0.7],
#               segment='0').run()
#
# """
# PAR-2 GFP, PAR-2 mCherry, anterior, half dosage
#
# nwg151dr444_g
# nwg151dr466_r
#
# """
#
# BatchAnalysis(direcs=['180804_nwg0151xdr466_wt_tom4,15,30'],
#               dest=dest + 'nwg0151xdr466/',
#               mag=1,
#               dclass=Importers.Data0,
#               settings=s.N2s2,
#               bounds=[0.3, 0.7],
#               segment='1').run()
