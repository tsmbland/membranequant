from IA import *
import AFSettings as s
import BgCurves as b

# Carry out normal quantification, with segmentation using own bgcurves
# ie cortical concentration should be zero

dest = adirec + '/BgCurves_Control/'


class BatchAnalysis:
    def __init__(self, direcs, dest, mag, dclass, bg_g=None, bg_a=None, bg_r=None, bg_c=None, settings=None,
                 bounds=(0, 1), setup=False, segment=None, funcs=None, parallel=True, display=False):
        self.direcs = direcs
        self.dest = dest
        self.mag = mag
        self.dclass = dclass
        self.setup = False
        self.settings = settings
        self.bounds = bounds
        self.bg_g = bg_g
        self.bg_a = bg_a
        self.bg_r = bg_r
        self.bg_c = bg_c
        self.segment = segment
        self.funcs = 'all'
        self.parallel = True
        self.display = display

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
                print(e)

    def f_segmentation(self, e):
        data = self.dclass(e)

        if self.segment == '2':
            """

            """
            img = af_subtraction(data.GFP, data.AF, self.settings)
            Segmenter2(img, data.RFP, self.bg_c, self.bg_r, data.ROI, parallel=self.parallel, save=True,
                       direc=e, plot=self.display).run()

        elif self.segment == '3':
            """

            """
            img = af_subtraction(data.GFP, data.AF, self.settings)
            Segmenter(img, self.bg_c, data.ROI, parallel=self.parallel, save=True, direc=e, plot=self.display).run()

        else:
            pass

    def f_analysis(self, e):
        data = self.dclass(e)
        coors = np.loadtxt('%s/ROI_fitted.txt' % e, delimiter='\t')
        if data.GFP is not None:
            Analyser(img=data.GFP, coors=coors, bg=self.bg_g, funcs=self.funcs, bounds=self.bounds,
                     direc=data.direc,
                     name='g').run()
            Analyser(img=data.AF, coors=coors, bg=self.bg_a, funcs=self.funcs, bounds=self.bounds,
                     direc=data.direc,
                     name='a').run()
            Analyser(img=af_subtraction(data.GFP, data.AF, self.settings), coors=coors, bg=self.bg_c,
                     funcs=self.funcs, bounds=self.bounds, direc=data.direc, name='c').run()

        if data.RFP is not None:
            Analyser(img=data.RFP, coors=coors, bg=self.bg_r, funcs=self.funcs, bounds=self.bounds,
                     direc=data.direc,
                     name='r').run()
            Analyser(img=bg_subtraction(data.RFP, coors, mag=self.mag), coors=coors, bg=self.bg_r,
                     funcs=self.funcs,
                     bounds=self.bounds, direc=data.direc, name='b').run()


"""
PAR-2 GFP, anterior

kk1273_g

"""

BatchAnalysis(direcs=['180223_kk1273_wt_tom3,15,pfsin'],
              dest=dest + 'kk1273/',
              mag=1,
              dclass=Importers.Data3,
              bg_g=b.d['kk1273_g'],
              bg_a=b.d['kk1273_a'],
              bg_c=b.d['kk1273_c'],
              settings=s.N2s1,
              bounds=[0.3, 0.7],
              segment='3').run()

"""
PAR-2 mCherry, PAR-1 GFP, anterior

nwg42_g
nwg42_r

"""

BatchAnalysis(direcs=['180322_nwg42_wt_tom4,15,30,pfsout'],
              dest=dest + 'nwg42/',
              mag=1,
              dclass=Importers.Data0,
              bg_g=b.d['nwg42_g'],
              bg_a=b.d['nwg42_a'],
              bg_c=b.d['nwg42_c'],
              bg_r=b.d['nwg42_r'],
              settings=s.N2s2,
              bounds=[0.3, 0.7],
              segment='2').run()

"""
PKC GFP, all over

nwg129_g

"""

BatchAnalysis(direcs=['180316_nwg0129_par3_tom3,15,pfsout'],
              dest=dest + 'nwg129/',
              mag=1,
              dclass=Importers.Data3,
              bg_g=b.d['nwg129_g'],
              bg_a=b.d['nwg129_a'],
              bg_c=b.d['nwg129_c'],
              settings=s.N2s2,
              bounds=[0, 1],
              segment='3').run()

"""
PAR-2 GFP, PH mCherry, anterior

nwg129_g

"""
BatchAnalysis(direcs=['180416_nwg0106_wt_tom4,15,30,pfsout'],
              dest=dest + 'nwg106/',
              mag=1,
              dclass=Importers.Data0,
              bg_g=b.d['nwg106_g'],
              bg_a=b.d['nwg106_a'],
              bg_c=b.d['nwg106_c'],
              bg_r=b.d['nwg42_r'],
              settings=s.N2s2,
              bounds=[0.3, 0.7],
              segment='3').run()

"""
PAR-2 GFP, PAR-2 mCherry, anterior
Problem: some embryos have PAR-2 patch in anterior

nwg151_g
nwg151_r

"""
BatchAnalysis(direcs=['180730_nwg0151_wt_tom4,15,30',
                      '180726_nwg0151_wt_tom4,15,30'],
              dest=dest + 'nwg0151/',
              mag=1,
              dclass=Importers.Data0,
              bg_g=b.d['nwg0151_g'],
              bg_a=b.d['nwg0151_a'],
              bg_c=b.d['nwg0151_c'],
              bg_r=b.d['nwg0151_r'],
              settings=s.N2s2,
              bounds=[0.3, 0.7],
              segment='2').run()

"""
PAR-2 GFP, PAR-2 mCherry, anterior, half dosage

nwg151dr444_g
nwg151dr466_r

"""

BatchAnalysis(direcs=['180804_nwg0151xdr466_wt_tom4,15,30'],
              dest=dest + 'nwg0151xdr466/',
              mag=1,
              dclass=Importers.Data0,
              bg_g=b.d['nwg0151xdr466_g'],
              bg_a=b.d['nwg0151xdr466_a'],
              bg_c=b.d['nwg0151xdr466_c'],
              bg_r=b.d['nwg0151xdr466_r'],
              settings=s.N2s2,
              bounds=[0.3, 0.7],
              segment='2').run()
