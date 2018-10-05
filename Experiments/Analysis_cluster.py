from IA import *
import AFSettings as s
import BgCurves as b

dest = adirec + '/Experiments/'


class Analysis:
    def __init__(self, direcs, dest, mag, dclass, bg_g=None, bg_a=None, bg_r=None, bg_c=None, settings=None,
                 bounds=(0, 1), setup=False, segment=None, funcs=None, parallel=True, display=True):
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
        self.funcs = ['spa', 'asi', 'fbc']
        self.parallel = True
        self.display = display

    def run(self):

        # Setup
        if self.setup:
            setup(self.direcs, self.dest)
        embryos_list_total = func(self.dest)

        # Segmentation
        if self.segment is not None:
            for e in embryos_list_total:
                self.f_segmentation(e)
                print(e)

        # Analysis
        if self.parallel:
            Parallel(n_jobs=-1, verbose=50)(
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
