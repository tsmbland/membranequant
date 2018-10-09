import sys
import pickle
import numpy as np

sys.path.append('..')
import IA as x
import AFSettings as s

"""

NWG0129, NWG42 and KK1273 are the best


"""

dest = '/Users/blandt/Desktop/Analysis/BgCurves/'


class Params:
    def __init__(self, mag=None, dclass=None, settings=None, bounds=(0, 1), segment=None, funcs=None):
        self.mag = mag
        self.dclass = dclass
        self.settings = settings
        self.bounds = bounds
        self.segment = segment
        self.funcs = funcs


class Analysis:
    def __init__(self, direc):
        self.direc = direc
        with open('%s/params.pkl' % direc, 'rb') as f:
            self.params = pickle.load(f)

    def run(self):

        # Segmentation
        if self.params.segment is not None:
            self.f_segmentation(self.direc)

        # Analysis
        self.f_analysis(self.direc)

    def f_segmentation(self, e):
        data = self.params.dclass(e)
        if self.params.segment == '0':
            """

            """

            img = x.af_subtraction(data.GFP, data.AF, self.params.settings)
            x.Segmenter3(img, data.ROI, plot=False, save=True, direc=e, mag=self.params.mag).run()

        elif self.params.segment == '1':
            """

            """

            x.Segmenter3(data.RFP, data.ROI, plot=False, save=True, direc=e, mag=self.params.mag).run()

        else:
            pass

    def f_analysis(self, e):
        data = self.params.dclass(e)
        coors = np.loadtxt('%s/ROI_fitted.txt' % e, delimiter='\t')
        if data.GFP is not None:
            x.Analyser(img=data.GFP, coors=coors, funcs=self.params.funcs, bounds=self.params.bounds,
                       direc=data.direc,
                       name='g', thickness=100, mag=self.params.mag).run()
            x.Analyser(img=data.AF, coors=coors, funcs=self.params.funcs, bounds=self.params.bounds,
                       direc=data.direc,
                       name='a', thickness=100, mag=self.params.mag).run()
            x.Analyser(img=x.af_subtraction(data.GFP, data.AF, self.params.settings), coors=coors,
                       funcs=self.params.funcs, bounds=self.params.bounds, direc=data.direc, name='c',
                       thickness=100).run()

        if data.RFP is not None:
            x.Analyser(img=data.RFP, coors=coors, funcs=self.params.funcs, bounds=self.params.bounds,
                       direc=data.direc,
                       name='r', thickness=100, mag=self.params.mag).run()
            x.Analyser(img=x.bg_subtraction(data.RFP, coors, mag=self.params.mag), coors=coors,
                       funcs=self.params.funcs,
                       bounds=self.params.bounds, direc=data.direc, name='b', thickness=100, mag=self.params.mag).run()


"""
PAR-2 GFP, anterior

kk1273_g

"""

x.save_params(dest=dest + 'kk1273/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        settings=s.N2s1,
                        bounds=[0.3, 0.7],
                        segment='0'))

"""
PAR-2 mCherry, PAR-1 GFP, anterior

nwg42_g
nwg42_r

"""

x.save_params(dest=dest + 'nwg42/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='0'))

"""
PKC GFP, all over

nwg129_g

"""

x.save_params(dest=dest + 'nwg129/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        settings=s.N2s2,
                        bounds=[0, 1],
                        segment='0'))

"""
PAR-2 GFP, PH mCherry, anterior

nwg106_g
nwg106_r

"""
x.save_params(dest=dest + 'nwg106/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='0'))

"""
PAR-2 GFP, PAR-2 mCherry, anterior
Problem: some embryos have PAR-2 patch in anterior

nwg151_g
nwg151_r

"""
x.save_params(dest=dest + 'nwg0151/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='0'))

"""
PAR-2 GFP, PAR-2 mCherry, anterior, half dosage

nwg151dr444_g
nwg151dr466_r

"""

x.save_params(dest=dest + 'nwg0151xdr466/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='1'))
