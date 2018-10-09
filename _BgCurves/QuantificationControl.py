import sys
import pickle
import numpy as np

sys.path.append('..')
import IA as x
import AFSettings as s
import BgCurves as b

dest = '/Users/blandt/Desktop/Analysis/BgCurves_Control/'


class Params:
    def __init__(self, mag=None, dclass=None, bg_g=None, bg_a=None, bg_r=None, bg_c=None, settings=None, bounds=(0, 1),
                 segment=None, funcs=None):
        self.mag = mag
        self.dclass = dclass
        self.settings = settings
        self.bounds = bounds
        self.bg_g = bg_g
        self.bg_a = bg_a
        self.bg_r = bg_r
        self.bg_c = bg_c
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

        if self.params.segment == '2':
            """

            """
            img = x.af_subtraction(data.GFP, data.AF, self.params.settings)
            x.Segmenter2(img, data.RFP, self.params.bg_c, self.params.bg_r, data.ROI, direc=e,
                         mag=self.params.mag).run()

        elif self.params.segment == '3':
            """

            """
            img = x.af_subtraction(data.GFP, data.AF, self.params.settings)
            x.Segmenter(img, self.params.bg_c, data.ROI, direc=e, mag=self.params.mag).run()

        else:
            pass

    def f_analysis(self, e):

        data = self.params.dclass(e)
        coors = np.loadtxt('%s/ROI_fitted.txt' % e, delimiter='\t')
        if data.GFP is not None:
            x.Analyser(img=data.GFP, coors=coors, bg=self.params.bg_g, funcs=self.params.funcs,
                       bounds=self.params.bounds,
                       direc=data.direc, name='g', mag=self.params.mag).run()
            x.Analyser(img=data.AF, coors=coors, bg=self.params.bg_a, funcs=self.params.funcs,
                       bounds=self.params.bounds,
                       direc=data.direc, name='a', mag=self.params.mag).run()
            x.Analyser(img=x.af_subtraction(data.GFP, data.AF, self.params.settings), coors=coors, bg=self.params.bg_c,
                       funcs=self.params.funcs, bounds=self.params.bounds, direc=data.direc, name='c',
                       mag=self.params.mag).run()

        if data.RFP is not None:
            x.Analyser(img=data.RFP, coors=coors, bg=self.params.bg_r, funcs=self.params.funcs,
                       bounds=self.params.bounds,
                       direc=data.direc, name='r', mag=self.params.mag).run()
            x.Analyser(img=x.bg_subtraction(data.RFP, coors, mag=self.params.mag), coors=coors, bg=self.params.bg_r,
                       funcs=self.params.funcs, bounds=self.params.bounds, direc=data.direc, name='b',
                       mag=self.params.mag).run()


"""
PAR-2 GFP, anterior

kk1273_g

"""

x.save_params(dest=dest + 'kk1273/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['kk1273_g'],
                        bg_a=b.d['kk1273_a'],
                        bg_c=b.d['kk1273_c'],
                        settings=s.N2s1,
                        bounds=[0.3, 0.7],
                        segment='3'))

"""
PAR-2 mCherry, PAR-1 GFP, anterior

nwg42_g
nwg42_r

"""

x.save_params(dest=dest + 'nwg42/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='2'))

"""
PKC GFP, all over

nwg129_g

"""

x.save_params(dest=dest + 'nwg129/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['nwg129_g'],
                        bg_a=b.d['nwg129_a'],
                        bg_c=b.d['nwg129_c'],
                        settings=s.N2s2,
                        bounds=[0, 1],
                        segment='3'))

"""
PAR-2 GFP, PH mCherry, anterior

nwg129_g

"""
x.save_params(dest=dest + 'nwg106/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg106_g'],
                        bg_a=b.d['nwg106_a'],
                        bg_c=b.d['nwg106_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='3'))

"""
PAR-2 GFP, PAR-2 mCherry, anterior
Problem: some embryos have PAR-2 patch in anterior

nwg151_g
nwg151_r

"""
x.save_params(dest=dest + 'nwg0151/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg0151_g'],
                        bg_a=b.d['nwg0151_a'],
                        bg_c=b.d['nwg0151_c'],
                        bg_r=b.d['nwg0151_r'],
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='2'))

"""
PAR-2 GFP, PAR-2 mCherry, anterior, half dosage

nwg151dr444_g
nwg151dr466_r

"""

x.save_params(dest=dest + 'nwg0151xdr466/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg0151xdr466_g'],
                        bg_a=b.d['nwg0151xdr466_a'],
                        bg_c=b.d['nwg0151xdr466_c'],
                        bg_r=b.d['nwg0151xdr466_r'],
                        settings=s.N2s2,
                        bounds=[0.3, 0.7],
                        segment='2'))
