import sys
import pickle
import numpy as np

sys.path.append('..')
import IA as x

dest = '/Users/blandt/Desktop/Analysis/AFSettings/'


class Params:
    def __init__(self, mag=None, dclass=None):
        self.mag = mag
        self.dclass = dclass


class Analysis:
    def __init__(self, direc):
        self.direc = direc
        with open('%s/params.pkl' % direc, 'rb') as f:
            self.params = pickle.load(f)

    def run(self):
        self.f_analysis(self.direc)

    def f_analysis(self, e):
        data = self.params.dclass(e)
        coors = data.ROI
        if data.GFP is not None:
            x.Analyser(data.GFP, coors, funcs=['cyt', 'ext'], direc=data.direc, name='g', mag=self.params.mag).run()
            x.Analyser(data.AF, coors, funcs=['cyt', 'ext'], direc=data.direc, name='a', mag=self.params.mag).run()
        if data.RFP is not None:
            x.Analyser(data.RFP, coors, funcs=['cyt', 'ext'], direc=data.direc, name='r', mag=self.params.mag).run()


"""
N2s1
N2, Tom3, 15, pfsin

"""
x.save_params(dest=dest + 'N2s1/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3))

"""
N2s2
N2, Tom3, 15, pfsout

"""
x.save_params(dest=dest + 'N2s2/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3))

"""
N2s3
N2, Tom3, 5, pfsout

"""
x.save_params(dest=dest + 'N2s3/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3))

"""
N2s4
N2, PAR2 Nelio

"""
x.save_params(dest=dest + 'N2s4/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0))

"""
N2s5
N2, PAR6 Nelio

"""
x.save_params(dest=dest + 'N2s5/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0))

"""
N2s7
N2, Florent 180110

"""
x.save_params(dest=dest + 'N2s7/',
              cl=Params(mag=5 / 3,
                        dclass=x.Importers.Data4))

"""
N2s8
N2, Florent 180221

"""
x.save_params(dest=dest + 'N2s8/',
              cl=Params(mag=5 / 3,
                        dclass=x.Importers.Data4))

"""
N2s9
N2, Florent 180301

"""
x.save_params(dest=dest + 'N2s9/',
              cl=Params(mag=5 / 3,
                        dclass=x.Importers.Data5))

"""
N2s10
N2, Tom4, 15, 30, pfsout, after microscope move

"""
x.save_params(dest=dest + 'N2s10/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0))

"""
N2s11
N2, Tom4, 15, 30, pfsout, after microscope move, with bleach

"""
x.save_params(dest=dest + 'N2s11/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0))

"""
OD70s1
OD70, Tom5, 30, pfsout

"""
x.save_params(dest=dest + 'OD70s1/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0))

"""
KK1254_0

"""

"""
Box241_0

"""
