import sys
import pickle
import numpy as np

sys.path.append('..')
import IA as x
import AFSettings as s
import BgCurves as b


# dest = '/Users/blandt/Desktop/Analysis/Experiments/'
dest = '../../working/Tom/ModelData/ImageAnalysis/Analysis/Experiments/'


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
e1802__par6_rundown_nelio
Nelio's PAR-6 rundown data with NWG26 line (Jan-Feb 2018)

"""

x.save_params(dest=dest + 'e1802__par6_rundown_nelio/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s5,
                        bounds=[0, 1],
                        segment='2'))

"""
e1803__par2_rundown
PAR-2 rundown experiment (Feb-March 2018)
PFS in for all embryos so not directly comparable with other experiments

"""

x.save_params(dest=dest + 'e1803__par2_rundown/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['nwg129_g'],
                        bg_a=b.d['nwg129_a'],
                        bg_c=b.d['nwg129_c'],
                        settings=s.N2s1,
                        bounds=[0.9, 0.1],
                        segment='3'))
"""
e1803__par2_rundown2
PAR-2 rundown experiment
PFS out

"""

x.save_params(dest=dest + 'e1803__par2_rundown2/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['nwg129_g'],
                        bg_a=b.d['nwg129_a'],
                        bg_c=b.d['nwg129_c'],
                        settings=s.N2s2,
                        bounds=[0.9, 0.1],
                        segment='3'))

"""
e1803__par2par1_rundown
PAR-2 rundown, looking at PAR-1 and PAR-2 to assess recruitment (March 2018)

"""

x.save_params(dest=dest + 'e1803__par2par1_rundown/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s2,
                        bounds=[0.9, 0.1],
                        segment='2'))

"""
e1803__pkc_it71
Looking at PKC in embryos with it71 mutant PAR-3, comparing with wt


"""

x.save_params(dest=dest + 'e1803__pkc_it71/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        settings=s.N2s2,
                        bounds=[0, 1],
                        segment='3'))

"""
e1803__pkc_rundown_florent_exp1

"""

x.save_params(dest=dest + 'e1803__pkc_rundown_florent_exp1/',
              cl=Params(mag=5 / 3,
                        dclass=x.Importers.Data1,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s7,
                        bounds=[0, 1],
                        segment='2'))

"""
e1803__pkc_rundown_florent_exp2

"""

x.save_params(dest=dest + 'e1803__pkc_rundown_florent_exp2/',
              cl=Params(mag=5 / 3,
                        dclass=x.Importers.Data2,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s8,
                        bounds=[0, 1],
                        segment='2'))

"""
e1803__pkc_rundown_florent_exp3

"""

x.save_params(dest=dest + 'e1803__pkc_rundown_florent_exp3/',
              cl=Params(mag=5 / 3,
                        dclass=x.Importers.Data2,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s9,
                        bounds=[0, 1],
                        segment='2'))

"""
e1804__par2_rundown_nelio
Nelio's PAR-2 rundown data in NWG76 (March-April 2018)


"""

x.save_params(dest=dest + 'e1804__par2_rundown_nelio/',
              cl=Params(mag=2,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s4,
                        bounds=[0.9, 0.1],
                        segment='2'))

"""
e1804__ph_rundown
PH-rundown control experiment (April 2018)

"""

x.save_params(dest=dest + 'e1804__ph_rundown/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.OD70s1,
                        bounds=[0.9, 0.1],
                        segment='2'))

"""
e1805__od70

"""

x.save_params(dest=dest + 'e1805__od70/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s2,
                        bounds=[0, 1],
                        segment='2'))

"""
e1806__nwg91
June 2018
Playing with PAR-2/PKC double line in variety of conditions, looking for correlation between PAR-2 and PKC


"""

x.save_params(dest=dest + 'e1806__nwg91/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s2,
                        bounds=[0, 1],
                        segment='2'))

"""
e1806__par2_mutants
Comparision of cytoplasmic affinity of different PAR-2 mutants in PKC+/- conditions (June 2018)


"""

x.save_params(dest=dest + 'e1806__par2_mutants/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['nwg129_g'],
                        bg_a=b.d['nwg129_a'],
                        bg_c=b.d['nwg129_c'],
                        settings=s.N2s2,
                        bounds=[0.9, 0.1],
                        segment='3'))

"""
e1806__par2_mutants2
Comparision of cytoplasmic affinity of different PAR-2 mutants in PKC+/- conditions (June 2018)
On SD after move to new room

"""

x.save_params(dest=dest + 'e1806__par2_mutants2/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['nwg129_g'],
                        bg_a=b.d['nwg129_a'],
                        bg_c=b.d['nwg129_c'],
                        settings=s.N2s2,
                        bounds=[0.9, 0.1],
                        segment='3'))

"""
e1807__optogenetics
Throws error

"""

# BatchAnalysis(direcs=[direcslist('180712_sv2061_wt_tom4,15,30,100msecG')],
#               dest=dest + 'e1807__optogenetics',
#               mag=1,
#               dclass=Importers.Data0,
#               bgcurve=b.bgG4,
#               settings=s.N2s0,
#               bounds=[0, 1],
#               segment='2').run()

"""
e1807__par2_c1b
Looking at nwg0145 line, aiming to understand if PAR-2 homodimerises


"""

x.save_params(dest=dest + 'e1807__par2_c1b/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s2,
                        bounds=[0, 1],
                        segment='2'))

"""
e1807__par2_gbp
PAR-2 GBP


"""

x.save_params(dest=dest + 'e1807__par2_gbp/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data0,
                        bg_g=b.d['nwg42_g'],
                        bg_a=b.d['nwg42_a'],
                        bg_c=b.d['nwg42_c'],
                        bg_r=b.d['nwg42_r'],
                        settings=s.N2s10,
                        bounds=[0, 1],
                        segment='2'))
#
"""
e1807__s241e

"""

x.save_params(dest=dest + 'e1807__s241e/',
              cl=Params(mag=1,
                        dclass=x.Importers.Data3,
                        bg_g=b.d['nwg129_g'],
                        bg_a=b.d['nwg129_a'],
                        bg_c=b.d['nwg129_c'],
                        settings=s.N2s2,
                        bounds=[0.9, 0.1],
                        segment='3'))

"""
e1808__par2_c1b_2
Looking at nwg0158 line, aiming to understand if PAR-2 homodimerises

Throws error

"""

# BatchAnalysis(direcs=['180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e1',
#                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e2',
#                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e3',
#                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e4',
#                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e5',
#                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30_MP',
#                       '180726_nwg0158_wt_tom4,15,30',
#                       '180730_nwg0151_wt_tom4,15,30',
#                       '180726_nwg0151_wt_tom4,15,30'],
#               dest=dest + 'e1808__par2_c1b_2/',
#               mag=1,
#               dclass=Importers.Data0,
#               bgcurve=b.bgG4,
#               settings=s.N2s2,
#               bounds=[0, 1],
#               segment='3').run()


# for i, d in enumerate(x.direcslist2(dest, 3)):
#     print(i)
#     # Analysis(d).run()
