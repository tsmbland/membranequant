from IA import *
import AFSettings as s
import BgCurves as b

dest = adirec + '/Experiments/'


class BatchAnalysis:
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


# """
# e1802__par6_rundown_nelio
# Nelio's PAR-6 rundown data with NWG26 line (Jan-Feb 2018)
#
# """
#
# BatchAnalysis(direcs={'nwg26_wt': ['180127_nwg26_wt_Nelio2',
#                                    '180128_nwg26_wt_Nelio2',
#                                    '180129_nwg26_wt_Nelio2',
#                                    '180218(am)_nwg26_wt_Nelio2'],
#                       'nwg26_rd': ['180127_nwg26_par6rundown_Nelio2',
#                                    '180128_nwg26_par6rundown_Nelio2',
#                                    '180129_nwg26_par6rundown_Nelio2',
#                                    '180218(am)_nwg26_par6rundown_Nelio2',
#                                    '180218(pm)_nwg26_par6rundown_Nelio2']},
#               dest=dest + 'e1802__par6_rundown_nelio/',
#               mag=1,
#               dclass=Importers.Data0,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s5,
#               bounds=[0, 1],
#               segment='2').run()
#
# """
# e1803__par2_rundown
# PAR-2 rundown experiment (Feb-March 2018)
# PFS in for all embryos so not directly comparable with other experiments
#
# """
#
# BatchAnalysis(direcs={'nwg0123_wt': ['180309_nwg0123_par3_tom3,15,pfsin',
#                                      '180223_nwg0123_24hr0par2,par3_tom3,15,pfsin'],
#                       'nwg0123_rd': ['180309_nwg0123_par3,0945par2_tom3,15,pfsin',
#                                      '180309_nwg0123_par3,1100par2_tom3,15,pfsin',
#                                      '180309_nwg0123_par3,1200par2_tom3,15,pfsin',
#                                      '180309_nwg0123_par3,1300par2_tom3,15,pfsin',
#                                      '180309_nwg0123_par3,1400par2_tom,15,pfsin',
#                                      '180223_nwg0123_24hr10par2,par3_tom3,15,pfsin'],
#                       'kk1273_wt': ['180223_kk1273_wt_tom3,15,pfsin']},
#               dest=dest + 'e1803__par2_rundown/',
#               mag=1,
#               dclass=Importers.Data3,
#               bg_g=b.d['nwg129_g'],
#               bg_a=b.d['nwg129_a'],
#               bg_c=b.d['nwg129_c'],
#               settings=s.N2s1,
#               bounds=[0.9, 0.1],
#               segment='3').run()
#
# """
# e1803__par2_rundown2
# PAR-2 rundown experiment
# PFS out
#
# """
#
# BatchAnalysis(direcs={'nwg0123_rd': ['180302_nwg0123_24hr10par2,par3_tom3,15,pfsout',
#                                      '180302_nwg0123_24hr50par2,par3_tom3,15,pfsout']},
#               dest=dest + 'e1803__par2_rundown2/',
#               mag=1,
#               dclass=Importers.Data3,
#               bg_g=b.d['nwg129_g'],
#               bg_a=b.d['nwg129_a'],
#               bg_c=b.d['nwg129_c'],
#               settings=s.N2s2,
#               bounds=[0.9, 0.1],
#               segment='3').run()
#
# """
# e1803__par2par1_rundown
# PAR-2 rundown, looking at PAR-1 and PAR-2 to assess recruitment (March 2018)
#
# """
#
# BatchAnalysis(direcs={'nwg42_wt': ['180316_nwg42_wt_tom4,15,30,pfsout',
#                                    '180322_nwg42_wt_tom4,15,30,pfsout'],
#                       'nwg0132_wt': ['180322_nwg0132_par3_tom4,15,30,pfsout'],
#                       'nwg0132_rd': ['180322_nwg0132_par3,0945par2_tom4,15,30,pfsout',
#                                      '180322_nwg0132_par3,1115par2_tom4,15,30,pfsout',
#                                      '180322_nwg0132_par3,1255par2_tom4,15,30,pfsout',
#                                      '180501_nwg0132_1320par2_tom4,15,30',
#                                      '180501_nwg0132_1630par2_tom4,15,30']},
#               dest=dest + 'e1803__par2par1_rundown/',
#               mag=1,
#               dclass=Importers.Data0,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s2,
#               bounds=[0.9, 0.1],
#               segment='2').run()
#
# """
# e1803__pkc_it71
# Looking at PKC in embryos with it71 mutant PAR-3, comparing with wt
#
#
# """
#
# BatchAnalysis(direcs={'nwg0129_wt': ['180316_nwg0129_par3_tom3,15,pfsout'],
#                       'kk1228_wt': ['180316_kk1228_wt_tom3,15,pfsout']},
#               dest=dest + 'e1803__pkc_it71/',
#               mag=1,
#               dclass=Importers.Data3,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               settings=s.N2s2,
#               bounds=[0, 1],
#               segment='3').run()
#
# """
# e1803__pkc_rundown_florent_exp1
#
# """
#
# BatchAnalysis(direcs={'nwg91_wt': ['180110_NWG91_0PKC_Florent1'],
#                       'nwg91_rd': ['180110_NWG91_25PKC_Florent1',
#                                    '180110_NWG91_75PKC_Florent1'],
#                       'nwg93_wt': ['180110_NWG93_0PKC_Florent1'],
#                       'nwg93_rd': ['180110_NWG93_25PKC_Florent1',
#                                    '180110_NWG93_75PKC_Florent1']},
#               dest=dest + 'e1803__pkc_rundown_florent_exp1/',
#               mag=5 / 3,
#               dclass=Importers.Data1,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s7,
#               bounds=[0, 1],
#               segment='2').run()
#
# """
# e1803__pkc_rundown_florent_exp2
#
# """
#
# BatchAnalysis(direcs={'nwg91_wt': ['180221_NWG91_0PKC_Florent2'],
#                       'nwg91_rd': ['180221_NWG91_100PKC_Florent2',
#                                    '180221_NWG91_25PKC_Florent2',
#                                    '180221_NWG91_50PKC_Florent2'],
#                       'nwg93_wt': ['180221_NWG93_0PKC_Florent2'],
#                       'nwg93_rd': ['180221_NWG93_100PKC_Florent2',
#                                    '180221_NWG93_25PKC_Florent2',
#                                    '180221_NWG93_50PKC_Florent2']},
#               dest=dest + 'e1803__pkc_rundown_florent_exp2/',
#               mag=5 / 3,
#               dclass=Importers.Data2,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s8,
#               bounds=[0, 1],
#               segment='2').run()
#
# """
# e1803__pkc_rundown_florent_exp3
#
# """
#
# BatchAnalysis(direcs={'nwg91_rd': ['180301_NWG91_25PKC_Florent3',
#                                    '180301_NWG91_50PKC_Florent3',
#                                    '180301_NWG91_75PKC_Florent3']},
#               dest=dest + 'e1803__pkc_rundown_florent_exp3/',
#               mag=5 / 3,
#               dclass=Importers.Data2,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s9,
#               bounds=[0, 1],
#               segment='2').run()
#
# """
# e1804__par2_rundown_nelio
# Nelio's PAR-2 rundown data in NWG76 (March-April 2018)
#
#
# """
#
# BatchAnalysis(direcs={'nwg76_wt': ['180329_nwg76_wt_Nelio1',
#                                    '180330_nwg76_wt_Nelio1',
#                                    '180402_nwg76_wt_Nelio1',
#                                    '180403_nwg76_wt_Nelio1',
#                                    '180404_nwg76_wt_Nelio1'],
#                       'nwg76_rd': ['180329_nwg76_p2rundown_Nelio1',
#                                    '180330_nwg76_p2rundown_Nelio1',
#                                    '180402_nwg76_p2rundown_Nelio1',
#                                    '180403_nwg76_p2rundown_Nelio1',
#                                    '180404_nwg76_p2rundown_Nelio1']},
#               dest=dest + 'e1804__par2_rundown_nelio/',
#               mag=2,
#               dclass=Importers.Data0,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s4,
#               bounds=[0.9, 0.1],
#               segment='2').run()
#
"""
e1804__ph_rundown
PH-rundown control experiment (April 2018)

"""

BatchAnalysis(direcs={'ph_rd': ['180420_nwg1_0920xfp_tom4,5,30',
                                '180420_nwg1_1100xfp_tom4,5,30',
                                '180420_nwg1_1300xfp_tom4,5,30',
                                '180420_nwg1_1600(180419)xfp_tom4,5,30'],
                      'ph_wt': ['180420_nwg1_wt_tom4,5,30']},
              dest=dest + 'e1804__ph_rundown/',
              mag=1,
              dclass=Importers.Data0,
              bg_g=b.d['nwg42_g'],
              bg_a=b.d['nwg42_a'],
              bg_c=b.d['nwg42_c'],
              bg_r=b.d['nwg42_r'],
              settings=s.OD70s1,
              bounds=[0.9, 0.1],
              segment='2').run()
#
# """
# e1805__od70
#
# """
#
# BatchAnalysis(direcs={'od70_wt': ['180501_od70_wt_tom4,5,30',
#                                   '180420_od70_wt_tom4,5,30']},
#               dest=dest + 'e1805__od70/',
#               mag=1,
#               dclass=Importers.Data0,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s2,
#               bounds=[0, 1],
#               segment='2').run()
#
# """
# e1806__nwg91
# June 2018
# Playing with PAR-2/PKC double line in variety of conditions, looking for correlation between PAR-2 and PKC
#
#
# """
#
# BatchAnalysis(direcs={'nwg91_wt': ['180607_nwg91_wt_tom4,15,30'],
#                       'nwg91_chin1': ['180611_nwg91_24hrchin1_tom4,15,30',
#                                       '180612_nwg91_48hrchin1_tom4,15,30'],
#                       'nwg_par1': ['180611_nwg91_24hrpar1_tom4,15,30',
#                                    '180612_nwg91_48hrpar1_tom4,15,30'],
#                       'nwg_spd5': ['180611_nwg91_24hrspd5_tom4,15,30']},
#               dest=dest + 'e1806__nwg91/',
#               mag=1,
#               dclass=Importers.Data0,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s2,
#               bounds=[0, 1],
#               segment='2').run()
#
# """
# e1806__par2_mutants
# Comparision of cytoplasmic affinity of different PAR-2 mutants in PKC+/- conditions (June 2018)
#
#
# """
#
# BatchAnalysis(direcs={'kk1273_wt': ['180501_kk1273_wt_tom3,15+ph',
#                                     '180525_kk1273_wt_tom3,15'],
#                       'kk1273_par6': ['180618_kk1273_48hrpar6_tom3,15'],
#                       'nwg0123_wt': ['180302_nwg0123_24hr0par2,par3_tom3,15,pfsout',
#                                      '180322_nwg0123_par3_tom4,15,30,pfsout'],
#                       'nwg0062_wt': ['180509_nwg0062_wt_tom3,15,pfsout'],
#                       'nwg0062_par6': ['180618_nwg62_48hrpar6_tom3,15'],
#                       'jh1799_wt': ['180525_jh1799_wt_tom3,15',
#                                     '180606_jh1799_wt_tom3,15',
#                                     '180611_jh1799_wt_tom3,15'],
#                       'jh1799_par6': ['180606_jh1799_48hrpar6_tom3,15'],
#                       'jh1799_ctrl': ['180611_jh1799_48hrctrlrnai_tom3,15'],
#                       'jh2882_wt': ['180606_jh2882_wt_tom3,15'],
#                       'jh2882_par6': ['180606_jh2882_48hrpar6_tom3,15'],
#                       'jh2817_wt': ['180618_jh2817_wt_tom3,15'],
#                       'jh2817_par6': ['180618_jh2817_48hrpar6_tom3,15'],
#                       'th129_wt': ['180618_th129_wt_tom3,15'],
#                       'th129_par6 ': ['180618_th129_48hrpar6_tom3,15'],
#                       'nwg123_wt_bleach': ['180804_nwg0123_wt_tom4,15,30+bleach']},
#               dest=dest + 'e1806__par2_mutants/',
#               mag=1,
#               dclass=Importers.Data3,
#               bg_g=b.d['nwg129_g'],
#               bg_a=b.d['nwg129_a'],
#               bg_c=b.d['nwg129_c'],
#               settings=s.N2s2,
#               bounds=[0.9, 0.1],
#               segment='3').run()
#
# """
# e1806__par2_mutants2
# Comparision of cytoplasmic affinity of different PAR-2 mutants in PKC+/- conditions (June 2018)
# On SD after move to new room
#
# """
#
# BatchAnalysis(direcs={'kk1273_wt': ['180622_kk1273_wt_tom3,15'],
#                       'kk1273_pkc': ['180622_kk1273_48hrpkc_tom3,15'],
#                       'jh1799_wt': ['180622_jh1799_wt_tom3,15'],
#                       'jh1799_pkc': ['180622_jh1799_48hrpkc_tom3,15'],
#                       'jh2817_wt': ['180622_jh2817_wt_tom3,15'],
#                       'jh2817_pkc': ['180622_jh2817_48hrpkc_tom3,15'],
#                       'th129_wt': ['180622_th129_wt_tom3,15'],
#                       'th129_pkc': ['180622_th129_48hrpkc_tom3,15']},
#               dest=dest + 'e1806__par2_mutants2/',
#               mag=1,
#               dclass=Importers.Data3,
#               bg_g=b.d['nwg129_g'],
#               bg_a=b.d['nwg129_a'],
#               bg_c=b.d['nwg129_c'],
#               settings=s.N2s2,
#               bounds=[0.9, 0.1],
#               segment='3').run()
#
# """
# e1807__optogenetics
# Throws error
#
# """
#
# # BatchAnalysis(direcs=[direcslist('180712_sv2061_wt_tom4,15,30,100msecG')],
# #               dest=dest + 'e1807__optogenetics',
# #               mag=1,
# #               dclass=Importers.Data0,
# #               bgcurve=b.bgG4,
# #               settings=s.N2s0,
# #               bounds=[0, 1],
# #               segment='2').run()
#
# """
# e1807__par2_c1b
# Looking at nwg0145 line, aiming to understand if PAR-2 homodimerises
#
#
# """
#
# BatchAnalysis(direcs={'nwg0145_wt': ['180706_nwg0145_wt_tom4,15,30'],
#                       'nwg0145_pma': ['180710_nwg0145_perm1+pma_tom4,15,30']},
#               dest=dest + 'e1807__par2_c1b/',
#               mag=1,
#               dclass=Importers.Data0,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s2,
#               bounds=[0, 1],
#               segment='2').run()
#
# """
# e1807__par2_gbp
# PAR-2 GBP
#
#
# """
#
# BatchAnalysis(direcs={'nwg51_gbp': ['180730_nwg0151xnwg0143_wt_tom4,15,30'],
#                       'nwg51_gbp_bleach': ['180730_nwg0151xnwg0143_wt_tom4,15,30+bleach'],
#                       'nwg51_wt': ['180730_nwg0151_wt_tom4,15,30',
#                                    '180726_nwg0151_wt_tom4,15,30'],
#                       'nwg51_wt_bleach': ['180730_nwg0151_wt_tom4,15,30+bleach',
#                                           '180726_nwg0151_wt_tom4,15,30+bleach'],
#                       'nwg51_dr466': ['180804_nwg0151xdr466_wt_tom4,15,30'],
#                       'nwg51_dr466_bleach': ['180804_nwg0151xdr466_wt_tom4,15,30+bleach']},
#               dest=dest + 'e1807__par2_gbp/',
#               mag=1,
#               dclass=Importers.Data0,
#               bg_g=b.d['nwg42_g'],
#               bg_a=b.d['nwg42_a'],
#               bg_c=b.d['nwg42_c'],
#               bg_r=b.d['nwg42_r'],
#               settings=s.N2s10,
#               bounds=[0, 1],
#               segment='2').run()
# #
# """
# e1807__s241e
#
# """
#
# BatchAnalysis(direcs={'kk1273_wt': ['180726_kk1273_wt_tom4,15,30'],
#                       'nwg62_wt': ['180726_nw62_wt_tom4,15,30'],
#                       'nwg126_wt': ['180726_nwg0126_wt_tom4,15,30',
#                                     '180727_nwg0126_wt_tom4,15,30'],
#                       'kk1273_wt_bleach': ['180726_kk1273_wt_tom4,15,30+bleach'],
#                       'nwg62_wt_bleach': ['180726_nw62_wt_tom4,15,30+bleach'],
#                       'nwg126_wt_bleach': ['180726_nwg0126_wt_tom4,15,30+bleach',
#                                            '180727_nwg0126_wt_tom4,15,30+bleach']},
#               dest=dest + 'e1807__s241e/',
#               mag=1,
#               dclass=Importers.Data3,
#               bg_g=b.d['nwg129_g'],
#               bg_a=b.d['nwg129_a'],
#               bg_c=b.d['nwg129_c'],
#               settings=s.N2s2,
#               bounds=[0.9, 0.1],
#               segment='3').run()
#
# """
# e1808__par2_c1b_2
# Looking at nwg0158 line, aiming to understand if PAR-2 homodimerises
#
# Throws error
#
# """
#
# # BatchAnalysis(direcs=['180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e1',
# #                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e2',
# #                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e3',
# #                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e4',
# #                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/e5',
# #                       '180803_nwg0158_pmawashin,perm,dmso_tom4,15,30_MP',
# #                       '180726_nwg0158_wt_tom4,15,30',
# #                       '180730_nwg0151_wt_tom4,15,30',
# #                       '180726_nwg0151_wt_tom4,15,30'],
# #               dest=dest + 'e1808__par2_c1b_2/',
# #               mag=1,
# #               dclass=Importers.Data0,
# #               bgcurve=b.bgG4,
# #               settings=s.N2s2,
# #               bounds=[0, 1],
# #               segment='3').run()
