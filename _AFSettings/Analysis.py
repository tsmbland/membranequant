from IA import *

dest = adirec + '/AFSettings/'


class BatchAnalysis:
    def __init__(self, direcs, dest, mag, dclass, setup=True, parallel=True):
        self.direcs = direcs
        self.dest = dest
        self.mag = mag
        self.dclass = dclass
        self.setup = setup
        self.parallel = parallel

    def run(self):

        # Setup
        if self.setup:
            copy_data(self.direcs, self.dest)
        embryos_list_total = embryos_direcslist(append_batch(self.dest, self.direcs))

        # Analysis
        if self.parallel:
            Parallel(n_jobs=multiprocessing.cpu_count(), verbose=50)(
                delayed(self.f_analysis)(e) for e in embryos_list_total)

        else:
            for e in embryos_list_total:
                self.f_analysis(e)
                print(e)

    def f_analysis(self, e):
        data = self.dclass(e)
        coors = data.ROI
        if data.GFP is not None:
            Analyser(data.GFP, coors, funcs=['cyt', 'ext'], direc=data.direc, name='g').run()
            Analyser(data.AF, coors, funcs=['cyt', 'ext'], direc=data.direc, name='a').run()
        if data.RFP is not None:
            Analyser(data.RFP, coors, funcs=['cyt', 'ext'], direc=data.direc, name='r').run()


"""
N2s1
N2, Tom3, 15, pfsin

"""
BatchAnalysis(direcs=['180227_n2_wt_tom3,15,pfsin',
                      '180417_n2_wt_tom3,15,pfsin'],
              dest=dest + 'N2s1/',
              mag=1,
              dclass=Importers.Data3).run()

"""
N2s2
N2, Tom3, 15, pfsout

"""
BatchAnalysis(direcs=['180302_n2_wt_tom3,15,pfsout',
                      '180412_n2_wt_tom3,15,pfsout'],
              dest=dest + 'N2s2/',
              mag=1,
              dclass=Importers.Data3).run()

"""
N2s3
N2, Tom3, 5, pfsout

"""
BatchAnalysis(direcs=['180417_n2_wt_tom3,5,pfsout'],
              dest=dest + 'N2s3/',
              mag=1,
              dclass=Importers.Data3).run()

"""
N2s4
N2, PAR2 Nelio

"""
BatchAnalysis(direcs=['180404_n2_wt_Nelio1'],
              dest=dest + 'N2s4/',
              mag=1,
              dclass=Importers.Data0).run()

"""
N2s5
N2, PAR6 Nelio

"""
BatchAnalysis(direcs=['180218_N2_wt_Nelio2'],
              dest=dest + 'N2s5/',
              mag=1,
              dclass=Importers.Data0).run()

"""
N2s7
N2, Florent 180110

"""
BatchAnalysis(direcs=['180110_N2_wt_Florent1'],
              dest=dest + 'N2s7/',
              mag=5 / 3,
              dclass=Importers.Data4).run()

"""
N2s8
N2, Florent 180221

"""
BatchAnalysis(direcs=['180221_N2_wt_Florent2'],
              dest=dest + 'N2s8/',
              mag=5 / 3,
              dclass=Importers.Data4).run()

"""
N2s9
N2, Florent 180301

"""
BatchAnalysis(direcs=['180301_N2_wt_Florent3'],
              dest=dest + 'N2s9/',
              mag=5 / 3,
              dclass=Importers.Data5).run()

"""
N2s10
N2, Tom4, 15, 30, pfsout, after microscope move

"""
BatchAnalysis(direcs=['180730_n2_wt_tom4,15,30'],
              dest=dest + 'N2s10/',
              mag=1,
              dclass=Importers.Data0).run()

"""
N2s11
N2, Tom4, 15, 30, pfsout, after microscope move, with bleach

"""
BatchAnalysis(direcs=['180730_n2_wt_tom4,15,30+bleach'],
              dest=dest + 'N2s11/',
              mag=1,
              dclass=Importers.Data0).run()

"""
OD70s1
OD70, Tom5, 30, pfsout

"""
BatchAnalysis(direcs=['180501_od70_wt_tom4,5,30',
                      '180420_od70_wt_tom4,5,30'],
              dest=dest + 'OD70s1/',
              mag=1,
              dclass=Importers.Data0).run()

"""
KK1254_0

"""


"""
Box241_0

"""