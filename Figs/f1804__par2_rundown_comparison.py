from IA import *
import Experiments.e1803__par2_rundown as e1
import Experiments.e1804__par2_rundown_nelio as e2

####################################################################

# Normalise data

nwg0123_wt = normalise(e1.nwg0123_wt, e1.kk1273_wt, ['corts_GFP', 'cyts_GFP', 'totals_GFP'])
nwg0123_rd = normalise(e1.nwg0123_rd, e1.kk1273_wt, ['corts_GFP', 'cyts_GFP', 'totals_GFP'])
kk1273_wt = normalise(e1.kk1273_wt, e1.kk1273_wt, ['corts_GFP', 'cyts_GFP', 'totals_GFP'])

nwg76_rd = normalise(e2.nwg76_rd, e2.nwg76_wt, ['corts_GFP', 'cyts_GFP', 'totals_GFP'])
nwg76_wt = normalise(e2.nwg76_wt, e2.nwg76_wt, ['corts_GFP', 'cyts_GFP', 'totals_GFP'])


####################################################################

# Cyt vs mem

def func(dataset, c, f):
    plt.scatter(dataset.cyts_GFP, dataset.corts_GFP, s=5, facecolors=f, edgecolors=c, linewidth=1)


# func(nwg0123_wt, c='b', f='b')
# func(nwg0123_rd, c='b', f='none')
# func(kk1273_wt, c='r', f='r')
# func(nwg76_wt, c='g', f='g')
# func(nwg76_rd, c='g', f='none')
#
# plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
# plt.ylabel('[Cortical PAR-2] (a.u.)')
#
# sns.despine()
# plt.rcParams.update({'font.size': 10})
# plt.show()


####################################################################

# Cortex vs ratio

def func(dataset, c):
    plt.scatter(dataset.corts_GFP, dataset.corts_GFP / dataset.cyts_GFP, s=5, facecolors='none', edgecolors=c,
                linewidth=1)


# func(nwg0123_wt, c='k')
# func(nwg0123_rd, c='k')
# func(kk1273_wt, c='k')
# func(nwg76_wt, c='r')
# func(nwg76_rd, c='r')
#
# sns.despine()
# plt.rcParams.update({'font.size': 10})
# plt.xlim([0, 1.2])
# plt.ylim([0, 1.2])
# plt.show()
