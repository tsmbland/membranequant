from Experiments.e1803__par2par1_rundown import *


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.rfp_spatial[x, :], dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], s=0.2, c=c)


# func(nwg0132_wt, c='k')
# func(nwg0132_rd, c='k')
# plt.xlabel('Cortical PAR-2')
# plt.ylabel('Cortical / Cytoplasmic PAR-1 ')
# sns.despine()
# plt.show()


def func3(dataset, c, f):
    plt.scatter(dataset.corts_RFP, dataset.corts_GFP / dataset.cyts_GFP, s=5, facecolors=f, edgecolors=c, linewidth=1)


# func3(nwg42_wt, 'r', f='r')
# func3(nwg0132_wt, 'b', f='b')
# func3(nwg0132_rd, 'b', f='none')
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.xlabel('[Cortical PAR-2] (a.u.)')
# plt.ylabel('Cortical / Cytoplasmic PAR1 (a.u.)')
# sns.despine()
# plt.show()


####################################################################

# Spatial distribution

def func2(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2, linestyle='-')
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c=c, alpha=0.2, linestyle='--')

    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c, linestyle='-')
    plt.plot(np.mean(dataset.rfp_spatial, 0) / np.mean(dataset.cyts_RFP), c=c, linestyle='--')


# func2(nwg42_wt, c='r')
# func2(nwg0132_wt, c='b')
# # func2(nwg0132_rd, c='g')
# plt.show()


####################################################################

# Dosage correlation

# def func4(dataset, c, f):
#     plt.scatter(dataset.totals_RFP, dataset.totals_GFP, edgecolors=c, facecolors=f, s=5, linewidth=1)
#
#
# func4(nwg42_wt, 'r', f='r')
# func4(nwg0132_wt, 'b', f='b')
# func4(nwg0132_rd, 'b', f='none')
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.xlabel('Total PAR2 (a.u.)')
# plt.ylabel('Total PAR1 (a.u.)')
# sns.despine()
# plt.show()


####################################################################

# PAR2 membrane binding

# Cyt vs mem

# ax4 = plt.subplot2grid((1, 1), (0, 0))
# ax4.scatter(nwg42_wt.cyts_RFP, nwg42_wt.corts_RFP,  s=5, facecolors='none', edgecolors='r', linewidth=1, label='wt')
# ax4.scatter(nwg0132_wt.cyts_RFP, nwg0132_wt.corts_RFP, s=5, facecolors='none', edgecolors='b', linewidth=1, label='par-3(it71)')
# ax4.scatter(nwg0132_rd.cyts_RFP, nwg0132_rd.corts_RFP,  s=5, facecolors='none', edgecolors='b', linewidth=1)
# ax4.set_xlabel('[Cytoplasmic PAR-2] (a.u.)')
# ax4.set_ylabel('[Cortical PAR-2] (a.u.)')
#
# sns.despine()
# plt.rcParams.update({'font.size': 10})
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.show()
