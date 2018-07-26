from Experiments.e1804__par2_rundown_nelio import *


####################################################################

# Cyt vs mem

# plt.scatter(nwg76_wt.cyts_GFP, nwg76_wt.corts_GFP,  s=5, facecolors='g', edgecolors='g', linewidth=1)
# plt.scatter(nwg76_rd.cyts_GFP, nwg76_rd.corts_GFP, s=5, facecolors='none', edgecolors='g', linewidth=1)
# plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
# plt.ylabel('[Cortical PAR-2] (a.u.)')
#
# sns.despine()
# plt.show()


####################################################################

# Cortex vs ratio

# ax4 = plt.subplot2grid((1, 1), (0, 0))
# ax4.scatter(nwg76_wt.corts_GFP, nwg76_wt.corts_GFP / nwg76_wt.cyts_GFP,  s=5, facecolors='none', edgecolors='r', linewidth=1, label='wt')
# ax4.scatter(nwg76_rd.corts_GFP, nwg76_rd.corts_GFP / nwg76_rd.cyts_GFP, s=5, facecolors='none', edgecolors='b', linewidth=1, label='par-3(it71)')
# # ax4.scatter(nwg0123_wt.corts, nwg0123_wt.corts / nwg0123_wt.cyts,  s=5, facecolors='none', edgecolors='b', linewidth=1)
#
# sns.despine()
# plt.rcParams.update({'font.size': 10})
# plt.ylim([0, 35])
# plt.show()


####################################################################

# Dosage correlation

def func(dataset, c):
    plt.scatter(dataset.totals_GFP, dataset.totals_RFP, c=c, s=10)


# func(nwg76_wt, 'r')
# func(nwg76_rd, 'b')
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.xlabel('Total PAR2')
# plt.ylabel('Total PAR6')
# plt.show()


####################################################################

# Spatial distribution

def func2(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2, linestyle='-')
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c=c, alpha=0.2, linestyle='--')

    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c, linestyle='-')
    plt.plot(np.mean(dataset.rfp_spatial, 0) / np.mean(dataset.cyts_RFP), c=c, linestyle='--')


# func2(nwg76_wt, c='r')
# plt.show()


####################################################################

# Cyt vs mem


def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.cyts_GFP[x], np.mean(dataset.gfp_spatial[x, dataset.rfp_spatial[x, :] < 500]),  s=5, c=c)


# func(nwg76_wt, 'r')
# func(nwg76_rd, 'b')
# plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
# plt.ylabel('[Cortical PAR-2] (a.u.)')
# sns.despine()
# plt.show()



####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.rfp_spatial[x, :], dataset.gfp_spatial[x, :], s=0.2, c=c)


# func(nwg76_wt, c='k')
# func(nwg76_rd, c='r')
# plt.xlabel('Cortical PAR-6')
# plt.ylabel('Cortical PAR-2')
# sns.despine()
# plt.show()


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.cyts_RFP[x] / dataset.rfp_spatial[x, :], s=0.2, c=c)


# func(nwg76_wt, c='k')
# func(nwg76_rd, c='r')
# plt.ylim([0, 10])
# plt.xlabel('Cortical PAR-2')
# plt.ylabel('Cytoplasmic / Cortical PAR-6')
# sns.despine()
# plt.show()


####################################################################

# PAR-2 in posterior vs PAR-6 in anterior


def func7(dataset):
    for x in range(len(dataset.gfp_spatial[:, 0])):

        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Posterior pPAR vs anterior aPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds2), bounded_mean(dataset.rfp_spatial[x, :], bounds1), c='k', s=10)

        # Posterior pPAR vs posterior aPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds2), bounded_mean(dataset.rfp_spatial[x, :], bounds2), c='b', s=10)


# func7(nwg76_wt)
# func7(nwg76_rd)
# # plt.gca().set_ylim(top=15000)
# sns.despine()
# plt.show()





