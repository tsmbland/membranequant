from Experiments.e1802__par6_rundown_nelio import *


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.rfp_spatial[x, :], s=0.5, c=c)


# func(nwg26_wt, c='k')
# func(nwg26_rd, c='r')
# plt.xlabel('Cortical PAR-6')
# plt.ylabel('Cortical PAR-2')
# sns.despine()
# plt.show()


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.cyts_RFP[x] / dataset.rfp_spatial[x, :], s=0.5, c=c)


# func(nwg26_wt, c='k')
# func(nwg26_rd, c='k')
# # plt.ylim([0, 10])
# plt.xlabel('Cortical PAR-6')
# plt.ylabel('Cytoplasmic / Cortical PAR-2')
# sns.despine()
# plt.yscale('log')
# plt.show()


####################################################################

# Spatial distribution

def func2(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2, linestyle='-')
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c=c, alpha=0.2, linestyle='--')

    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c, linestyle='-')
    plt.plot(np.mean(dataset.rfp_spatial, 0) / np.mean(dataset.cyts_RFP), c=c, linestyle='--')


# func2(nwg26_wt, c='r')
# plt.show()


####################################################################

# Dosage correlation

def func4(dataset, c):
    plt.scatter(dataset.totals_GFP, dataset.totals_RFP, c=c, s=10)


# func4(nwg26_wt, 'r')
# func4(nwg26_rd, 'b')
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.xlabel('Total PAR6')
# plt.ylabel('Total PAR2')
# plt.show()


####################################################################

# PAR-6 dosage vs PAR-2 mem:cyt

def func5(dataset, c):
    plt.scatter(dataset.totals_GFP, dataset.corts_RFP / dataset.cyts_RFP, c=c, s=10)


# func5(nwg26_wt, 'r')
# func5(nwg26_rd, 'b')
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.xlabel('Total PAR6')
# plt.ylabel('PAR2 mem:cyt')
# sns.despine()
# plt.show()


####################################################################

# PAR-2 membrane binding


def func5(dataset, c):
    plt.scatter(dataset.cyts_RFP, dataset.corts_RFP, c=c, s=10)


# func5(nwg26_wt, 'r')
# func5(nwg26_rd, 'b')
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.xlabel('Cytoplasmic PAR-2')
# plt.ylabel('Cortical PAR-2')
# plt.show()


####################################################################


def func(dataset, c='k'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.totals_GFP[x],
                    np.mean(dataset.rfp_spatial[x, dataset.gfp_spatial[x, :] < 1000]) / dataset.cyts_RFP[x], s=10, c=c)


# func(nwg26_wt, 'r')
# func(nwg26_rd, 'b')
# sns.despine()
# plt.ylabel('Mean Cortical / Cytoplasmic PAR-2 in regions without PAR-6')
# plt.xlabel('Total PAR-6')
# plt.gca().set_ylim(bottom=0)
# plt.show()


####################################################################


def func7(dataset):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Anterior aPAR vs anterior pPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds1),
                    dataset.cyts_RFP[x] / bounded_mean(dataset.rfp_spatial[x, :], bounds1), c='k', s=10)

        # # Anterior aPAR vs posterior pPAR
        # plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds1),
        #             bounded_mean(dataset.rfp_spatial[x, :], bounds2) / dataset.cyts_RFP[x], c='b', s=10)


# func7(nwg26_wt)
# func7(nwg26_rd)
# # plt.gca().set_ylim(top=15000)
# plt.yscale('log')
# sns.despine()
# plt.show()


################################################################

# Mean PKC vs Local PKC

def func9(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        bounds0 = (0, 1)
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Total aPAR vs anterior aPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds0),
                    bounded_mean(dataset.gfp_spatial[x, :], bounds1), s=10,
                    c=bounded_mean(dataset.rfp_spatial[x, :], bounds1) / dataset.cyts_RFP[x], vmin=0, vmax=20,
                    cmap='winter')

        # Total aPAR vs posterior aPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds0),
                    bounded_mean(dataset.gfp_spatial[x, :], bounds2), s=10,
                    c=bounded_mean(dataset.rfp_spatial[x, :], bounds2) / dataset.cyts_RFP[x], vmin=0, vmax=20,
                    cmap='winter')


# func9(nwg26_wt, c='k')
# func9(nwg26_rd, c='k')
# sns.despine()
# plt.xlabel('Total PKC-3')
# plt.ylabel('Local PKC-3')
# plt.show()
