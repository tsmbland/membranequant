import Experiments.e1803__pkc_rundown_florent_exp1 as exp1
import Experiments.e1803__pkc_rundown_florent_exp2 as exp2
import Experiments.e1803__pkc_rundown_florent_exp3 as exp3
from IA import *


"""
Exp1 quantitatively not matching up with others

"""

################################################################

# Spatial distribution

"""
Plots the cortical/cytoplasm intensities around the circumference. Separate functions for GFP and mCherry

"""


def func1(dataset, c='k', linestyle='-'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2, linestyle=linestyle)
    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c, linestyle=linestyle)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex / Cytoplasm (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


def func2(dataset, c='k', linestyle='-'):
    for x in range(len(dataset.rfp_spatial[:, 0])):
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c=c, alpha=0.2, linestyle=linestyle)
    plt.plot(np.mean(dataset.rfp_spatial, 0) / np.mean(dataset.cyts_RFP), c=c, linestyle=linestyle)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex / Cytoplasm (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


# func1(exp1.nwg91_wt, c='g', linestyle='-')
# func2(exp1.nwg91_wt, c='r', linestyle='-')
# func1(exp1.nwg93_wt, c='g', linestyle='--')
# func2(exp1.nwg93_wt, c='r', linestyle='--')
# plt.show()


###############################################################

# Spatial distribution with stdev

def func1(dataset, c):
    gfp_spatial = np.concatenate((dataset.gfp_spatial[:, :50], np.flip(dataset.gfp_spatial[:, 50:], 1)))
    cyts_GFP = np.append(dataset.cyts_GFP, dataset.cyts_GFP)
    mean = np.mean(gfp_spatial, 0) / np.mean(cyts_GFP)
    stdev = np.std(gfp_spatial / np.tile(cyts_GFP, (50, 1)).T, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


def func2(dataset, c):
    rfp_spatial = np.concatenate((dataset.rfp_spatial[:, :50], np.flip(dataset.rfp_spatial[:, 50:], 1)))
    cyts_RFP = np.append(dataset.cyts_RFP, dataset.cyts_RFP)
    mean = np.mean(rfp_spatial, 0) / np.mean(cyts_RFP)
    stdev = np.std(rfp_spatial / np.tile(cyts_RFP, (50, 1)).T, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


func1(exp2.nwg91_wt, c='g')
func2(exp2.nwg91_wt, c='r')

func1(exp2.nwg93_wt, c='b')
func2(exp2.nwg93_wt, c='k')
plt.show()




################################################################

# Correlation (cortex, loc by loc)

"""

Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset, c='k'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.rfp_spatial[x, :], s=0.5, c=c)


# func(exp1.nwg93_wt, c='r')
# func(exp1.nwg93_rd, c='r')
# func(exp1.nwg91_wt, c='g')
# func(exp1.nwg91_rd, c='g')
# # plt.gca().set_xlim(left=-5000)
# sns.despine()
# plt.xlabel('Cortical PKC-3')
# plt.ylabel('Cortical PAR-2')
# plt.show()



################################################################

# PKC antagonism

"""

Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset, c='k'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.cyts_RFP[x] / dataset.rfp_spatial[x, :], s=0.2, c=c)


# func(exp1.nwg93_wt, c='r')
# func(exp1.nwg93_rd, c='r')
# func(exp1.nwg91_wt, c='g')
# func(exp1.nwg91_rd, c='g')
# # plt.gca().set_xlim(left=-5000)
# # plt.ylim([0, 3])
# sns.despine()
# plt.xlabel('Cortical PKC-3')
# plt.ylabel('Cytoplasmic / Cortical PAR-2')
# plt.yscale('log')
# plt.show()



####################################################################

# PAR-6 dosage vs PAR-2 mem:cyt

def func5(dataset, c):
    plt.scatter(dataset.totals_GFP, dataset.corts_RFP / dataset.cyts_RFP, c=c, s=10)


# func5(exp2.nwg93_wt, c='r')
# func5(exp2.nwg93_rd, c='r')
# func5(exp2.nwg91_wt, c='g')
# func5(exp2.nwg91_rd, c='g')
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.xlabel('Total PKC-3')
# plt.ylabel('Posterior PAR2 mem:cyt')
# sns.despine()
# plt.show()



################################################################


def func(dataset, c='k'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.totals_GFP[x],
                    np.mean(dataset.rfp_spatial[x, dataset.gfp_spatial[x, :] < 5000]) / dataset.cyts_RFP[x], s=10, c=c)


# func(exp2.nwg93_wt, c='r')
# func(exp2.nwg93_rd, c='r')
# func(exp2.nwg91_wt, c='g')
# func(exp2.nwg91_rd, c='g')
# sns.despine()
# plt.xlabel('Cortical PKC-3')
# plt.ylabel('Cytoplasmic / Cortical PAR-2')
# plt.gca().set_ylim(bottom=0)
# plt.show()


################################################################


def func7(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Anterior aPAR vs anterior pPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds1),
                    bounded_mean(dataset.rfp_spatial[x, :], bounds1) / dataset.cyts_RFP[x], c=c, s=10)

        # Anterior aPAR vs posterior pPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds1),
                    bounded_mean(dataset.rfp_spatial[x, :], bounds2) / dataset.cyts_RFP[x], c=c, s=5)


# func7(exp1.nwg91_wt, c='r')
# func7(exp1.nwg91_rd, c='r')
# func7(exp1.nwg93_wt, c='k')
# func7(exp1.nwg93_rd, c='k')
# sns.despine()
# plt.xlabel('Cortical PKC-3')
# plt.ylabel('Cortical / Cytoplasmic PAR-2')
# # plt.gca().set_ylim(bottom=0)
# plt.show()



################################################################


def func8(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Anterior aPAR vs anterior pPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds1),
                    dataset.cyts_RFP[x] / bounded_mean(dataset.rfp_spatial[x, :], bounds1), c=c, s=10)

        # Posterior aPAR vs posterior pPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds2),
                    dataset.cyts_RFP[x] / bounded_mean(dataset.rfp_spatial[x, :], bounds2), c=c, s=10)


# func8(exp1.nwg91_wt, c='r')
# func8(exp1.nwg91_rd, c='r')
# func8(exp1.nwg93_wt, c='k')
# func8(exp1.nwg93_rd, c='k')
# sns.despine()
# plt.xlabel('Cortical PKC-3')
# plt.ylabel('Cortical / Cytoplasmic PAR-2')
# # plt.gca().set_ylim(bottom=0)
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
                    bounded_mean(dataset.gfp_spatial[x, :], bounds1), s=5 * bounded_mean(dataset.rfp_spatial[x, :], bounds1) / dataset.cyts_RFP[x], c='k')

        # Total aPAR vs posterior aPAR
        plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds0),
                    bounded_mean(dataset.gfp_spatial[x, :], bounds2), s=5 * bounded_mean(dataset.rfp_spatial[x, :], bounds2) / dataset.cyts_RFP[x], c='k')


# func9(exp1.nwg91_wt, c='r')
# func9(exp1.nwg91_rd, c='r')
# func9(exp1.nwg93_wt, c='k')
# func9(exp1.nwg93_rd, c='k')
# sns.despine()
# plt.xlabel('Total PKC-3')
# plt.ylabel('Local PKC-3')
# plt.show()
