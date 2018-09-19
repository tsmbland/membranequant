import Experiments.e1803__pkc_rundown_florent_exp1 as exp1
import Experiments.e1803__pkc_rundown_florent_exp2 as exp2
import Experiments.e1803__pkc_rundown_florent_exp3 as exp3
from IA import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0

################################################################

# Spatial distribution

"""
Plots the cortical/cytoplasm intensities around the circumference. Separate functions for GFP and mCherry

"""


def func1(dataset, c='k', linestyle='-'):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.g_spa[x, :] / dataset.g_cyt[x], c=c, alpha=0.2, linestyle=linestyle)
    plt.plot(np.mean(dataset.g_spa, 0) / np.mean(dataset.g_cyt), c=c, linestyle=linestyle)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex / Cytoplasm (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


def func2(dataset, c='k', linestyle='-'):
    for x in range(len(dataset.r_spa[:, 0])):
        plt.plot(dataset.r_spa[x, :] / dataset.r_cyt[x], c=c, alpha=0.2, linestyle=linestyle)
    plt.plot(np.mean(dataset.r_spa, 0) / np.mean(dataset.r_cyt), c=c, linestyle=linestyle)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex / Cytoplasm (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


f += 1
plt.close()
func1(exp1.nwg91_wt, c='g', linestyle='-')
func2(exp1.nwg91_wt, c='r', linestyle='-')
func1(exp1.nwg93_wt, c='g', linestyle='--')
func2(exp1.nwg93_wt, c='r', linestyle='--')
plt.savefig('%s/f%s.png' % (fdirec, f))


###############################################################

# Spatial distribution with stdev, a to p, normalised

def func1(dataset, c, a=[0, 0], l='-'):
    gfp_spatial = np.concatenate((np.flip(dataset.g_spa[:, :50], 1), dataset.g_spa[:, 50:]))
    mean = np.mean(gfp_spatial, 0)
    if a[0] == 0:
        a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    gfp_spatial = a[0] * gfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(gfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c, linestyle=l)
    return a


def func2(dataset, c, a=[0, 0], l='-'):
    rfp_spatial = np.concatenate((np.flip(dataset.r_spa[:, :50], 1), dataset.r_spa[:, 50:]))
    mean = np.mean(rfp_spatial, 0)
    if a[0] == 0:
        a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    rfp_spatial = a[0] * rfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(rfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c, linestyle=l)
    return a


f += 1
plt.close()
a1 = func1(exp2.nwg91_wt, c='r')
a2 = func2(exp2.nwg91_wt, c='c')
# func1(exp2.nwg93_wt, c='r', a=a1, l='--')
# func2(exp2.nwg93_wt, c='c', a=a2, l='--')
sns.despine()
plt.xticks([])
plt.yticks([0, 1])
# plt.xlabel('x / circumference')
# plt.ylabel('Cortex / Cytoplasm (a.u.)')
plt.savefig('%s/f%s.png' % (fdirec, f))

################################################################

# Correlation (cortex, loc by loc)

"""

Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset, c='k'):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_spa[x, :], dataset.r_spa[x, :] / dataset.r_cyt[x], s=1.0, c=c)


f += 1
plt.close()
func(exp1.nwg93_wt, c='r')
func(exp1.nwg93_rd, c='r')
func(exp1.nwg91_wt, c='k')
func(exp1.nwg91_rd, c='k')
# plt.gca().set_xlim(left=-5000)
sns.despine()
# plt.xlabel('Cortical PKC-3')
# plt.ylabel('Cortical PAR-2')
plt.savefig('%s/f%s.png' % (fdirec, f))

################################################################

# PKC antagonism

"""

Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset, c='k'):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_spa[x, :], dataset.r_cyt[x] / dataset.r_spa[x, :], s=1, c=c)


f += 1
plt.close()
func(exp1.nwg93_wt, c='r')
func(exp1.nwg93_rd, c='r')
func(exp1.nwg91_wt, c='g')
func(exp1.nwg91_rd, c='g')
# plt.gca().set_xlim(left=-5000)
plt.ylim([0, 3])
sns.despine()
plt.xlabel('Cortical PKC-3')
plt.ylabel('Cytoplasmic / Cortical PAR-2')
# plt.yscale('log')
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# PAR-6 dosage vs PAR-2 mem:cyt

def func5(dataset, c):
    plt.scatter(dataset.g_tot, dataset.r_mem / dataset.r_cyt, c=c, s=10)


f += 1
plt.close()
func5(exp2.nwg93_wt, c='r')
func5(exp2.nwg93_rd, c='r')
func5(exp2.nwg91_wt, c='g')
func5(exp2.nwg91_rd, c='g')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.xlabel('Total PKC-3')
plt.ylabel('Posterior PAR2 mem:cyt')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


################################################################


def func(dataset, c='k'):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_tot[x],
                    np.mean(dataset.r_spa[x, dataset.g_spa[x, :] < 5000]) / dataset.r_cyt[x], s=10, c=c)


f += 1
plt.close()
func(exp2.nwg93_wt, c='r')
func(exp2.nwg93_rd, c='r')
func(exp2.nwg91_wt, c='g')
func(exp2.nwg91_rd, c='g')
sns.despine()
plt.xlabel('Cortical PKC-3')
plt.ylabel('Cytoplasmic / Cortical PAR-2')
plt.gca().set_ylim(bottom=0)
plt.savefig('%s/f%s.png' % (fdirec, f))


################################################################


def func7(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Anterior aPAR vs anterior pPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds1),
                    bounded_mean(dataset.r_spa[x, :], bounds1) / dataset.r_cyt[x], c=c, s=10)

        # # Anterior aPAR vs posterior pPAR
        # plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds1),
        #             bounded_mean(dataset.rfp_spatial[x, :], bounds2) / dataset.cyts_RFP[x], c=c, s=5)


f += 1
plt.close()
func7(exp1.nwg91_wt, c='r')
func7(exp1.nwg91_rd, c='r')
func7(exp1.nwg93_wt, c='k')
func7(exp1.nwg93_rd, c='k')
sns.despine()
plt.xlabel('Cortical PKC-3')
plt.ylabel('Cortical / Cytoplasmic PAR-2')
# plt.gca().set_ylim(bottom=0)
plt.savefig('%s/f%s.png' % (fdirec, f))


################################################################


def func8(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Anterior aPAR vs anterior pPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds1),
                    dataset.r_cyt[x] / bounded_mean(dataset.r_spa[x, :], bounds1), c=c, s=10)

        # Posterior aPAR vs posterior pPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds2),
                    dataset.r_cyt[x] / bounded_mean(dataset.r_spa[x, :], bounds2), c=c, s=10)


f += 1
plt.close()
func8(exp1.nwg91_wt, c='r')
func8(exp1.nwg91_rd, c='r')
func8(exp1.nwg93_wt, c='k')
func8(exp1.nwg93_rd, c='k')
sns.despine()
plt.xlabel('Cortical PKC-3')
plt.ylabel('Cortical / Cytoplasmic PAR-2')
# plt.gca().set_ylim(bottom=0)
plt.savefig('%s/f%s.png' % (fdirec, f))


################################################################

# Mean PKC vs Local PKC

def func9(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        bounds0 = (0, 1)
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Total aPAR vs anterior aPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds0),
                    bounded_mean(dataset.g_spa[x, :], bounds1),
                    s=5 * bounded_mean(dataset.r_spa[x, :], bounds1) / dataset.r_cyt[x], c='k')

        # Total aPAR vs posterior aPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds0),
                    bounded_mean(dataset.g_spa[x, :], bounds2),
                    s=5 * bounded_mean(dataset.r_spa[x, :], bounds2) / dataset.r_cyt[x], c='k')


f += 1
plt.close()
func9(exp1.nwg91_wt, c='r')
func9(exp1.nwg91_rd, c='r')
func9(exp1.nwg93_wt, c='k')
func9(exp1.nwg93_rd, c='k')
sns.despine()
plt.xlabel('Total PKC-3')
plt.ylabel('Local PKC-3')
plt.savefig('%s/f%s.png' % (fdirec, f))
