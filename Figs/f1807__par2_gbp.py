from IA import *
from Experiments.e1807__par2_gbp import *

####################################################################

# Spatial distribution

"""
Plots the cortical/cytoplasm intensities around the circumference. Separate functions for GFP and mCherry

"""


def func1(dataset, c='k'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex / Cytoplasm (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


def func2(dataset, c='k'):
    for x in range(len(dataset.rfp_spatial[:, 0])):
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.rfp_spatial, 0) / np.mean(dataset.cyts_RFP), c=c)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex / Cytoplasm (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


# func1(nwg51_gbp, c='g')
# func2(nwg51_gbp, c='r')
# func1(nwg51_dr466, c='b')
# func2(nwg51_dr466, c='k')
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


# func1(nwg51_gbp, c='g')
# func2(nwg51_gbp, c='r')
# plt.show()
#
# func1(nwg51_wt, c='g')
# func2(nwg51_wt, c='r')
# plt.show()
#
# func1(nwg51_dr466, c='g')
# func2(nwg51_dr466, c='r')
# plt.show()


###############################################################

# Spatial distribution with stdev, normalised

def func1(dataset, c):
    gfp_spatial = np.concatenate((np.flip(dataset.gfp_spatial[:, :50], 1), dataset.gfp_spatial[:, 50:]))
    mean = np.mean(gfp_spatial, 0)
    a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    print(a)
    gfp_spatial = a[0] * gfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(gfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


def func2(dataset, c):
    rfp_spatial = np.concatenate((np.flip(dataset.rfp_spatial[:, :50], 1), dataset.rfp_spatial[:, 50:]))
    mean = np.mean(rfp_spatial, 0)
    a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    print(a)
    rfp_spatial = a[0] * rfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(rfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


def func3(dataset, c):
    gfp_spatial = np.concatenate((np.flip(dataset.gfp_spatial[:, :50], 1), dataset.gfp_spatial[:, 50:]))
    mean = np.mean(gfp_spatial, 0)
    a = [8.44113621e-06,  -1.63550640e-02]
    gfp_spatial = a[0] * gfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(gfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


def func4(dataset, c):
    rfp_spatial = np.concatenate((np.flip(dataset.rfp_spatial[:, :50], 1), dataset.rfp_spatial[:, 50:]))
    mean = np.mean(rfp_spatial, 0)
    a = [0.00011719,  0.0480236]
    rfp_spatial = a[0] * rfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(rfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


# func1(nwg51_dr466, c='g')
# func2(nwg51_dr466, c='r')
# sns.despine()
# plt.xticks([])
# plt.yticks([])
# plt.ylim([-0.2, 1.2])
# plt.show()
#
#
# func3(nwg51_gbp, c='g')
# func4(nwg51_gbp, c='r')
# sns.despine()
# plt.xticks([])
# plt.yticks([])
# plt.ylim([-0.2, 1.2])
# plt.show()




####################################################################


# Correlation (cortex, loc by loc)

"""
Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x],
                    s=0.1, c=c)


# func(nwg51_gbp, 'k')
# func(nwg51_wt, 'r')
# plt.show()

####################################################################

# Dosage correlation

"""
Plots the correlation between the dosage of G and R proteins

"""


def func(dataset, c):
    plt.scatter(dataset.totals_GFP, dataset.totals_RFP, c=c, s=10)
    # plt.gca().set_ylim(bottom=0)
    # plt.gca().set_xlim(left=0)
    plt.xlabel('Total GFP')
    plt.ylabel('Total mCherry')


# func(nwg51_gbp, c='k')
# func(nwg51_dr466, c='r')
# func(nwg51_wt, c='g')
# plt.show()


####################################################################

# ASI

def bar(ax, data, pos, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)
    ax.set_xticklabels([])
    ax.set_xticks([])


def tidy(axes, labels, positions):
    for a in axes:
        a.set_xticks(positions)
        a.set_xticklabels(labels, fontsize=10)


def func8(concs):
    a = np.zeros([len(concs[:, 0])])
    for x in range(len(concs[:, 0])):
        a[x] = - asi(concs[x, :])
    return a


# ax = plt.subplot2grid((1, 1), (0, 0))
# # ax.set_ylabel('ASI')
# bar(ax, func8(nwg51_dr466.gfp_spatial), 4, 'g')
# bar(ax, func8(nwg51_dr466.rfp_spatial), 8, 'r')
# bar(ax, func8(nwg51_gbp.gfp_spatial), 12, 'g')
# bar(ax, func8(nwg51_gbp.rfp_spatial), 16, 'r')
# # tidy([ax], ['GFP', 'mCherry', 'GFP', 'mCherry'], [4, 8, 12, 16])
# ax.set_xticks([])
# sns.despine()
# plt.show()
#
# # Tests for significance
# from scipy.stats import ttest_ind
#
# t, p = ttest_ind(func8(nwg51_dr466.gfp_spatial), func8(nwg51_gbp.gfp_spatial))
# print(p)
# # ****
#
# t, p = ttest_ind(func8(nwg51_dr466.rfp_spatial), func8(nwg51_gbp.rfp_spatial))
# print(p)
# # ns


################################################################

# Embryos

"""
Displays a rotated image of a single embryo
Should specify the shape of the super image by setting rows and columns below

"""


def func(data, img, pos, vmin, vmax, title='', ylabel=''):
    ax0 = plt.subplot2grid((3, 2), pos)
    ax0.imshow(rotated_embryo(img, data.ROI_fitted, 300), cmap='gray', vmin=vmin, vmax=vmax)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(title, fontsize=9)
    ax0.set_ylabel(ylabel, fontsize=9)


# func(Data(nwg51_gbp_bleach.direcs[3]), Data(nwg51_gbp_bleach.direcs[3]).GFP, (0, 0), 3000, 30000, title='PAR-2 GFP',
#      ylabel='')
# func(Data(nwg51_gbp_bleach.direcs[3]), Data(nwg51_gbp_bleach.direcs[3]).RFP, (0, 1), 1000, 5000, title='PAR-2 mCherry',
#      ylabel='')
# func(Data(nwg51_gbp_bleach.direcs[10]), Data(nwg51_gbp_bleach.direcs[10]).GFP, (1, 0), 3000, 30000, title='',
#      ylabel='')
# func(Data(nwg51_gbp_bleach.direcs[10]), Data(nwg51_gbp_bleach.direcs[10]).RFP, (1, 1), 1000, 5000, title='',
#      ylabel='')
# func(Data(nwg51_gbp_bleach.direcs[5]), Data(nwg51_gbp_bleach.direcs[5]).GFP, (2, 0), 3000, 30000, title='',
#      ylabel='')
# func(Data(nwg51_gbp_bleach.direcs[5]), Data(nwg51_gbp_bleach.direcs[5]).RFP, (2, 1), 1000, 5000, title='',
#      ylabel='')
# plt.show()

# func(Data(nwg51_dr466_bleach.direcs[0]), Data(nwg51_dr466_bleach.direcs[0]).GFP, (0, 0), 3000, 30000, title='PAR-2 GFP',
#      ylabel='')
# func(Data(nwg51_dr466_bleach.direcs[0]), Data(nwg51_dr466_bleach.direcs[0]).RFP, (0, 1), 1000, 5000,
#      title='PAR-2 mCherry',
#      ylabel='')
# func(Data(nwg51_dr466_bleach.direcs[1]), Data(nwg51_dr466_bleach.direcs[1]).GFP, (1, 0), 3000, 30000, title='',
#      ylabel='')
# func(Data(nwg51_dr466_bleach.direcs[1]), Data(nwg51_dr466_bleach.direcs[1]).RFP, (1, 1), 1000, 5000, title='',
#      ylabel='')
# func(Data(nwg51_dr466_bleach.direcs[2]), Data(nwg51_dr466_bleach.direcs[2]).GFP, (2, 0), 3000, 30000, title='',
#      ylabel='')
# func(Data(nwg51_dr466_bleach.direcs[2]), Data(nwg51_dr466_bleach.direcs[2]).RFP, (2, 1), 1000, 5000, title='',
#      ylabel='')
# plt.show()

################################################################


# Save rotated embryo images

def func(data, img, direc, min, max):
    saveimg_jpeg(rotated_embryo(img, data.ROI_fitted, 300), direc, min=min, max=max)


func(Data(nwg51_gbp_bleach.direcs[3]), Data(nwg51_gbp_bleach.direcs[3]).GFP, '../PosterFigs/gbp_gfp.jpg', min=7573, max=35446)
func(Data(nwg51_gbp_bleach.direcs[3]), Data(nwg51_gbp_bleach.direcs[3]).RFP, '../PosterFigs/gbp_rfp.jpg', min=1647, max=4919)

func(Data(nwg51_dr466_bleach.direcs[0]), Data(nwg51_dr466_bleach.direcs[0]).GFP, '../PosterFigs/wt_gfp.jpg', min=7573, max=35446)
func(Data(nwg51_dr466_bleach.direcs[0]), Data(nwg51_dr466_bleach.direcs[0]).RFP, '../PosterFigs/wt_rfp.jpg', min=1647, max=4919)






