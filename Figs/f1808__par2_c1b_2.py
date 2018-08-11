from IA import *
from Experiments.e1808__par2_c1b_2 import *

################################################################

# Spatial distribution (absolute amounts)

"""
Plots the cortical/cytoplasm intensities around the circumference. Separate functions for GFP and mCherry

"""


def func1(dataset, c='k'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_spatial, 0), c=c)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


def func2(dataset, c='k'):
    for x in range(len(dataset.rfp_spatial[:, 0])):
        plt.plot(dataset.rfp_spatial[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.rfp_spatial, 0), c=c)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


################################################################

# Spatial distribution (individual embryos)

def func(dataset):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        # plt.plot(dataset.gfp_spatial[x, :])
        plt.plot(dataset.rfp_spatial[x, :])
        print(dataset.direcs[x])
    plt.show()


# func(mp)

################################################################

# Cross section

"""
Plots an intensity cross section along the long axis if the embryo. Separate functions for GFP and mCherry

"""


def func1(dataset, c):
    for x in range(len(dataset.gfp_csection[:, 0])):
        plt.plot(dataset.gfp_csection[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_csection, 0), c=c)
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    sns.despine()


def func2(dataset, c):
    for x in range(len(dataset.rfp_csection[:, 0])):
        plt.plot(dataset.rfp_csection[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.rfp_csection, 0), c=c)
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    sns.despine()


################################################################

# Correlation (cortex, loc by loc)

"""
Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.rfp_spatial[x, :], s=0.1, c=c)


# plt.xlabel('Cortical GFP::C1B::PAR-2')
# plt.ylabel('Cortical PAR-2')
# sns.despine()
# plt.gca().set_xlim(right=55000)
# plt.show()



####################################################################

# 'Cytoplasmic' levels

def bar(ax, data, pos, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)
    ax.set_xticklabels([])
    ax.set_xticks([])


# ax0 = plt.subplot2grid((1, 1), (0, 0))
# bar(ax0, nwg0145_wt.cyts_GFP, 4, 'k')
# bar(ax0, nwg0145_pma.cyts_GFP, 8, 'k')
#
# bar(ax0, nwg0145_wt.cyts_RFP, 14, 'k')
# bar(ax0, nwg0145_pma.cyts_RFP, 18, 'k')
#
# labels = ['GFP w/o PMA', 'GFP w/ PMA', 'mCherry w/o PMA', 'mCherry w/ PMA']
# positions = [4, 8, 14, 18]
# plt.xticks(positions, labels)
# plt.ylabel('Mean Cytoplasmic Pixel Intensity')
# sns.despine()
# plt.show()


################################################################

# Embryos

"""
Displays a rotated image of a single embryo
Should specify the shape of the super image by setting rows and columns below

"""


def func(data):
    plt.imshow(rotated_embryo(data.GFP, data.ROI_fitted, 300), cmap='gray', vmin=3000, vmax=30000)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.imshow(rotated_embryo(data.RFP, data.ROI_fitted, 300), cmap='gray', vmin=2000, vmax=5000)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# func(Data(nwg0145_wt.direcs[1]))
# func(Data(nwg0145_pma.direcs[7]))
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
        a[x] = asi(concs[x, :])
    return a


# ax = plt.subplot2grid((1, 1), (0, 0))
# ax.set_ylabel('ASI')
# bar(ax, func8(nwg158_mp.gfp_spatial), 4, 'k')
# bar(ax, func8(nwg158_mp.rfp_spatial), 8, 'k')
# tidy([ax], ['GFP', 'mCherry'], [4, 8])
# sns.despine()
# plt.show()


####################################################################

# Dosage analysis

# ax = plt.subplot2grid((1, 1), (0, 0))
# ax.set_ylabel('Dosage')
# bar(ax, nwg158_mp.totals_GFP, 12, 'k')
# bar(ax, nwg158_mp.totals_RFP, 16, 'k')
# bar(ax, nwg158_pma_mp.totals_GFP, 4, 'k')
# bar(ax, nwg158_pma_mp.totals_RFP, 8, 'k')
# bar(ax, nwg151_wt.totals_GFP, 20, 'k')
# bar(ax, nwg151_wt.totals_RFP, 24, 'k')
# tidy([ax], ['GFP', 'mCherry', 'GFP', 'mCherry', 'GFP', 'mCherry'], [4, 8, 12, 16, 20, 24])
# sns.despine()
# plt.show()


################################################################

# Embryos, timelapse

"""
Displays a rotated image of a single embryo
Should specify the shape of the super image by setting rows and columns below

"""


def func(dataset, tmin, tmax):
    row = 0
    for t in range(tmin, tmax):
        data = Data(dataset.direcs[t])
        ax0 = plt.subplot2grid((len(range(tmin, tmax)), 2), (row, 0))
        ax0.imshow(rotated_embryo(data.GFP, data.ROI_fitted, 300), cmap='gray', vmin=3000, vmax=30000)
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = plt.subplot2grid((len(range(tmin, tmax)), 2), (row, 1))
        ax1.imshow(rotated_embryo(data.RFP, data.ROI_fitted, 300), cmap='gray', vmin=2000, vmax=5000)
        ax1.set_xticks([])
        ax1.set_yticks([])

        row += 1
    plt.show()


# func(e1, 0, 5)
# func(e1, 5, 9)
