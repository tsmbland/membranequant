from IA import *


################################################################

# Bar plots

"""
Set of 4 bar plots displaying cytoplasmic, cortical cytoplasmic/cortical and total GFP

"""


def bar(ax, data, pos, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)
    ax.set_xticklabels([])
    ax.set_xticks([])


def plots(axes, data, pos, c='k'):
    bar(axes[0], data.cyts_GFP, pos, c)
    bar(axes[1], data.corts_GFP, pos, c)
    bar(axes[2], data.corts_GFP / data.cyts_GFP, pos, c)
    bar(axes[3], data.totals_GFP, pos, c)


def init():
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax0.set_ylabel('[Cytoplasmic PAR-2] (a.u.)')
    ax1 = plt.subplot2grid((2, 2), (0, 1))
    ax1.set_ylabel('[Cortical PAR-2] (a.u.)')
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_ylabel('Cortex / Cytoplasm (a.u.)')
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.set_ylabel('Total PAR-2 (a.u.)')
    return ax0, ax1, ax2, ax3


def tidy(axes, labels, positions):
    for a in axes:
        a.set_xticks(positions)
        a.set_xticklabels(labels, fontsize=7)


# [ax0, ax1, ax2, ax3] = init()
# plots([ax0, ax1, ax2, ax3], dataset, 4, c='')
# labels = []
# positions = []
# tidy([ax0, ax1, ax2, ax3], labels, positions)
# sns.despine()
# plt.show()


################################################################

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


# func1(dataset, c='g')
# func2(dataset, c='r')
# plt.show()


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


# func1(dataset, c='g')
# func2(dataset, c='r')
# plt.show()


################################################################

# Embryos

"""
Displays a rotated image of a single embryo
Should specify the shape of the super image by setting rows and columns below

"""


def func(data, pos, title='', ylabel=''):
    ax = plt.subplot2grid((rows, columns), pos)
    ax.imshow(rotated_embryo(data.GFP, data.ROI_fitted, 300), cmap='gray', vmin=3000, vmax=40000)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


# func(Data(dataset.direcs[0]), (0, 0), title='', ylabel='')
# plt.show()


################################################################

# Correlation (cortex, loc by loc)

"""
Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.rfp_spatial[x, :], s=0.1)


# func(dataset)
# plt.show()


####################################################################

# Correlation (cortex, global)

"""
Plots the cortical correlation between the G and R channels on an embryo by embryo basis

"""


def func(dataset, c, f):
    plt.scatter(dataset.corts_RFP, dataset.corts_GFP / dataset.cyts_GFP, s=5, facecolors=f, edgecolors=c, linewidth=1)
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.xlabel('[Cortical GFP] (a.u.)')
    plt.ylabel('Cortical / Cytoplasmic RFP (a.u.)')
    sns.despine()


# func(dataset, c='k', f='k')
# plt.show()


####################################################################

# Dosage correlation

"""
Plots the correlation between the dosage of G and R proteins

"""


def func(dataset, c):
    plt.scatter(dataset.totals_GFP, dataset.totals_RFP, c=c, s=10)
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.xlabel('Total GFP')
    plt.ylabel('Total mCherry')


# func(dataset, c='k')
# plt.show()


####################################################################

# Cyt vs mem

"""
Plots the correlation between cytoplasmic and cortical signals

"""


def func1(dataset, c, f):
    plt.scatter(dataset.cyts_GFP, dataset.corts_GFP, s=5, facecolors=f, edgecolors=c, linewidth=1)
    plt.xlabel('[Cytoplasmic GFP] (a.u.)')
    plt.ylabel('[Cortical GFP] (a.u.)')
    sns.despine()
    plt.rcParams.update({'font.size': 10})


def func2(dataset, c, f):
    plt.scatter(dataset.cyts_RFP, dataset.corts_RFP, s=5, facecolors=f, edgecolors=c, linewidth=1)
    plt.xlabel('[Cytoplasmic mCherry] (a.u.)')
    plt.ylabel('[Cortical mCherry] (a.u.)')
    sns.despine()
    plt.rcParams.update({'font.size': 10})

# func1(dataset, c='g', f='g')
# func2(dataset, c='r', f='r')
# plt.show()


####################################################################
