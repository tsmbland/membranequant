from IA import *
from Experiments.e1807__par2_c1b import *

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


# func1(nwg0145_wt, c='g')
# func2(nwg0145_wt, c='r')
# func1(nwg0145_pma, c='k')
# func2(nwg0145_pma, c='b')
# plt.show()


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


# # func1(nwg0145_wt, c='g')
# func2(nwg0145_wt, c='r')
# # func1(nwg0145_pma, c='k')
# func2(nwg0145_pma, c='b')
# plt.show()


################################################################

# Spatial distribution (individual embryos)

def func(dataset):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c='g')
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c='r')
        print(dataset.direcs[x])
        plt.show()


# func(nwg0145_wt)
# func(nwg0145_pma)


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


# func1(nwg0145_wt, c='g')
# func2(nwg0145_wt, c='r')
# func1(nwg0145_pma, c='k')
# func2(nwg0145_pma, c='b')
# plt.show()


################################################################

# Correlation (cortex, loc by loc)

"""
Plots the cortical correlation between the G and R channels on a location by location basis

"""


def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.rfp_spatial[x, :], s=0.1, c=c)


func(nwg0145_wt, c='r')
func(nwg0145_pma, c='k')
plt.xlabel('Cortical GFP::C1B::PAR-2')
plt.ylabel('Cortical PAR-2')
sns.despine()
plt.gca().set_xlim(right=55000)
plt.show()



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


