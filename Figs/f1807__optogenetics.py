from Experiments.e1807__optogenetics import *

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


# func1(e6, c='g')
# func2(e6, c='r')
# plt.show()



####################################################################


def func(dataset, c='k'):
    plt.scatter(range(1, len(dataset.corts_RFP) + 1), dataset.corts_RFP / dataset.cyts_RFP, s=5, edgecolors=c, linewidth=1)


func(e2, c='k')
func(e3, c='b')
func(e4, c='r')
func(e5, c='g')
func(e6, c='y')

# plt.gca().set_ylim(bottom=0)
plt.ylabel('mCherry Cortex / Cytoplasm')
plt.xlabel('Image number')
sns.despine()
plt.show()


################################################################

# Embryos

"""
Displays a rotated image of a single embryo
Should specify the shape of the super image by setting rows and columns below

"""


def func(data):
    plt.imshow(rotated_embryo(data.GFP, data.ROI_fitted, 300), cmap='gray', vmin=3000, vmax=50000)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.imshow(rotated_embryo(data.RFP, data.ROI_fitted, 300), cmap='gray', vmin=2000, vmax=7000)
    plt.xticks([])
    plt.yticks([])
    plt.show()


func(Data(e4.direcs[0]))
func(Data(e4.direcs[9]))
plt.show()

