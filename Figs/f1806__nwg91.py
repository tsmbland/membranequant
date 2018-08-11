from Experiments.e1806__nwg91 import *


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.rfp_spatial[x, :], s=0.5, c=c)


# func(par1, c='b')
# func(chin1, c='g')
# func(spd5, c='k')
# func(wt, c='r')
# plt.xlabel('Cortical PKC-3 GFP')
# plt.ylabel('Cortical PAR-2 mCherry')
# sns.despine()
# plt.show()


####################################################################

# PAR-2 removal

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.cyts_RFP[x] / dataset.rfp_spatial[x, :], s=0.5, c=c)


# func(par1, c='b')
# func(chin1, c='g')
# func(spd5, c='k')
# func(wt, c='r')
# plt.xlabel('Cortical PKC-3 GFP')
# plt.ylabel('Cytoplasmic / Cortical PAR-2 mCherry')
# plt.ylim([0.01, 1000])
# sns.despine()
# plt.yscale('log')
# plt.show()


####################################################################

# Spatial distribution

def func2(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c='g', alpha=0.2, linestyle='-')
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c='r', alpha=0.2, linestyle='-')

    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c='g', linestyle='-')
    plt.plot(np.mean(dataset.rfp_spatial, 0) / np.mean(dataset.cyts_RFP), c='r', linestyle='-')


# func2(wt, c='r')
# # func2(par1, c='b')
# # func2(chin1, c='g')
# # func2(spd5, c='k')
# plt.xlabel('x / circumference')
# plt.ylabel('Cortex / Cytoplasm (a.u.)')
# sns.despine()
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


# func1(wt, c='g')
# func2(wt, c='r')
# plt.show()


# func1(par1, c='g')
# func2(par1, c='r')
# plt.show()
#
# func1(chin1, c='g')
# func2(chin1, c='r')
# plt.show()


###############################################################

# Spatial distribution with stdev, normalised, a to p

def func1(dataset, c):
    gfp_spatial = np.concatenate((np.flip(dataset.gfp_spatial[:, :50], 1), dataset.gfp_spatial[:, 50:]))
    mean = np.mean(gfp_spatial, 0)
    a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    gfp_spatial = a[0] * gfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(gfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


def func2(dataset, c):
    rfp_spatial = np.concatenate((np.flip(dataset.rfp_spatial[:, :50], 1), dataset.rfp_spatial[:, 50:]))
    mean = np.mean(rfp_spatial, 0)
    a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    rfp_spatial = a[0] * rfp_spatial + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(rfp_spatial, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


# func1(wt, c='r')
# func2(wt, c='c')
# sns.despine()
# plt.xticks([])
# plt.yticks([0, 1])
# # plt.xlabel('x / circumference')
# # plt.ylabel('Cortex / Cytoplasm (a.u.)')
# plt.rcParams['savefig.dpi'] = 600
# plt.show()


# func1(par1, c='g')
# func2(par1, c='r')
# plt.show()
#
# func1(chin1, c='g')
# func2(chin1, c='r')
# plt.show()


####################################################################

# Dosage analysis

# def bar(ax, data, pos, name, c):
#     ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
#     ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
#                s=5)
#     ax.set_xticks(list(ax.get_xticks()) + [pos])
#     ax.set_xlim([0, max(ax.get_xticks()) + 4])
#
#     labels = [w.get_text() for w in ax.get_xticklabels()]
#     labels += [name]
#     ax.set_xticklabels(labels)
#
#
# ax0 = plt.subplot2grid((1, 2), (0, 0))
# ax0.set_ylabel('PKC3 dosage')
# ax0.set_xticks([])
#
# ax1 = plt.subplot2grid((1, 2), (0, 1))
# ax1.set_ylabel('PAR2 dosage')
# ax1.set_xticks([])
#
# bar(ax0, wt.totals_GFP, 4, 'wt', 'k')
# bar(ax0, par1.totals_GFP, 8, 'par1 RNAi', 'k')
# bar(ax0, chin1.totals_GFP, 12, 'chin1 RNAi', 'k')
# bar(ax0, spd5.totals_GFP, 16, 'spd5 RNAi', 'k')
#
# bar(ax1, wt.totals_RFP, 4, 'wt', 'k')
# bar(ax1, par1.totals_RFP, 8, 'par1 RNAi', 'k')
# bar(ax1, chin1.totals_RFP, 12, 'chin1 RNAi', 'k')
# bar(ax1, spd5.totals_RFP, 16, 'spd5 RNAi', 'k')
#
# sns.despine()
# plt.show()



################################################################

# Dual channel embryo

data = Data(wt.direcs[0])

img10 = rotated_embryo(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 300)
saveimg(img10, 'img10.TIF')
plt.imshow(img10)
plt.show()

img11 = straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50)
saveimg(img11, 'img11.TIF')
plt.imshow(img11)
plt.show()

bg = straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50), 50)
img20 = rotated_embryo(data.RFP - np.nanmean(bg[np.nonzero(bg)]), data.ROI_fitted, 300)
saveimg(img20, 'img20.TIF')
plt.imshow(img20)
plt.show()

img21 = straighten(data.RFP - np.nanmean(bg[np.nonzero(bg)]), data.ROI_fitted, 50)
saveimg(img21, 'img21.TIF')
plt.imshow(img21)
plt.show()
