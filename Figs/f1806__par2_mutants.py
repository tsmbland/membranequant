from Experiments.e1806__par2_mutants import *

array = [kk1273_wt, kk1273_par6, nwg0123_wt, nwg0062_wt, nwg0062_par6, jh1799_wt, jh1799_par6, jh2882_wt, jh2817_wt,
         jh2817_par6, th129_wt, th129_par6]


def bar(ax, data, pos, name, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)

    ax.set_xticks(list(ax.get_xticks()) + [pos])
    ax.set_xlim([0, max(ax.get_xticks()) + 4])


def plots(axes, data, pos, name, c='k'):
    bar(axes[0], data.cyts_GFP, pos, name, c)
    bar(axes[1], data.corts_GFP, pos, name, c)
    bar(axes[2], data.corts_GFP / data.cyts_GFP, pos, name, c)
    bar(axes[3], data.totals_GFP, pos, name, c)


# # Set up axes
# ax0 = plt.subplot2grid((2, 2), (0, 0))
# ax0.set_ylabel('[Cytoplasmic PAR-2] (a.u.)')
# ax0.set_xticks([])
# ax1 = plt.subplot2grid((2, 2), (0, 1))
# ax1.set_ylabel('[Cortical PAR-2] (a.u.)')
# ax1.set_xticks([])
# ax2 = plt.subplot2grid((2, 2), (1, 0))
# ax2.set_ylabel('Cortex / Cytoplasm (a.u.)')
# ax2.set_xticks([])
# ax3 = plt.subplot2grid((2, 2), (1, 1))
# ax3.set_ylabel('Total PAR-2 (a.u.)')
# ax3.set_xticks([])
#
# # Plots
# plots([ax0, ax1, ax2, ax3], kk1273_wt, 4, 'kk1273_wt')
# plots([ax0, ax1, ax2, ax3], kk1273_par6, 8, 'kk1273_par6')
# plots([ax0, ax1, ax2, ax3], nwg0123_wt, 12, 'nwg0123_wt')
# plots([ax0, ax1, ax2, ax3], nwg0062_wt, 16, 'nwg0062_wt')
# plots([ax0, ax1, ax2, ax3], nwg0062_par6, 20, 'nwg0062_par6')
# plots([ax0, ax1, ax2, ax3], jh1799_wt, 24, 'jh1799_wt')
# plots([ax0, ax1, ax2, ax3], jh1799_par6, 28, 'jh1799_par6')
# plots([ax0, ax1, ax2, ax3], jh2882_wt, 32, 'jh2882_wt')
# plots([ax0, ax1, ax2, ax3], jh2817_wt, 36, 'jh2817_wt')
# plots([ax0, ax1, ax2, ax3], jh2817_par6, 40, 'jh2817_par6')
# plots([ax0, ax1, ax2, ax3], th129_wt, 44, 'th129_wt')
# plots([ax0, ax1, ax2, ax3], th129_par6, 48, 'th129_par6')
# labels = ['kk1273_wt', 'kk1273_par6', 'nwg0123_wt', 'nwg0062_wt', 'nwg0062_par6', 'jh1799_wt', 'jh1799_par6',
#           'jh2882_wt', 'jh2817_wt', 'jh2817_par6', 'th129_wt', 'th129_par6']
# ax0.set_xticklabels(labels, rotation=45, fontsize=8)
# ax1.set_xticklabels(labels, rotation=45, fontsize=8)
# ax2.set_xticklabels(labels, rotation=45, fontsize=8)
# ax3.set_xticklabels(labels, rotation=45, fontsize=8)
# sns.despine()
# plt.show()


###############################################################

# Spatial distribution with stdev, a to p

def func(dataset, c):
    gfp_spatial = np.concatenate((np.flip(dataset.gfp_spatial[:, :50], 1), dataset.gfp_spatial[:, 50:]))
    cyts_GFP = np.append(dataset.cyts_GFP, dataset.cyts_GFP)
    mean = np.mean(gfp_spatial, 0) / np.mean(cyts_GFP) * 0.255
    stdev = np.std(gfp_spatial / np.tile(cyts_GFP, (50, 1)).T, 0) * 0.255
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


# func(kk1273_wt, 'k')
# # func(kk1273_par6, 'b')
# func(nwg0123_wt, 'g')
# func(nwg0062_wt, 'b')
# # func(nwg0062_par6, 'k')
# # func(jh1799_wt,  'r')
# # func(jh1799_par6, 'b')
# # func(jh2882_wt,  'k')
# # func(jh2817_wt, 'k')
# # func(jh2817_par6,  'k')
# # func(th129_wt, 'k')
# # func(th129_par6, 'r')
#
# # plt.xlabel('x / circumference')
# # plt.ylabel('Cortex / Cytoplasm (a.u.)')
# plt.xticks([])
# sns.despine()
# plt.show()


####################################################################

# Cross section

# def func1(dataset, c):
#     for x in range(len(dataset.gfp_csection[:, 0])):
#         plt.plot(dataset.gfp_csection[x, :], c=c, alpha=0.2)
#     plt.plot(np.mean(dataset.gfp_csection, 0), c=c)
#
#
# func1(kk1273_wt, 'r')
# func1(nwg0123_wt, 'b')
# plt.xlabel('Position')
# plt.ylabel('Intensity')
# sns.despine()
# plt.show()


####################################################################

# Bar chart

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


# ax = plt.subplot2grid((1, 1), (0, 0))
# # ax.set_ylabel('Membrane : Cytoplasm')
# bar(ax, 0.255 * kk1273_wt.corts_GFP / kk1273_wt.cyts_GFP, 4, 'k')
# bar(ax, 0.255 * kk1273_par6.corts_GFP / kk1273_par6.cyts_GFP, 8, 'b')
# bar(ax, 0.255 * jh1799_wt.corts_GFP / jh1799_wt.cyts_GFP, 12, 'k')
# bar(ax, 0.255 * jh1799_par6.corts_GFP / jh1799_par6.cyts_GFP, 16, 'b')
# tidy([ax], ['', '', '', ''], [4, 8, 12, 16])
# sns.despine()
# plt.xticks([])
# plt.show()

# # Tests for significance
# from scipy.stats import ttest_ind
#
# t, p = ttest_ind(kk1273_wt.corts_GFP / kk1273_wt.cyts_GFP, kk1273_par6.corts_GFP / kk1273_par6.cyts_GFP)
# print(p)
# # ****
#
# t, p = ttest_ind(jh1799_wt.corts_GFP / jh1799_wt.cyts_GFP, jh1799_par6.corts_GFP / jh1799_par6.cyts_GFP)
# print(p)
# # ns



# ################################################################
#
#
# # Save rotated embryo images
#
# def func(data, img, direc, min, max):
#     saveimg_jpeg(rotated_embryo(img, data.ROI_fitted, 300), direc, min=min, max=max)
#
#
# func(Data(nwg51_gbp_bleach.direcs[0]), Data(nwg51_gbp_bleach.direcs[0]).GFP, '../PosterFigs/gbp_gfp.jpg', min=7573,
#      max=35446)
# func(Data(nwg51_gbp_bleach.direcs[0]), Data(nwg51_gbp_bleach.direcs[0]).RFP, '../PosterFigs/gbp_rfp.jpg', min=1647,
#      max=4919)
#
# func(Data(nwg51_dr466_bleach.direcs[0]), Data(nwg51_dr466_bleach.direcs[0]).GFP, '../PosterFigs/wt_gfp.jpg', min=7573,
#      max=35446)
# func(Data(nwg51_dr466_bleach.direcs[0]), Data(nwg51_dr466_bleach.direcs[0]).RFP, '../PosterFigs/wt_rfp.jpg', min=1647,
#      max=4919)


####################################################################

# Bar chart 2

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


ax = plt.subplot2grid((1, 1), (0, 0))
# ax.set_ylabel('Membrane : Cytoplasm')
bar(ax, 0.255 * kk1273_wt.corts_GFP / kk1273_wt.cyts_GFP, 4, 'k')
bar(ax, 0.255 * nwg0123_wt.corts_GFP / nwg0123_wt.cyts_GFP, 8, 'g')
bar(ax, 0.255 * nwg0062_wt.corts_GFP / nwg0062_wt.cyts_GFP, 12, 'b')


tidy([ax], ['', '', ''], [4, 8, 12])
sns.despine()
plt.xticks([])
plt.show()

# Tests for significance
from scipy.stats import ttest_ind

t, p = ttest_ind(kk1273_wt.corts_GFP / kk1273_wt.cyts_GFP, nwg0123_wt.corts_GFP / nwg0123_wt.cyts_GFP)
print(p)



