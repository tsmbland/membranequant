from Experiments.e1806__par2_mutants2 import *


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
# plots([ax0, ax1, ax2, ax3], kk1273_pkc, 8, 'kk1273_pkc')
#
# plots([ax0, ax1, ax2, ax3], jh1799_wt, 12, 'jh1799_wt')
# plots([ax0, ax1, ax2, ax3], jh1799_pkc, 16, 'jh1799_pkc')
#
# plots([ax0, ax1, ax2, ax3], jh2817_wt, 20, 'jh2817_wt')
# plots([ax0, ax1, ax2, ax3], jh2817_pkc, 24, 'jh2817_pkc')
#
# plots([ax0, ax1, ax2, ax3], th129_wt, 28, 'th129_wt')
# plots([ax0, ax1, ax2, ax3], th129_pkc, 32, 'th129_pkc')
#
#
# labels = ['kk1273_wt', 'kk1273_pkc', 'jh1799_wt', 'jh1799_pkc', 'jh2817_wt', 'jh2817_pkc', 'th129_wt', 'th129_pkc']
# ax0.set_xticklabels(labels, rotation=45, fontsize=8)
# ax1.set_xticklabels(labels, rotation=45, fontsize=8)
# ax2.set_xticklabels(labels, rotation=45, fontsize=8)
# ax3.set_xticklabels(labels, rotation=45, fontsize=8)
#
# sns.despine()
# plt.show()


###############################################################

# Spatial distribution


# def func(dataset, c='k'):
#     for x in range(len(dataset.gfp_spatial[:, 0])):
#         plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts[x], c=c, alpha=0.2)
#     plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts), c=c)
#
#
# # func(kk1273_wt, c='r')
# func(kk1273_pkc, c='r')
# #
# # func(jh1799_wt, c='b')
# func(jh1799_pkc, c='b')
# #
# # func(jh2817_wt, c='g')
# func(jh2817_pkc, c='g')
# #
# # func(th129_wt, c='k')
# func(th129_pkc, c='k')
# #
#
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# plt.show()


###############################################################

# Spatial distribution with stdev

def func(dataset, c):
    gfp_spatial = np.concatenate((dataset.gfp_spatial[:, :50], np.flip(dataset.gfp_spatial[:, 50:], 1)))
    cyts_GFP = np.append(dataset.cyts_GFP, dataset.cyts_GFP)
    mean = np.mean(gfp_spatial, 0) / np.mean(cyts_GFP)
    stdev = np.std(gfp_spatial / np.tile(cyts_GFP, (50, 1)).T, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


# func(kk1273_wt, c='r')
# func(kk1273_pkc, c='b')
# func(jh1799_wt, c='r')
# func(jh1799_pkc, c='b')
# func(jh2817_wt, c='r')
# func(jh2817_pkc, c='b')
# func(th129_wt, c='r')
# func(th129_pkc, c='b')

plt.xlabel('x / circumference')
plt.ylabel('Cortex / Cytoplasm (a.u.)')
sns.despine()
plt.show()
