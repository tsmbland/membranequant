from Experiments.e1806__nwg91 import *


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.scatter(dataset.gfp_spatial[x, :], dataset.rfp_spatial[x, :], s=0.2, c=c)


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
        plt.scatter(dataset.gfp_spatial[x, :], dataset.cyts_RFP[x] / dataset.rfp_spatial[x, :], s=0.2, c=c)


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


func2(wt, c='r')
# func2(par1, c='b')
# func2(chin1, c='g')
# func2(spd5, c='k')
plt.xlabel('x / circumference')
plt.ylabel('Cortex / Cytoplasm (a.u.)')
sns.despine()
plt.show()


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
