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


# Set up axes
ax0 = plt.subplot2grid((2, 2), (0, 0))
ax0.set_ylabel('[Cytoplasmic PAR-2] (a.u.)')
ax0.set_xticks([])
ax1 = plt.subplot2grid((2, 2), (0, 1))
ax1.set_ylabel('[Cortical PAR-2] (a.u.)')
ax1.set_xticks([])
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax2.set_ylabel('Cortex / Cytoplasm (a.u.)')
ax2.set_xticks([])
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.set_ylabel('Total PAR-2 (a.u.)')
ax3.set_xticks([])

# Plots
plots([ax0, ax1, ax2, ax3], kk1273_wt, 4, 'kk1273_wt')
plots([ax0, ax1, ax2, ax3], kk1273_par6, 8, 'kk1273_par6')
plots([ax0, ax1, ax2, ax3], nwg0123_wt, 12, 'nwg0123_wt')
plots([ax0, ax1, ax2, ax3], nwg0062_wt, 16, 'nwg0062_wt')
plots([ax0, ax1, ax2, ax3], nwg0062_par6, 20, 'nwg0062_par6')
plots([ax0, ax1, ax2, ax3], jh1799_wt, 24, 'jh1799_wt')
plots([ax0, ax1, ax2, ax3], jh1799_par6, 28, 'jh1799_par6')
plots([ax0, ax1, ax2, ax3], jh2882_wt, 32, 'jh2882_wt')
plots([ax0, ax1, ax2, ax3], jh2817_wt, 36, 'jh2817_wt')
plots([ax0, ax1, ax2, ax3], jh2817_par6, 40, 'jh2817_par6')
plots([ax0, ax1, ax2, ax3], th129_wt, 44, 'th129_wt')
plots([ax0, ax1, ax2, ax3], th129_par6, 48, 'th129_par6')
labels = ['kk1273_wt', 'kk1273_par6', 'nwg0123_wt', 'nwg0062_wt', 'nwg0062_par6', 'jh1799_wt', 'jh1799_par6', 'jh2882_wt', 'jh2817_wt', 'jh2817_par6', 'th129_wt', 'th129_par6']
ax0.set_xticklabels(labels, rotation=45, fontsize=8)
ax1.set_xticklabels(labels, rotation=45, fontsize=8)
ax2.set_xticklabels(labels, rotation=45, fontsize=8)
ax3.set_xticklabels(labels, rotation=45, fontsize=8)
sns.despine()
plt.show()


###############################################################

# Spatial distribution

# # Averages
# plt.plot(np.mean(kk1273_wt.gfp_spatial, 0))
# plt.plot(np.mean(kk1273_par6.gfp_spatial, 0))
# plt.plot(np.mean(nwg0123_wt.gfp_spatial, 0))
# plt.plot(np.mean(nwg0062_wt.gfp_spatial, 0))
# plt.plot(np.mean(nwg0062_par6.gfp_spatial, 0))
# plt.plot(np.mean(jh1799_wt.gfp_spatial, 0))
# plt.plot(np.mean(jh1799_par6.gfp_spatial, 0))
# plt.plot(np.mean(jh2882_wt.gfp_spatial, 0))
# plt.plot(np.mean(jh2817_wt.gfp_spatial, 0))
# plt.plot(np.mean(jh2817_par6.gfp_spatial, 0))
# plt.plot(np.mean(th129_wt.gfp_spatial, 0))
# plt.plot(np.mean(th129_par6.gfp_spatial, 0))


# Individual line

def func(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c)


func(kk1273_wt, 'r')
func(kk1273_par6, 'k')
func(nwg0123_wt, 'b')
# func(nwg0062_wt, 'k')
# func(nwg0062_par6, 'k')
# func(jh1799_wt,  'k')
# func(jh1799_par6, 'k')
# func(jh2882_wt,  'k')
# func(jh2817_wt, 'k')
# func(jh2817_par6,  'k')
# func(th129_wt, 'k')
# func(th129_par6, 'k')

plt.xlabel('x / circumference')
plt.ylabel('Cortex / Cytoplasm (a.u.)')
sns.despine()
plt.show()


####################################################################

# Cross section

def func1(dataset, c):
    for x in range(len(dataset.gfp_csection[:, 0])):
        plt.plot(dataset.gfp_csection[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_csection, 0), c=c)


# func1(kk1273_wt, 'r')
# func1(nwg0123_wt, 'b')
# plt.xlabel('Position')
# plt.ylabel('Intensity')
# sns.despine()
# plt.show()

