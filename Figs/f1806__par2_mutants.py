from Experiments.e1806__par2_mutants import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0

#############################################################################################

array = [kk1273_wt, kk1273_par6, nwg0123_wt, nwg0062_wt, nwg0062_par6, jh1799_wt, jh1799_par6, jh2882_wt, jh2817_wt,
         jh2817_par6, th129_wt, th129_par6]


def bar(ax, data, pos, name, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)

    ax.set_xticks(list(ax.get_xticks()) + [pos])
    ax.set_xlim([0, max(ax.get_xticks()) + 4])


def plots(axes, data, pos, name, c='k'):
    bar(axes[0], data.g_cyt, pos, name, c)
    bar(axes[1], data.g_mem, pos, name, c)
    bar(axes[2], data.g_mem / data.g_cyt, pos, name, c)
    bar(axes[3], data.g_tot, pos, name, c)


f += 1
plt.close()

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
plots([ax0, ax1, ax2, ax3], jh2882_par6, 36, 'jh2882_par6')
plots([ax0, ax1, ax2, ax3], jh2817_wt, 40, 'jh2817_wt')
plots([ax0, ax1, ax2, ax3], jh2817_par6, 44, 'jh2817_par6')
plots([ax0, ax1, ax2, ax3], th129_wt, 48, 'th129_wt')
plots([ax0, ax1, ax2, ax3], th129_par6, 52, 'th129_par6')
labels = ['kk1273_wt', 'kk1273_par6', 'nwg0123_wt', 'nwg0062_wt', 'nwg0062_par6', 'jh1799_wt', 'jh1799_par6',
          'jh2882_wt', 'jh2882_par6', 'jh2817_wt', 'jh2817_par6', 'th129_wt', 'th129_par6']
ax0.set_xticklabels(labels, rotation=45, fontsize=8)
ax1.set_xticklabels(labels, rotation=45, fontsize=8)
ax2.set_xticklabels(labels, rotation=45, fontsize=8)
ax3.set_xticklabels(labels, rotation=45, fontsize=8)
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


###############################################################

# Spatial distribution with stdev, a to p

def func(dataset, c):
    g_spa = np.concatenate((np.flip(dataset.g_spa[:, :50], 1), dataset.g_spa[:, 50:]))
    g_cyt = np.append(dataset.g_cyt, dataset.g_cyt)
    mean = np.mean(g_spa, 0) / np.mean(g_cyt) * 0.255
    stdev = np.std(g_spa / np.tile(g_cyt, (50, 1)).T, 0) * 0.255
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


f += 1
plt.close()
func(kk1273_wt, 'k')
func(kk1273_par6, 'b')
# func(nwg0123_wt, 'g')
# func(nwg0062_wt, 'b')
# func(nwg0062_par6, 'k')
# func(jh1799_wt,  'k')
# func(jh1799_par6, 'b')
# func(jh2882_wt,  'k')
# func(jh2817_wt, 'k')
# func(jh2817_par6,  'k')
# func(th129_wt, 'k')
# func(th129_par6, 'r')

# plt.xlabel('x / circumference')
# plt.ylabel('Cortex / Cytoplasm (a.u.)')
plt.xticks([])
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Cross section

def func1(dataset, c):
    for x in range(len(dataset.g_cse[:, 0])):
        plt.plot(dataset.g_cse[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.g_cse, 0), c=c)


f += 1
plt.close()
func1(kk1273_wt, 'r')
func1(nwg0123_wt, 'b')
plt.xlabel('Position')
plt.ylabel('Intensity')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


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


f += 1
plt.close()
ax = plt.subplot2grid((1, 1), (0, 0))
# ax.set_ylabel('Membrane : Cytoplasm')
bar(ax, 0.255 * kk1273_wt.g_mem / kk1273_wt.g_cyt, 4, 'k')
bar(ax, 0.255 * kk1273_par6.g_mem / kk1273_par6.g_cyt, 8, 'b')
bar(ax, 0.255 * jh1799_wt.g_mem / jh1799_wt.g_cyt, 12, 'k')
bar(ax, 0.255 * jh1799_par6.g_mem / jh1799_par6.g_cyt, 16, 'b')
tidy([ax], ['', '', '', ''], [4, 8, 12, 16])
sns.despine()
plt.xticks([])
plt.savefig('%s/f%s.png' % (fdirec, f))


# Tests for significance
# from scipy.stats import ttest_ind
#
# t, p = ttest_ind(kk1273_wt.g_mem / kk1273_wt.g_cyt, kk1273_par6.g_mem / kk1273_par6.g_cyt)
# print(p)
# # ****
#
# t, p = ttest_ind(jh1799_wt.g_mem / jh1799_wt.g_cyt, jh1799_par6.g_mem / jh1799_par6.g_cyt)
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


f += 1
plt.close()
ax = plt.subplot2grid((1, 1), (0, 0))
# ax.set_ylabel('Membrane : Cytoplasm')
bar(ax, 0.255 * kk1273_wt.g_mem / kk1273_wt.g_cyt, 4, 'k')
bar(ax, 0.255 * nwg0123_wt.g_mem / nwg0123_wt.g_cyt, 8, 'g')
bar(ax, 0.255 * nwg0062_wt.g_mem / nwg0062_wt.g_cyt, 12, 'b')

tidy([ax], ['', '', ''], [4, 8, 12])
sns.despine()
plt.xticks([])
plt.savefig('%s/f%s.png' % (fdirec, f))

# # Tests for significance
# from scipy.stats import ttest_ind
#
# t, p = ttest_ind(kk1273_wt.g_mem / kk1273_wt.g_cyt, nwg0123_wt.g_mem / nwg0123_wt.g_cyt)
# print(p)
