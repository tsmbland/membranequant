from Experiments.e1807__s241e import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0

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
    bar(axes[0], data.g_cyt, pos, c)
    bar(axes[1], data.g_mem, pos, c)
    bar(axes[2], data.g_mem / data.g_cyt, pos, c)
    bar(axes[3], data.g_tot, pos, c)


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


f += 1
plt.close()
[ax0, ax1, ax2, ax3] = init()
plots([ax0, ax1, ax2, ax3], kk1273_wt, 4, c='k')
plots([ax0, ax1, ax2, ax3], nwg62_wt, 8, c='r')
plots([ax0, ax1, ax2, ax3], nwg126_wt, 12, c='b')
labels = []
positions = []
tidy([ax0, ax1, ax2, ax3], labels, positions)
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))

################################################################

# Spatial distribution

"""
Plots the cortical/cytoplasm intensities around the circumference. Separate functions for GFP and mCherry

"""


def func1(dataset, c='k'):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.g_spa[x, :] / dataset.g_cyt[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.g_spa, 0) / np.mean(dataset.g_cyt), c=c)
    plt.xlabel('x / circumference')
    plt.ylabel('Cortex / Cytoplasm (a.u.)')
    # plt.gca().set_ylim(bottom=0)
    sns.despine()


f += 1
plt.close()
func1(kk1273_wt, c='k')
func1(nwg62_wt, c='r')
func1(nwg126_wt, c='b')
# func1(kk1273_wt_bleach, c='k')
# func1(nwg62_wt_bleach, c='b')
# func1(nwg126_wt_bleach, c='b')
plt.savefig('%s/f%s.png' % (fdirec, f))
