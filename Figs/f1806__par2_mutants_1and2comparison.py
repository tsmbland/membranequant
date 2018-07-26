from IA import *
import Experiments.e1806__par2_mutants as e1
import Experiments.e1806__par2_mutants2 as e2



def bar(ax, data, pos, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)
    ax.set_xticklabels([])
    ax.set_xticks([])


def plots(axes, data, pos, c='k'):
    bar(axes[0], data.cyts_GFP, pos, c)
    bar(axes[1], data.corts_GFP, pos, c)
    bar(axes[2], data.corts_GFP / data.cyts_GFP, pos,  c)
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


# # Plots
# [ax0, ax1, ax2, ax3] = init()
# plots([ax0, ax1, ax2, ax3], e1.kk1273_wt, 4, c='r')
# plots([ax0, ax1, ax2, ax3], e1.kk1273_par6, 8,  c='k')
# plots([ax0, ax1, ax2, ax3], e1.nwg0123_wt, 12, c='b')
# plots([ax0, ax1, ax2, ax3], e1.th129_wt, 20, c='r')
# plots([ax0, ax1, ax2, ax3], e1.th129_par6, 24,  c='k')
# plots([ax0, ax1, ax2, ax3], e1.nwg0062_wt, 32,  c='r')
# plots([ax0, ax1, ax2, ax3], e1.nwg0062_par6, 36, c='k')
# plots([ax0, ax1, ax2, ax3], e1.jh1799_wt, 44,  c='r')
# plots([ax0, ax1, ax2, ax3], e1.jh1799_par6, 48,  c='k')
# plots([ax0, ax1, ax2, ax3], e1.jh2882_wt, 56, c='r')
# plots([ax0, ax1, ax2, ax3], e1.jh2882_par6, 60, c='k')
# plots([ax0, ax1, ax2, ax3], e1.jh2817_wt, 68,  c='r')
# plots([ax0, ax1, ax2, ax3], e1.jh2817_par6, 72, c='k')
#
# labels = ['wt (CRISPR)', 'wt (Bombarded)', 'PAR-2 S241A', 'PAR-2 (178-412)', 'PAR-2 C56S', 'PAR-2 R163A']
# positions = [8, 22, 34, 46, 58, 70]
#
# ax0.set_xticks(positions)
# ax0.set_xticklabels(labels, fontsize=7)
# ax1.set_xticks(positions)
# ax1.set_xticklabels(labels, fontsize=7)
# ax2.set_xticks(positions)
# ax2.set_xticklabels(labels, fontsize=7)
# ax3.set_xticks(positions)
# ax3.set_xticklabels(labels, fontsize=7)
#
# sns.despine()
# plt.show()



###############################################################

# Spatial distribution

def func(dataset, c='k'):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c)


# func(e1.jh2882_wt, c='r')
# func(e1.jh2882_par6, c='k')
# # func(e1.nwg0123_wt, c='b')
# # func(e2.kk1273_pkc, c='g')
#
#
# plt.xlabel('x / circumference')
# plt.ylabel('Cortex / Cytoplasm (a.u.)')
# plt.gca().set_ylim(bottom=0)
# sns.despine()
# plt.show()



###############################################################

# Cross section

def func1(dataset, c):
    for x in range(len(dataset.gfp_csection[:, 0])):
        plt.plot(dataset.gfp_csection[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_csection, 0), c=c)


# func1(e1.th129_wt, c='r')
# func1(e1.th129_par6, c='k')
# plt.xlabel('Position')
# plt.ylabel('Intensity')
# sns.despine()
# plt.show()


###############################################################

# Embryos

def func(data, pos, title='', ylabel=''):
    ax = plt.subplot2grid((3, 7), pos)
    ax.imshow(rotated_embryo(data.GFP, data.ROI_fitted, 300), cmap='gray', vmin=3000, vmax=40000)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


func(Data(e1.kk1273_wt.direcs[0]), (0, 0), title='wt (CRISPR)', ylabel='wt')
func(Data(e1.kk1273_par6.direcs[0]), (1, 0), ylabel='PAR-6 RNAi')
func(Data(e1.nwg0123_wt.direcs[0]), (2, 0), ylabel='PAR-3 it71')
func(Data(e1.th129_wt.direcs[0]), (0, 1), title='wt (Bombarded)')
func(Data(e1.th129_par6.direcs[0]), (1, 1))
func(Data(e1.nwg0062_wt.direcs[1]), (0, 2), title='PAR-2 S241A')
func(Data(e1.nwg0062_par6.direcs[1]), (1, 2))
func(Data(e1.jh1799_wt.direcs[0]), (0, 3), title='PAR-2 (178-412)')
func(Data(e1.jh1799_par6.direcs[0]), (1, 3))
func(Data(e1.jh2882_wt.direcs[0]), (0, 4), title='PAR-2 C56S')
func(Data(e1.jh2882_par6.direcs[0]), (1, 4))
func(Data(e1.jh2817_wt.direcs[0]), (0, 5), title='PAR-2 R163A')
func(Data(e1.jh2817_par6.direcs[0]), (1, 5))
func(Data('N2/180412_n2_wt_tom3,15,pfsout/0'), (0, 6), title='N2')

plt.show()

