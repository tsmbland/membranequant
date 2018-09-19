import Experiments.e1806__par2_mutants as e1
import Experiments.e1806__par2_mutants2 as e2
from IA import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0


####################################################################

# Dosage analysis

def bar(ax, data, pos, name, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)
    ax.set_xticks(list(ax.get_xticks()) + [pos])
    ax.set_xlim([0, max(ax.get_xticks()) + 4])

    labels = [w.get_text() for w in ax.get_xticklabels()]
    labels += [name]
    ax.set_xticklabels(labels)


f += 1
plt.close()
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax0.set_xticks([])
ax1 = plt.subplot2grid((1, 2), (0, 1))
ax1.set_xticks([])
bar(ax0, e1.jh1799_wt.g_tot, 4, '', 'k')
bar(ax0, e1.jh1799_ctrl.g_tot, 8, '', 'k')
bar(ax0, e1.jh1799_par6.g_tot, 12, '', 'k')
bar(ax1, e2.jh1799_wt.g_tot, 4, '', 'k')
bar(ax1, e2.jh1799_pkc.g_tot, 8, '', 'k')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))

####################################################################

# Dosage analysis (jh2882)

f += 1
plt.close()
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax0.set_xticks([])
bar(ax0, e1.jh2882_wt.g_tot, 4, '', 'k')
bar(ax0, e1.jh2882_par6.g_tot, 8, '', 'k')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))
