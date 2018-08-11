import Experiments.e1806__par2_mutants as e1
import Experiments.e1806__par2_mutants2 as e2
from IA import *


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


ax0 = plt.subplot2grid((1, 2), (0, 0))
ax0.set_ylabel('Dosage')
ax0.set_xticks([])

ax1 = plt.subplot2grid((1, 2), (0, 1))
ax1.set_ylabel('Dosage')
ax1.set_xticks([])

bar(ax0, e1.jh1799_wt.totals_GFP, 4, '', 'k')
bar(ax0, e1.jh1799_ctrl.totals_GFP, 8, '', 'k')
bar(ax0, e1.jh1799_par6.totals_GFP, 12, '', 'k')
bar(ax1, e2.jh1799_wt.totals_GFP, 4, '', 'k')
bar(ax1, e2.jh1799_pkc.totals_GFP, 8, '', 'k')

sns.despine()
plt.show()
