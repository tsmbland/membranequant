import Experiments.e1803__par2_rundown as e1
import Experiments.e1804__par2_rundown_nelio as e2
import Experiments.e1806__par2_mutants as e3
from IA import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0

####################################################################

# Normalise data

nwg0123_wt = normalise(e1.nwg0123_wt, e1.kk1273_wt)
nwg0123_rd = normalise(e1.nwg0123_rd, e1.kk1273_wt)
kk1273_wt = normalise(e1.kk1273_wt, e1.kk1273_wt)

nwg76_rd = normalise(e2.nwg76_rd, e2.nwg76_wt)
nwg76_wt = normalise(e2.nwg76_wt, e2.nwg76_wt)

jh1799_wt = normalise(e3.jh1799_wt, e3.kk1273_wt)
jh1799_par6 = normalise(e3.jh1799_par6, e3.kk1273_wt)
kk1273_wt2 = normalise(e3.kk1273_wt, e3.kk1273_wt)


####################################################################

# Cyt vs mem

def func(dataset, c, f):
    plt.scatter(dataset.g_cyt, dataset.g_mem, s=10, facecolors=f, edgecolors=c, linewidth=1)


# Filter out points with aPAR in posterior
nwg76_rd.g_cyt = nwg76_rd.g_cyt[nwg76_rd.r_mem < 500]
nwg76_rd.g_mem = nwg76_rd.g_mem[nwg76_rd.r_mem < 500]

# Plot data
a = join3([kk1273_wt, nwg76_wt, nwg76_rd])
b = join3([nwg0123_wt, nwg0123_rd])
func(a, c='k', f='k')
func(b, c='g', f='g')

# Plot lines
# plt.plot(np.linspace(0, 1.2, 2), np.mean(a.corts_GFP / a.cyts_GFP) * np.linspace(0, 1.2, 2), c='g')
# plt.plot(np.linspace(0, 1.2, 2), np.mean(b.corts_GFP / b.cyts_GFP) * np.linspace(0, 1.2, 2), c='b')


f += 1
plt.close()
plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
plt.ylabel('[Cortical PAR-2] (a.u.)')
sns.despine()
plt.rcParams.update({'font.size': 10})
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Cortex vs ratio

def func(dataset, c):
    plt.scatter(dataset.r_mem, dataset.g_mem / dataset.g_cyt, s=5, facecolors='none', edgecolors=c,
                linewidth=1)


f += 1
plt.close()
func(a, c='g')
func(b, c='b')
sns.despine()
plt.rcParams.update({'font.size': 10})
plt.xlim([0, 1.2])
plt.ylim([0, 1.2])
plt.savefig('%s/f%s.png' % (fdirec, f))
