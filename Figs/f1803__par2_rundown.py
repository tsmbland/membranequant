from Experiments.e1803__par2_rundown import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0

# 1 ###################################################################

# Cyt vs mem

plt.close()
# plt.scatter(kk1273_wt.g_cyt, kk1273_wt.g_mem,  s=5, facecolors='r', edgecolors='r', linewidth=1)
plt.scatter(nwg0123_rd.g_cyt, nwg0123_rd.g_mem, s=5, facecolors='none', edgecolors='b', linewidth=1)
plt.scatter(nwg0123_wt.g_cyt, nwg0123_wt.g_mem, s=5, facecolors='b', edgecolors='b', linewidth=1)
plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
plt.ylabel('[Cortical PAR-2] (a.u.)')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))

# 2 ###################################################################

# Cortex vs ratio

plt.close()
ax4 = plt.subplot2grid((1, 1), (0, 0))
ax4.scatter(kk1273_wt.g_mem, kk1273_wt.g_mem / kk1273_wt.g_cyt, s=5, facecolors='none', edgecolors='r',
            linewidth=1, label='wt')
ax4.scatter(nwg0123_rd.g_mem, nwg0123_rd.g_mem / nwg0123_rd.g_cyt, s=5, facecolors='none', edgecolors='b',
            linewidth=1, label='par-3(it71)')
ax4.scatter(nwg0123_wt.g_mem, nwg0123_wt.g_mem / nwg0123_wt.g_cyt, s=5, facecolors='none', edgecolors='b',
            linewidth=1)
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
sns.despine()
plt.rcParams.update({'font.size': 10})
plt.savefig('%s/f%s.png' % (fdirec, f))
