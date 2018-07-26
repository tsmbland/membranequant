from Experiments.e1803__par2_rundown import *

####################################################################

# Cyt vs mem

plt.scatter(kk1273_wt.cyts_GFP, kk1273_wt.corts_GFP,  s=5, facecolors='r', edgecolors='r', linewidth=1)
plt.scatter(nwg0123_rd.cyts_GFP, nwg0123_rd.corts_GFP, s=5, facecolors='none', edgecolors='b', linewidth=1)
plt.scatter(nwg0123_wt.cyts_GFP, nwg0123_wt.corts_GFP,  s=5, facecolors='b', edgecolors='b', linewidth=1)
plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
plt.ylabel('[Cortical PAR-2] (a.u.)')

sns.despine()
plt.show()


####################################################################

# Cortex vs ratio

ax4 = plt.subplot2grid((1, 1), (0, 0))
ax4.scatter(kk1273_wt.corts_GFP, kk1273_wt.corts_GFP / kk1273_wt.cyts_GFP,  s=5, facecolors='none', edgecolors='r', linewidth=1, label='wt')
ax4.scatter(nwg0123_rd.corts_GFP, nwg0123_rd.corts_GFP / nwg0123_rd.cyts_GFP, s=5, facecolors='none', edgecolors='b', linewidth=1, label='par-3(it71)')
ax4.scatter(nwg0123_wt.corts_GFP, nwg0123_wt.corts_GFP / nwg0123_wt.cyts_GFP,  s=5, facecolors='none', edgecolors='b', linewidth=1)

# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# sns.despine()
# plt.rcParams.update({'font.size': 10})
# plt.show()

