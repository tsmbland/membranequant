from Experiments.e1803__par2par1_rundown import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.r_spa[x, :], dataset.g_spa[x, :] / dataset.g_cyt[x], s=0.2, c=c)


f += 1
plt.close()
func(nwg0132_wt, c='k')
func(nwg0132_rd, c='k')
plt.xlabel('Cortical PAR-2')
plt.ylabel('Cortical / Cytoplasmic PAR-1 ')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################


def func3(dataset, c, f):
    plt.scatter(dataset.r_mem, dataset.g_mem / dataset.g_cyt, s=5, facecolors=f, edgecolors=c, linewidth=1)


f += 1
plt.close()
func3(nwg42_wt, 'r', f='r')
func3(nwg0132_wt, 'b', f='b')
func3(nwg0132_rd, 'b', f='none')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.xlabel('[Cortical PAR-2] (a.u.)')
plt.ylabel('Cortical / Cytoplasmic PAR1 (a.u.)')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Spatial distribution

def func2(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.g_spa[x, :] / dataset.g_cyt[x], c=c, alpha=0.2, linestyle='-')
        plt.plot(dataset.r_spa[x, :] / dataset.r_cyt[x], c=c, alpha=0.2, linestyle='--')

    plt.plot(np.mean(dataset.g_spa, 0) / np.mean(dataset.g_cyt), c=c, linestyle='-')
    plt.plot(np.mean(dataset.r_spa, 0) / np.mean(dataset.r_cyt), c=c, linestyle='--')


f += 1
plt.close()
func2(nwg42_wt, c='r')
func2(nwg0132_wt, c='b')
# func2(nwg0132_rd, c='g')
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Dosage correlation

def func4(dataset, c, f):
    plt.scatter(dataset.r_tot, dataset.g_tot, edgecolors=c, facecolors=f, s=5, linewidth=1)


f += 1
plt.close()
func4(nwg42_wt, 'r', f='r')
func4(nwg0132_wt, 'b', f='b')
func4(nwg0132_rd, 'b', f='none')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.xlabel('Total PAR2 (a.u.)')
plt.ylabel('Total PAR1 (a.u.)')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))

####################################################################

# PAR2 membrane binding

# Cyt vs mem

f += 1
plt.close()
ax4 = plt.subplot2grid((1, 1), (0, 0))
ax4.scatter(nwg42_wt.r_cyt, nwg42_wt.r_mem, s=5, facecolors='none', edgecolors='r', linewidth=1, label='wt')
ax4.scatter(nwg0132_wt.r_cyt, nwg0132_wt.r_mem, s=5, facecolors='none', edgecolors='b', linewidth=1,
            label='par-3(it71)')
ax4.scatter(nwg0132_rd.r_cyt, nwg0132_rd.r_mem, s=5, facecolors='none', edgecolors='b', linewidth=1)
ax4.set_xlabel('[Cytoplasmic PAR-2] (a.u.)')
ax4.set_ylabel('[Cortical PAR-2] (a.u.)')
sns.despine()
plt.rcParams.update({'font.size': 10})
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.savefig('%s/f%s.png' % (fdirec, f))
