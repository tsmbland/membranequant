from Experiments.e1804__par2_rundown_nelio import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0

####################################################################

# Cyt vs mem

f += 1
plt.close()
plt.scatter(nwg76_wt.g_cyt, nwg76_wt.g_mem, s=5, facecolors='g', edgecolors='g', linewidth=1)
plt.scatter(nwg76_rd.g_cyt, nwg76_rd.g_mem, s=5, facecolors='none', edgecolors='g', linewidth=1)
plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
plt.ylabel('[Cortical PAR-2] (a.u.)')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))

####################################################################

# Cortex vs ratio

f += 1
plt.close()
ax4 = plt.subplot2grid((1, 1), (0, 0))
ax4.scatter(nwg76_wt.g_mem, nwg76_wt.g_mem / nwg76_wt.g_cyt, s=5, facecolors='none', edgecolors='r', linewidth=1,
            label='wt')
ax4.scatter(nwg76_rd.g_mem, nwg76_rd.g_mem / nwg76_rd.g_cyt, s=5, facecolors='none', edgecolors='b', linewidth=1,
            label='par-3(it71)')
# ax4.scatter(nwg0123_wt.corts, nwg0123_wt.corts / nwg0123_wt.cyts,  s=5, facecolors='none', edgecolors='b', linewidth=1)
sns.despine()
plt.rcParams.update({'font.size': 10})
plt.ylim([0, 35])
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Dosage correlation

def func(dataset, c):
    plt.scatter(dataset.g_tot, dataset.r_tot, c=c, s=10)


f += 1
plt.close()
func(nwg76_wt, 'r')
func(nwg76_rd, 'b')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.xlabel('Total PAR2')
plt.ylabel('Total PAR6')
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
func2(nwg76_wt, c='r')
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Cyt vs mem


def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_cyt[x], np.mean(dataset.g_spa[x, dataset.r_spa[x, :] < 500]), s=5, c=c)


f += 1
plt.close()
func(nwg76_wt, 'r')
func(nwg76_rd, 'b')
plt.xlabel('[Cytoplasmic PAR-2] (a.u.)')
plt.ylabel('[Cortical PAR-2] (a.u.)')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.r_spa[x, :], dataset.g_spa[x, :], s=0.2, c=c)


f += 1
plt.close()
func(nwg76_wt, c='k')
func(nwg76_rd, c='r')
plt.xlabel('Cortical PAR-6')
plt.ylabel('Cortical PAR-2')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_spa[x, :], dataset.r_cyt[x] / dataset.r_spa[x, :], s=0.2, c=c)


f += 1
plt.close()
func(nwg76_wt, c='k')
func(nwg76_rd, c='r')
plt.ylim([0, 10])
plt.xlabel('Cortical PAR-2')
plt.ylabel('Cytoplasmic / Cortical PAR-6')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# PAR-2 in posterior vs PAR-6 in anterior


def func7(dataset):
    for x in range(len(dataset.g_spa[:, 0])):
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Posterior pPAR vs anterior aPAR
        # plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds2), bounded_mean(dataset.r_spa[x, :], bounds1),
        #             c='k', s=10)

        # Posterior pPAR vs posterior aPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds2), bounded_mean(dataset.r_spa[x, :], bounds2),
                    c='b', s=10)


f += 1
plt.close()
func7(nwg76_wt)
func7(nwg76_rd)
# plt.gca().set_ylim(top=15000)
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################


# Remove datapoints where PAR-6 is above a threshold

def func(dataset):
    plt.scatter(dataset.g_cyt[dataset.r_mem < 500], dataset.g_mem[dataset.r_mem < 500], s=10)


f += 1
plt.close()
func(nwg76_wt)
func(nwg76_rd)
# plt.gca().set_ylim(top=15000)
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))
