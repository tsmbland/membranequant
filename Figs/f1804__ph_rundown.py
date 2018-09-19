from Experiments.e1804__ph_rundown import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0

###################################################################

# Cyt vs Mem

f += 1
plt.close()
plt.scatter(ph_rd.g_cyt, ph_rd.g_mem, s=5, facecolors='none', edgecolors='k', linewidth=1)
plt.scatter(ph_wt.g_cyt, ph_wt.g_mem, s=5, facecolors='k', edgecolors='k', linewidth=1)
plt.xlabel('Cytoplasmic PH(PLCδ1) concentration [a.u.]')
plt.ylabel('Cortical PH(PLCδ1) concentration [a.u.]')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))

####################################################################

# Cortex vs ratio

# f += 1
# plt.close()
# plt.scatter(ph_wt.corts, ph_wt.corts / ph_wt.cyts, s=5, facecolors='none', edgecolors='k', linewidth=1)
# plt.scatter(ph_rd.corts, ph_rd.corts / ph_rd.cyts, s=5, facecolors='none', edgecolors='k', linewidth=1)
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# sns.despine()
# plt.rcParams.update({'font.size': 10})
# plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Spatial distribution

def func1(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.g_spa[x, :] / dataset.g_cyt[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.g_spa, 0) / np.mean(dataset.g_cyt), c=c)


def func2(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.r_spa[x, :] / dataset.r_cyt[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.r_spa, 0) / np.mean(dataset.r_cyt), c=c)


f += 1
plt.close()
func1(ph_wt, 'g')
func2(ph_wt, 'r')
plt.gca().set_ylim(bottom=0)
plt.xlabel('x / circumference')
plt.ylabel('Cortex / Cytoplasm (a.u.)')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Cross section

def func1(dataset, c):
    for x in range(len(dataset.g_cse[:, 0])):
        plt.plot(dataset.g_cse[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.g_cse, 0), c=c)


def func2(dataset, c):
    for x in range(len(dataset.g_cse[:, 0])):
        plt.plot(dataset.r_cse[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.r_cse, 0), c=c)


f += 1
plt.close()
func1(ph_wt, 'g')
func2(ph_wt, 'r')
plt.xlabel('Position')
plt.ylabel('Intensity')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))
