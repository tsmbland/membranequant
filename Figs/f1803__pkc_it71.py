from Experiments.e1803__pkc_it71 import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0


####################################################################

# Cross section

def func1(dataset, c):
    for x in range(len(dataset.g_cse[:, 0])):
        plt.plot(dataset.g_cse[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.g_cse, 0), c=c)


f += 1
plt.close()
func1(nwg0129_wt, 'g')
func1(kk1228_wt, 'k')
plt.xlabel('Position')
plt.ylabel('Intensity')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Spatial distribution

def func1(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.g_spa[x, :] / dataset.g_cyt[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.g_spa, 0) / np.mean(dataset.g_cyt), c=c)


f += 1
plt.close()
func1(nwg0129_wt, 'g')
func1(kk1228_wt, 'k')
plt.xlabel('x / circumference')
plt.ylabel('Cortex / Cytoplasm')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))
