from Experiments.e1803__pkc_it71 import *


####################################################################

# Cross section

def func1(dataset, c):
    for x in range(len(dataset.gfp_csection[:, 0])):
        plt.plot(dataset.gfp_csection[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_csection, 0), c=c)


func1(nwg0129_wt, 'g')
func1(kk1228_wt, 'k')
plt.xlabel('Position')
plt.ylabel('Intensity')
sns.despine()
plt.show()


####################################################################

# Spatial distribution

def func1(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c)


func1(nwg0129_wt, 'g')
func1(kk1228_wt, 'k')
plt.xlabel('x / circumference')
plt.ylabel('Cortex / Cytoplasm')
sns.despine()
plt.show()



