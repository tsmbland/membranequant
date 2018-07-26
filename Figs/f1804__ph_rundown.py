from Experiments.e1804__ph_rundown import *

###################################################################

# Cyt vs Mem

plt.scatter(ph_rd.cyts_GFP, ph_rd.corts_GFP, s=5, facecolors='none', edgecolors='k', linewidth=1)
plt.scatter(ph_wt.cyts_GFP, ph_wt.corts_GFP, s=5, facecolors='k', edgecolors='k', linewidth=1)
plt.xlabel('Cytoplasmic PH(PLCδ1) concentration [a.u.]')
plt.ylabel('Cortical PH(PLCδ1) concentration [a.u.]')

sns.despine()
plt.show()


####################################################################

# Cortex vs ratio

# plt.scatter(ph_wt.corts, ph_wt.corts / ph_wt.cyts,  s=5, facecolors='none', edgecolors='k', linewidth=1)
# plt.scatter(ph_rd.corts, ph_rd.corts / ph_rd.cyts, s=5, facecolors='none', edgecolors='k', linewidth=1)
#
# plt.gca().set_ylim(bottom=0)
# plt.gca().set_xlim(left=0)
# sns.despine()
# plt.rcParams.update({'font.size': 10})
# plt.show()


####################################################################

# Spatial distribution

def func1(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.gfp_spatial[x, :] / dataset.cyts_GFP[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_spatial, 0) / np.mean(dataset.cyts_GFP), c=c)


def func2(dataset, c):
    for x in range(len(dataset.gfp_spatial[:, 0])):
        plt.plot(dataset.rfp_spatial[x, :] / dataset.cyts_RFP[x], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.rfp_spatial, 0) / np.mean(dataset.cyts_RFP), c=c)


# func1(ph_wt, 'g')
# func2(ph_wt, 'r')
# plt.gca().set_ylim(bottom=0)
# plt.xlabel('x / circumference')
# plt.ylabel('Cortex / Cytoplasm (a.u.)')
# sns.despine()
# plt.show()


####################################################################

# Cross section

def func1(dataset, c):
    for x in range(len(dataset.gfp_csection[:, 0])):
        plt.plot(dataset.gfp_csection[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.gfp_csection, 0), c=c)


def func2(dataset, c):
    for x in range(len(dataset.gfp_csection[:, 0])):
        plt.plot(dataset.rfp_csection[x, :], c=c, alpha=0.2)
    plt.plot(np.mean(dataset.rfp_csection, 0), c=c)



# func1(ph_wt, 'g')
# func2(ph_wt, 'r')
# plt.xlabel('Position')
# plt.ylabel('Intensity')
# sns.despine()
# plt.show()


