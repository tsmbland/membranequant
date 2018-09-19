from Experiments.e1802__par6_rundown_nelio import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0


# 1 ###################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_spa[x, :], dataset.r_spa[x, :], s=0.5, c=c)


plt.close()
func(nwg26_wt, c='k')
func(nwg26_rd, c='r')
plt.xlabel('Cortical PAR-6')
plt.ylabel('Cortical PAR-2')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


# 2 ###################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_spa[x, :], dataset.r_cyt[x] / dataset.r_spa[x, :], s=0.5, c=c)


plt.close()
func(nwg26_wt, c='k')
func(nwg26_rd, c='k')
# plt.ylim([0, 10])
plt.xlabel('Cortical PAR-6')
plt.ylabel('Cytoplasmic / Cortical PAR-2')
sns.despine()
plt.yscale('log')
plt.savefig('%s/f%s.png' % (fdirec, f))


# 3 ###################################################################

# Spatial distribution

def func2(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.g_spa[x, :] / dataset.g_cyt[x], c=c, alpha=0.2, linestyle='-')
        plt.plot(dataset.r_spa[x, :] / dataset.r_cyt[x], c=c, alpha=0.2, linestyle='--')

    plt.plot(np.mean(dataset.g_spa, 0) / np.mean(dataset.g_cyt), c=c, linestyle='-')
    plt.plot(np.mean(dataset.r_spa, 0) / np.mean(dataset.r_cyt), c=c, linestyle='--')


plt.close()
func2(nwg26_wt, c='r')
plt.savefig('%s/f%s.png' % (fdirec, f))


# 4 ###################################################################

# Dosage correlation

def func4(dataset, c):
    plt.scatter(dataset.g_tot, dataset.r_tot, c=c, s=10)


plt.close()
func4(nwg26_wt, 'r')
func4(nwg26_rd, 'b')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.xlabel('Total PAR6')
plt.ylabel('Total PAR2')
plt.savefig('%s/f%s.png' % (fdirec, f))


# 5 ###################################################################

# PAR-6 dosage vs PAR-2 mem:cyt

def func5(dataset, c):
    plt.scatter(dataset.g_tot, dataset.r_mem / dataset.r_cyt, c=c, s=10)


plt.close()
func5(nwg26_wt, 'r')
func5(nwg26_rd, 'b')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.xlabel('Total PAR6')
plt.ylabel('PAR2 mem:cyt')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


# 6 ###################################################################

# PAR-2 membrane binding


def func5(dataset, c):
    plt.scatter(dataset.r_cyt, dataset.r_mem, c=c, s=10)


plt.close()
func5(nwg26_wt, 'r')
func5(nwg26_rd, 'b')
plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.xlabel('Cytoplasmic PAR-2')
plt.ylabel('Cortical PAR-2')
plt.savefig('%s/f%s.png' % (fdirec, f))


# 7 ###################################################################


def func(dataset, c='k'):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_tot[x],
                    np.mean(dataset.r_spa[x, dataset.g_spa[x, :] < 1000]) / dataset.r_cyt[x], s=10, c=c)


plt.close()
func(nwg26_wt, 'r')
func(nwg26_rd, 'b')
sns.despine()
plt.ylabel('Mean Cortical / Cytoplasmic PAR-2 in regions without PAR-6')
plt.xlabel('Total PAR-6')
plt.gca().set_ylim(bottom=0)
plt.savefig('%s/f%s.png' % (fdirec, f))


# 8 ###################################################################


def func7(dataset):
    for x in range(len(dataset.g_spa[:, 0])):
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Anterior aPAR vs anterior pPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds1),
                    dataset.r_cyt[x] / bounded_mean(dataset.r_spa[x, :], bounds1), c='k', s=10)

        # # Anterior aPAR vs posterior pPAR
        # plt.scatter(bounded_mean(dataset.gfp_spatial[x, :], bounds1),
        #             bounded_mean(dataset.rfp_spatial[x, :], bounds2) / dataset.cyts_RFP[x], c='b', s=10)


plt.close()
func7(nwg26_wt)
func7(nwg26_rd)
# plt.gca().set_ylim(top=15000)
plt.yscale('log')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


# 9 ###############################################################

# Mean PKC vs Local PKC

def func9(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        bounds0 = (0, 1)
        bounds1 = (0.4, 0.6)
        bounds2 = (0.9, 0.1)

        # Total aPAR vs anterior aPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds0),
                    bounded_mean(dataset.g_spa[x, :], bounds1), s=10,
                    c=bounded_mean(dataset.r_spa[x, :], bounds1) / dataset.r_cyt[x], vmin=0, vmax=20,
                    cmap='winter')

        # Total aPAR vs posterior aPAR
        plt.scatter(bounded_mean(dataset.g_spa[x, :], bounds0),
                    bounded_mean(dataset.g_spa[x, :], bounds2), s=10,
                    c=bounded_mean(dataset.r_spa[x, :], bounds2) / dataset.r_cyt[x], vmin=0, vmax=20,
                    cmap='winter')


plt.close()
func9(nwg26_wt, c='k')
func9(nwg26_rd, c='k')
sns.despine()
plt.xlabel('Total PKC-3')
plt.ylabel('Local PKC-3')
plt.savefig('%s/f%s.png' % (fdirec, f))
