from Model import *

alg_singlesim(m=Model0, p=paramset0_2, compression=0)

res = loaddata(0, 0, 0)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.25, bottom=0.25)

i = -1
ax.plot(np.array(range(res.p.xsteps)) / (res.p.xsteps - 1) * res.p.L, res.aco[i, :], label='A', c='r')
ax.plot(np.array(range(res.p.xsteps)) / (res.p.xsteps - 1) * res.p.L, res.pco[i, :], label='P', c='dodgerblue')

# ax.axvspan(0, 20, alpha=0.2, color='r')
# ax.axvspan(20, 30, alpha=0.1, color='k')
# ax.axvspan(30, 50, alpha=0.2, color='b')

# ax.axvline(20, c='0.5', linestyle='--', ymax=0.95)
# ax.axvline(30, c='0.5', linestyle='--', ymax=0.95)

# ax.set_title('          Anterior domain', loc='left')
# ax.set_title('Boundary region', loc='center')
# ax.set_title('Posterior domain          ', loc='right')

ax.set_xlabel('x [μm]')
ax.set_ylabel('Concentration [a.u.]')
concmax = max(res.aco.max(), res.pco.max())
ax.set_ylim(0, 1.1 * concmax)


sns.despine()
plt.show()
