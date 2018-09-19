from Experiments.e1806__nwg91 import *

plt.rcParams['savefig.dpi'] = 600
fdirec = '../Figures/%s' % os.path.basename(__file__)[:-3]
if not os.path.exists(fdirec):
    os.mkdir(fdirec)
f = 0


####################################################################

# Correlation

def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_spa[x, :], dataset.r_spa[x, :], s=0.5, c=c)


f += 1
plt.close()
# func(par1, c='b')
# func(chin1, c='g')
func(spd5, c='r')
func(wt, c='k')
plt.xlabel('Cortical PKC-3 GFP')
plt.ylabel('Cortical PAR-2 mCherry')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# PAR-2 removal

def func(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.scatter(dataset.g_spa[x, :], dataset.r_cyt[x] / dataset.r_spa[x, :], s=0.5, c=c)


f += 1
plt.close()
func(par1, c='b')
func(chin1, c='g')
func(spd5, c='k')
func(wt, c='r')
plt.xlabel('Cortical PKC-3 GFP')
plt.ylabel('Cytoplasmic / Cortical PAR-2 mCherry')
plt.ylim([0.01, 1000])
sns.despine()
plt.yscale('log')
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Spatial distribution

def func2(dataset, c):
    for x in range(len(dataset.g_spa[:, 0])):
        plt.plot(dataset.g_spa[x, :] / dataset.g_cyt[x], c='g', alpha=0.2, linestyle='-')
        plt.plot(dataset.r_spa[x, :] / dataset.r_cyt[x], c='r', alpha=0.2, linestyle='-')

    plt.plot(np.mean(dataset.g_spa, 0) / np.mean(dataset.g_cyt), c='g', linestyle='-')
    plt.plot(np.mean(dataset.r_spa, 0) / np.mean(dataset.r_cyt), c='r', linestyle='-')


f += 1
plt.close()
func2(wt, c='r')
# func2(par1, c='b')
# func2(chin1, c='g')
# func2(spd5, c='k')
plt.xlabel('x / circumference')
plt.ylabel('Cortex / Cytoplasm (a.u.)')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))


###############################################################

# Spatial distribution with stdev

def func1(dataset, c):
    g_spa = np.concatenate((dataset.g_spa[:, :50], np.flip(dataset.g_spa[:, 50:], 1)))
    g_cyt = np.append(dataset.g_cyt, dataset.g_cyt)
    mean = np.mean(g_spa, 0) / np.mean(g_cyt)
    stdev = np.std(g_spa / np.tile(g_cyt, (50, 1)).T, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


def func2(dataset, c):
    r_spa = np.concatenate((dataset.r_spa[:, :50], np.flip(dataset.r_spa[:, 50:], 1)))
    r_cyt = np.append(dataset.r_cyt, dataset.r_cyt)
    mean = np.mean(r_spa, 0) / np.mean(r_cyt)
    stdev = np.std(r_spa / np.tile(r_cyt, (50, 1)).T, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


f += 1
plt.close()
func1(wt, c='g')
func2(wt, c='r')
# func1(par1, c='g')
# func2(par1, c='r')
# func1(chin1, c='g')
# func2(chin1, c='r')
plt.savefig('%s/f%s.png' % (fdirec, f))


###############################################################

# Spatial distribution with stdev, normalised, a to p

def func1(dataset, c):
    g_spa = np.concatenate((np.flip(dataset.g_spa[:, :50], 1), dataset.g_spa[:, 50:]))
    mean = np.mean(g_spa, 0)
    a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    g_spa = a[0] * g_spa + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(g_spa, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


def func2(dataset, c):
    r_spa = np.concatenate((np.flip(dataset.r_spa[:, :50], 1), dataset.r_spa[:, 50:]))
    mean = np.mean(r_spa, 0)
    a = np.polyfit([min(mean), max(mean)], [0, 1], 1)
    r_spa = a[0] * r_spa + a[1]
    mean = a[0] * mean + a[1]
    stdev = np.std(r_spa, 0)
    plt.fill_between(range(50), mean + stdev, mean - stdev, facecolor=c, alpha=0.2)
    plt.plot(mean, c=c)


f += 1
plt.close()
func1(wt, c='r')
func2(wt, c='c')
# func1(par1, c='g')
# func2(par1, c='r')
# func1(chin1, c='g')
# func2(chin1, c='r')
sns.despine()
plt.xticks([])
plt.yticks([0, 1])
# plt.xlabel('x / circumference')
# plt.ylabel('Cortex / Cytoplasm (a.u.)')
plt.savefig('%s/f%s.png' % (fdirec, f))


####################################################################

# Dosage analysis

def bar(ax, data, pos, name, c):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=5)
    ax.set_xticks(list(ax.get_xticks()) + [pos])
    ax.set_xlim([0, max(ax.get_xticks()) + 4])

    labels = [w.get_text() for w in ax.get_xticklabels()]
    labels += [name]
    ax.set_xticklabels(labels)


f += 1
plt.close()
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax0.set_ylabel('PKC3 dosage')
ax0.set_xticks([])
ax1 = plt.subplot2grid((1, 2), (0, 1))
ax1.set_ylabel('PAR2 dosage')
ax1.set_xticks([])
bar(ax0, wt.g_tot, 4, 'wt', 'k')
bar(ax0, par1.g_tot, 8, 'par1 RNAi', 'k')
bar(ax0, chin1.g_tot, 12, 'chin1 RNAi', 'k')
bar(ax0, spd5.g_tot, 16, 'spd5 RNAi', 'k')
bar(ax1, wt.r_tot, 4, 'wt', 'k')
bar(ax1, par1.r_tot, 8, 'par1 RNAi', 'k')
bar(ax1, chin1.r_tot, 12, 'chin1 RNAi', 'k')
bar(ax1, spd5.r_tot, 16, 'spd5 RNAi', 'k')
sns.despine()
plt.savefig('%s/f%s.png' % (fdirec, f))



################################################################

# Dual channel embryo

# data = Data(wt.direcs[0])
#
# img10 = rotated_embryo(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 300)
# saveimg(img10, 'img10.TIF')
# plt.imshow(img10)
# plt.show()
#
# img11 = straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50)
# saveimg(img11, 'img11.TIF')
# plt.imshow(img11)
# plt.show()
#
# bg = straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50), 50)
# img20 = rotated_embryo(data.RFP - np.nanmean(bg[np.nonzero(bg)]), data.ROI_fitted, 300)
# saveimg(img20, 'img20.TIF')
# plt.imshow(img20)
# plt.show()
#
# img21 = straighten(data.RFP - np.nanmean(bg[np.nonzero(bg)]), data.ROI_fitted, 50)
# saveimg(img21, 'img21.TIF')
# plt.imshow(img21)
# plt.show()
