from ImageAnalysis import *

direc = '../../../../Desktop/180501/180501_kk1273_wt_tom3,15+ph'

embryos = dirslist(direc)
ph_profiles = np.zeros([len(embryos), 1000])
sp_profiles = np.zeros([len(embryos), 1000])

for embryo in range(len(embryos)):
    sp = loadimage('%s/%s/SP_straight.tif' % (direc, embryos[embryo]))
    ph = loadimage('%s/%s/PH_straight.tif' % (direc, embryos[embryo]))

    sp_profile = np.interp(np.linspace(0, len(sp[0, :]), 1000), range(len(sp[0, :])), np.mean(sp, 0))
    ph_profile = np.interp(np.linspace(0, len(sp[0, :]), 1000), range(len(ph[0, :])), np.mean(ph, 0))

    sp_profiles[embryo, :] = sp_profile
    ph_profiles[embryo, :] = ph_profile

    # plt.plot(ph_profile, c='0.7')
    # plt.plot(sp_profile, c='0.7')

# Normalise profiles
line = np.polyfit([max(np.mean(sp_profiles, 0)), np.mean(np.mean(sp_profiles, 0)[-10:])],
                  [max(np.mean(ph_profiles, 0)), np.mean(np.mean(ph_profiles, 0)[-10:])], 1)

plt.plot(np.mean(ph_profiles, 0))
plt.plot(line[0] * np.mean(sp_profiles, 0) + line[1])
plt.show()

# plt.plot(np.mean(ph_profiles, 0), c='k')
# plt.plot(np.mean(sp_profiles, 0), c='k')


plt.xlabel('x')
plt.ylabel('Intensity')
plt.show()
