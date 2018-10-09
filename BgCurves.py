import IA as x
import numpy as np
import os
import glob

# Save all
r = x.ImportAllBatch2('/Users/blandt/Desktop/Analysis/BgCurves/')
for d in r:
    for e in vars(r[d]):
        if getattr(getattr(r[d], e), 'pro') is not None:
            profile = x.norm_to_bounds(getattr(getattr(r[d], e), 'pro').mean(axis=0), (0, 1), 10)
            np.savetxt('/Users/blandt/Desktop/Analysis/BackgroundCurves/' + '%s_%s.txt' % (d, e), profile)

# Import all
d = {}
for f in glob.glob('/Users/blandt/Desktop/Analysis/BackgroundCurves/*.txt'):
    d[os.path.basename(os.path.normpath(f))[:-4]] = np.loadtxt(f)

####

# # plt.plot(d['kk1273_a'])
# # plt.plot(d['kk1273_c'])
# # plt.plot(d['kk1273_g'])
# plt.plot(d['nwg42_a'], c='k')
# plt.plot(d['nwg42_b'], c='r')
# plt.plot(d['nwg42_c'], c='b')
# plt.plot(d['nwg42_g'], c='g')
# plt.plot(d['nwg42_r'], c='r')
# # plt.plot(d['nwg106_a'])
# # plt.plot(d['nwg106_b'])
# # plt.plot(d['nwg106_c'])
# # plt.plot(d['nwg106_g'])
# # plt.plot(d['nwg106_r'])
# # plt.plot(d['nwg129_a'])
# # plt.plot(d['nwg129_c'])
# # plt.plot(d['nwg129_g'])
# # plt.plot(d['nwg0151_a'])
# # plt.plot(d['nwg0151_b'])
# # plt.plot(d['nwg0151_c'])
# # plt.plot(d['nwg0151_g'])
# # plt.plot(d['nwg0151_r'])
# # plt.plot(d['nwg0151xdr466_a'])
# # plt.plot(d['nwg0151xdr466_b'])
# # plt.plot(d['nwg0151xdr466_c'])
# # plt.plot(d['nwg0151xdr466_g'])
# # plt.plot(d['nwg0151xdr466_r'])
# plt.show()
