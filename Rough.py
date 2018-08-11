from IA import *
# import BgCurves as b
# import AFSettings as s
# import time
# import shutil

# direc = '180302/180302_nwg0123_24hr0par2,par3_tom3,15,pfsout'
#
# embryos = direcslist(direc)
#
# for e in embryos:
#     data = Experiments(e)
#     settings = s.N2s2
#     coors = fit_coordinates_alg2(af_subtraction5(data.GFP, data.AF, settings), data.ROI_orig, calc_offsets, 1)
#
#     # plt.imshow(data.GFP)
#     # plt.plot(coors[:, 0], coors[:, 1])
#     # plt.scatter(coors[0, 0], coors[0, 1])
#     # plt.show()
#
#     np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')


# for d in direcslist('180525'):
#     organise(d)


# for e in direcslist('180309/180309_nwg0123_par3_tom3,15,pfsin'):
#     data = Experiments(e)
#     plt.imshow(data.GFP, cmap='gray')
#     a = offset_coordinates(data.ROI_fitted, 10)
#     a2 = np.vstack((a[:int(len(a[:, 0]) * 0.1), :], a[int(len(a[:, 0]) * 0.9):, :]))
#     plt.scatter(a2[:, 0], a2[:, 1], s=1, c='r')
#     plt.show()

# for e in direcslist('180606/180606_jh1799_48hrpar6_tom3,15'):
#     data = Experiments(e)
#     settings = s.N2s2
#     coors = fit_coordinates_alg2(af_subtraction5(data.GFP, data.AF, settings), data.ROI_orig, calc_offsets, 1)
#     np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')


# plt.imshow(data.GFP)
# plt.plot(coors[:, 0], coors[:, 1])
# plt.scatter(coors[0, 0], coors[0, 1])
# plt.show()


# organise('180607/180607_nwg91_wt_tom4,15,30')


# direc = '180309/180309_nwg0123_par3_tom3,15,pfsin/0'
# data = Experiments(direc)
# settings = s.N2s1
#
# corrected = af_subtraction2(data.GFP, data.AF, m=settings.m, c=settings.c, x=settings.x)
#
# print('start')
# start = time.time()
# coors = fit_coordinates_alg3(corrected, data.ROI_orig, b.blG1_wide, 2)
# end = time.time()
# print(end - start)
#
# plt.imshow(data.GFP)
# plt.plot(coors[:, 0], coors[:, 1])
# plt.scatter(coors[0, 0], coors[0, 1])
# plt.show()
#
# plt.imshow(straighten(corrected, coors, 50))
# plt.show()

# direc = '180223/180223_nwg0123_24hr0par2,par3_tom3,15,pfsin/'
# settings = s.N2s1
#
# for e in direcslist(direc):
#     data = Experiments(e)
#     corrected = af_subtraction2(data.GFP, data.AF, m=settings.m, c=settings.c, x=settings.x)
#     corrected_straight = straighten(corrected, data.ROI_fitted, 50)
#     profile = np.mean(corrected_straight, 1)
#     a, bg = fit_background_v2_2(profile, b.bgG1)
#     plt.plot(gaussian_plus2(b.bgG1, *a))
#     plt.plot(profile)
#     plt.plot(bg)
#     plt.show()

# print(direcslist('180611'))
# organise('180611/180611_jh1799_wt_tom3,15')

# print(direcslist('180525')[1:3])

# corrected = af_subtraction2(data.GFP, data.AF, m=settings.m, c=settings.c, x=settings.x)
# corrected_straight = straighten(corrected, data.ROI_fitted, 50)
# profile = np.mean(corrected_straight, 1)
#
# print('starting')
# start = time.time()
# a = fit_background_v2(profile, b.blG1_wide)
# end = time.time()
# print(end - start)
#
# print(a)
# plt.plot(profile)
# # plt.plot(a[0]*b.blG1_wide + a[1])
# plt.plot(range(50), gaussian_plus(b.blG1_wide, *a))
# plt.show()


# direc = '180622'
#
# for d in direcslist(direc):
#     organise(d)

# direcs = ['180607/180607_nwg91_wt_tom4,15,30',
#           '180611/180611_nwg91_24hrchin1_tom4,15,30',
#           '180611/180611_nwg91_24hrpar1_tom4,15,30',
#           '180611/180611_nwg91_24hrspd5_tom4,15,30']
#
# for d in direcs:
#     for e in direcslist(d):
#         data = Data(e)
#         cytoplasmic_signal_RFP(data)
#
# plt.show()

# organise('180316/180316_kk1228_wt_tom3,15,pfsout', start=3)

# direc = '180420/180420_nwg1_wt_tom4,5,30'
#
# ax0 = plt.subplot2grid((1, 4), (0, 0))
# ax1 = plt.subplot2grid((1, 4), (0, 1))
# ax2 = plt.subplot2grid((1, 4), (0, 2))
# ax3 = plt.subplot2grid((1, 4), (0, 3))
#
#
# for e in direcslist(direc):
#     data = Data(e)
#
#     ax0.imshow(rotated_embryo(data.DIC, data.ROI_fitted, 300), cmap='gray', vmin=5000, vmax=60000)
#     ax0.set_xticks([])
#     ax0.set_yticks([])
#     ax0.set_title('DIC')
#
#     ax1.imshow(rotated_embryo(data.GFP, data.ROI_fitted, 300), cmap='gray', vmin=3000, vmax=50000)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_title('488 535-50')
#
#     ax2.imshow(rotated_embryo(data.AF, data.ROI_fitted, 300), cmap='gray', vmin=500, vmax=20000)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_title('488 630-75')
#
#     ax3.imshow(rotated_embryo(data.RFP, data.ROI_fitted, 300), cmap='gray', vmin=500, vmax=20000)
#     ax3.set_xticks([])
#     ax3.set_yticks([])
#     ax3.set_title('561 630-75')
#
#     plt.show()


# organise('180712/180712_sv2061_wt_tom4,15,30,100msecG')


# direc = 'PKC_rundown_Florent/180221/180221_NWG91_50i'
#
# a = glob.glob('%s/*' % direc)

# for x in a:
#     print(os.path.basename(x))

# print(a)

# for x in range(len(a)//4):
#     os.makedirs('%s/%s' % (direc, x))
#
#     # print(glob.glob('%s/*' % direc)[2 * x])
#
#     # print(os.path.basename(glob.glob('%s/*' % direc)[2 * x]))
#     # print(os.path.basename(glob.glob('%s/*' % direc)[2 * x + 1]))
#
#     # shutil.move(a[3 * x], '%s/%s/%s' % (direc, x, os.path.basename(a[3 * x])))
#     # shutil.move(a[3 * x + 1], '%s/%s/%s' % (direc, x, os.path.basename(a[3 * x + 1])))
#     # shutil.move(a[3 * x + 2], '%s/%s/%s' % (direc, x, os.path.basename(a[3 * x + 2])))
#
#     # shutil.move(a[2 * x], '%s/%s/%s' % (direc, x, os.path.basename(a[2 * x])))
#     # shutil.move(a[2 * x + 1], '%s/%s/%s' % (direc, x, os.path.basename(a[2 * x + 1])))
#
#     shutil.move(a[4 * x], '%s/%s/%s' % (direc, x, os.path.basename(a[4 * x])))
#     shutil.move(a[4 * x + 1], '%s/%s/%s' % (direc, x, os.path.basename(a[4 * x + 1])))
#     shutil.move(a[4 * x + 2], '%s/%s/%s' % (direc, x, os.path.basename(a[4 * x + 2])))
#     shutil.move(a[4 * x + 3], '%s/%s/%s' % (direc, x, os.path.basename(a[4 * x + 3])))


# split_stage_positions('180803/180803_nwg0158_pmawashin,perm,dmso_tom4,15,30/43', start=1)
# organise('180804/180804_nwg0151dr466_wt_tom4,15,30', start=0)

# organise('180804/180804_nwg0123_wt_tom4,15,30')



# Legend

# plt.plot(0, 0, c='k', label='wild type')
# plt.plot(0, 0, c='g', label='par-3 -/-')
# plt.plot(0, 0, c='b', label='par-2 (s241a)*')
# plt.legend()
# plt.rcParams['savefig.dpi'] = 600
# sns.despine()
# plt.show()
