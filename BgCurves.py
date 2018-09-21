from IA import *
import AFSettings as s



def bgcurve(img, pole):
    """

    :param img:
    :param pole: 0 = all over, 1 = anterior, 2 = posterior
    :return:
    """

    # Average
    if pole == 0:
        profile = np.mean(img, 1)
    elif pole == 1:
        profile = np.mean(img[:, int(len(img[0, :]) * 0.3): int(len(img[0, :]) * 0.7)], 1)
    elif pole == 2:
        profile = np.mean(np.hstack((img[:, :int(len(img[0, :]) * 0.2)], img[:, int(len(img[0, :]) * 0.8):])), 1)
    else:
        return

    # Normalise
    line = np.polyfit([np.mean(profile[:10]), np.mean(profile[90:])], [0, 1], 1)
    profile_norm = line[0] * profile + line[1]

    return profile_norm


def func(direc, settings, pole):
    profiles = []
    embryos = direcslist(direc)

    for embryo in embryos:
        # Segment
        data = Data(embryo)
        img = af_subtraction(data.GFP, data.AF, settings=settings)
        coors = fit_coordinates_alg2(img=img, coors=data.ROI_orig, func=calc_offsets2, iterations=2)
        np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')

        # Create profile
        data = Data(embryo)
        img_straight = straighten(img, data.ROI_fitted, 100)
        prof = bgcurve(img_straight, pole)
        profiles.extend([prof])

    return np.mean(np.array(profiles), 0)


############ PAR-1 GFP anterior (PAR-2 mCherry)

# direc = '180322/180322_nwg42_wt_tom4,15,30,pfsout'
# settings = s.N2s2
# bgG1 = func(direc, settings, pole=1)
# np.savetxt('BackgroundCurves/bgG1', bgG1)

bgG1 = np.loadtxt('%s/BackgroundCurves/bgG1' % ddirec)

# ######### PAR-2 mCherry anterior
#
# direcs = ['180322/180322_nwg42_wt_tom4,15,30,pfsout']
# embryoslist = direcslist(direc)
# profiles = np.zeros([len(embryoslist), 50])
#
#
#
# for embryo in range(len(embryoslist)):
#     data = Data('%s/%s' % (direc, embryoslist[embryo]))
#     img = data.RFP_straight
#     profile = np.mean(img[:, int(len(img[0, :]) * 0.3): int(len(img[0, :]) * 0.7)], 1)
#     line = np.polyfit([np.mean(profile[:10]), np.mean(profile[40:])],
#                       [0, 1], 1)
#     profile_norm = line[0] * profile + line[1]
#     profiles[embryo, :] = profile_norm
# bgC1 = np.mean(profiles, 0)
# np.savetxt('BackgroundCurves/bgC1', bgC1)
#
# bgC1 = np.loadtxt('BackgroundCurves/bgC1')


############ PAR-2 mCherry, anterior, one copy

# direcs = ['180804/180804_nwg0151dr466_wt_tom4,15,30']
# embryoslist = embryos_direcslist(direcs)
# profiles = []
#
# for embryo in embryoslist:
#     data = Data(embryo)
#     img_straight = straighten(data.RFP, data.ROI_fitted, 100)
#     prof = bgcurve(img_straight, pole=1)
#     profiles.extend([prof])
#
# bgC2 = np.mean(np.array(profiles), 0)
# np.savetxt('BackgroundCurves/bgC2', bgC2)

bgC2 = np.loadtxt('%s/BackgroundCurves/bgC2' % ddirec)

############# PAR-2 GFP anterior (PH-cherry)

# direc = '180416/180416_nwg0106_wt_tom4,15,30,pfsout'
# settings = s.N2s2
# bgG2 = func(direc, settings, pole=1)
# np.savetxt('BackgroundCurves/bgG2', bgG2)

bgG2 = np.loadtxt('%s/BackgroundCurves/bgG2' % ddirec)

########### PAR-2 GFP anterior

# direc = '180223/180223_kk1273_wt_tom3,15,pfsin'
# settings = s.N2s1
# bgG3 = func(direc, settings, pole=1)
# np.savetxt('BackgroundCurves/bgG3', bgG3)

bgG3 = np.loadtxt('%s/BackgroundCurves/bgG3' % ddirec)

########### PKC GFP all over, PAR-3 mutant

# direc = '180316/180316_nwg0129_par3_tom3,15,pfsout'
# settings = s.N2s2
# bgG4 = func(direc, settings, pole=0)
# np.savetxt('BackgroundCurves/bgG4', bgG4)

bgG4 = np.loadtxt('%s/BackgroundCurves/bgG4' % ddirec)

####################################

# plt.plot(bgC2)
# # plt.plot(bgG1)
# # plt.plot(bgG2)
# # plt.plot(bgG3)
# plt.plot(bgG4)
# plt.show()
