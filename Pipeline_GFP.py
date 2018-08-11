import AFSettings as s
import BgCurves as b
from IA import *

# Specify embryo directory
# direc = '180420/180420_nwg1_0920xfp_tom4,5,30/Test!'
direc = '180309/180309_nwg0123_par3_tom3,15,pfsin/Test!'
# direc = '180726/180726_nw62_wt_tom4,15,30/Test!'

# Specify af settings and background curve
settings = s.OD70s1
bg = b.bgG2
d = Data


######### SEGMENTATION ##########

# Import data
data = d(direc)

# Correct autofluorescence
img = af_subtraction(data.GFP, data.AF, settings=settings)

# Segment
coors = fit_coordinates_alg3(img, data.ROI_orig, bg, 2)

# Save new coordinates
np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')

# ######## QUANTIFICATION ########
#
# # Import data
# data = Data(direc)
#
# # Correct autofluorescence
# img = af_subtraction(data.GFP, data.AF, settings=settings)
#
# # Straighten
# img_straight = straighten(img, data.ROI_fitted, 50)
#
# # Average
# profile = np.mean(img_straight, 1)
#
# # Get cortical signal
# a, bg2 = fit_background_v2_2(profile, bg[25:75])
# signal = profile - bg2
# cort = np.trapz(signal)
#
# # Get cytoplasmic signal
# cyt = cytoconc(img, data.ROI_fitted)
#
# # Get dosage
# total = cyt + (data.sa / data.vol) * cort


# Check segmentation
data = d(direc)
print(data.direc)

# plt.imshow(data.RFP)
# plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
# plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
# plt.show()

plt.imshow(straighten(data.GFP, data.ROI_fitted, 50))
plt.show()


# # Check bg fitting <- looks a bit off
# data = Data(direc)
# print(data.direc)
# img = af_subtraction(data.GFP, data.AF, settings=s.N2s2)
# img_straight = straighten(img, data.ROI_fitted, 50)
# profile = np.nanmean(np.hstack(
#     (img_straight[:, :int(len(img_straight[0, :]) * 0.1)], img_straight[:, int(len(img_straight[0, :]) * 0.9):])),
#     1)
# bg = fit_background_v2_2(profile, b.bgG4[25:75])
# signal = profile - bg
# plt.plot(profile)
# plt.plot(bg)
# # plt.plot(gaussian_plus2(b.bgG4[25:75], *a))
# plt.show()
