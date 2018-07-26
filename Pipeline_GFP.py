from IA import *
import AFSettings as s
import BgCurves as b

# Specify embryo directory
direc = '180607/180607_nwg91_wt_tom4,15,30/Test!'

# Specify af settings and background curve
settings = s.N2s2
bg = b.bgG4
d = Data

######### SEGMENTATION ##########

# Import data
data = d(direc)

# # Correct autofluorescence
# img = af_subtraction(data.GFP, data.AF, settings=settings)
#
# # Segment
# # coors = fit_coordinates_alg3(img, data.ROI_orig, bg, 2)
# coors = fit_coordinates_alg4(data, settings, bg, iterations=2, mag=1)
#
# # Save new coordinates
# np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')
#
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


# data = d(direc)
# print(data.direc)
#
# plt.imshow(af_subtraction(data.GFP, data.AF, s.N2s2), cmap='gray')
# plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
# plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
# plt.show()
#
# plt.imshow(data.RFP, cmap='gray')
# plt.plot(data.ROI_fitted[:, 0], data.ROI_fitted[:, 1])
# plt.scatter(data.ROI_fitted[0, 0], data.ROI_fitted[0, 1])
# plt.show()
#
# plt.imshow(straighten(af_subtraction(data.GFP, data.AF, s.N2s2), data.ROI_fitted, 50), cmap='gray')
# plt.show()
#
# plt.imshow(straighten(data.RFP, data.ROI_fitted, 50), cmap='gray')
# plt.show()

plt.imshow(composite(data, settings))
plt.show()
