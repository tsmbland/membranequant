from IA import *
import AFSettings as s
import BgCurves as b
import time

# Specify embryo directory
direc = '180309/180309_nwg0123_par3_tom3,15,pfsin/9'

# Specify background curve
bg = b.bgG4

######### SEGMENTATION ##########

# Import data
data = Data(direc)

# Segment
coors = fit_coordinates_alg3(data.RFP, data.ROI_orig, bg, 2)

# Save new coordinates
np.savetxt('%s/ROI_fitted.txt' % data.direc, coors, fmt='%.4f', delimiter='\t')

######## QUANTIFICATION ########

# Import data
data = Data(direc)

# Straighten
img_straight = straighten(data.RFP, data.ROI_fitted, 50)

# Average
profile = np.mean(img_straight, 1)

# Get cortical signal
a, bg2 = fit_background_v2_2(profile, bg[25:75])
signal = profile - bg2
cort = np.trapz(signal)

# Subtract background
mag = 1
bg = straighten(data.RFP, offset_coordinates(data.ROI_fitted, 50 * mag), 50 * mag)
mean1 = np.nanmean(bg[np.nonzero(bg)])

# Get cytoplasmic signal
cyt = cytoconc(data.RFP, data.ROI_fitted)

# Get dosage
total = cyt + (data.sa / data.vol) * cort
