from ImageAnalysis import *


cherry = []
leakage = []
bgs = []

list1 = ['2', '6', '7', '8', '10']

direc = '../../../../Desktop/180501/180501_od70_wt_tom4,5,30'
embryos = dirslist(direc)
for e in range(len(embryos)):
    data = Data('%s/%s' % (direc, embryos[e]))
    cyto = cytoconc(data.RFP, data.ROI_fitted)
    cherry.extend([cyto])

    GFP = cytoconc(data.GFP, data.ROI_fitted)
    AF = cytoconc(data.AF, data.ROI_fitted)
    N2 = (GFP - 431.56367848) / 1.92054322
    leakage.extend([AF - N2])

    if embryos[e] in list1:
        bg = reverse_polycrop(data.RFP, data.ROI_fitted, 10)
        bgconc = np.mean(bg[np.nonzero(bg)])
        bgs.extend([bgconc])


direc = '../../../../Desktop/Experiments/180420/180420_od70_wt_tom4,5,20(30)?'
embryos = dirslist(direc)
for e in range(len(embryos)):
    data = Data('%s/%s' % (direc, embryos[e]))
    cyto = cytoconc(data.RFP, data.ROI_fitted)
    cherry.extend([cyto])

    GFP = cytoconc(data.GFP, data.ROI_fitted)
    AF = cytoconc(data.AF, data.ROI_fitted)
    N2 = (GFP - 431.56367848) / 1.92054322
    leakage.extend([AF - N2])

    if embryos[e] in list1:
        bg = reverse_polycrop(data.RFP, data.ROI_fitted, 10)
        bgconc = np.mean(bg[np.nonzero(bg)])
        bgs.extend([bgconc])


leakage = np.array(leakage)
cherry = np.array(cherry)
bgs = np.array(bgs)

plt.scatter(cherry - np.mean(bgs), leakage)
# print(np.mean(AFs))
print(cherry - np.mean(bgs))
# print(np.mean(AFs) / np.mean(signals - np.mean(bgs)))

# print(signals - np.mean(bgs))

plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.show()


