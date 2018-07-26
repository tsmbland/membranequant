from ImageAnalysis.IA import *
import ImageAnalysis.AFSettings as s


direc = '180627/180627_screening_tom4,15,30'

for n, d in enumerate(direcslist(direc)):
    for e in direcslist(d):
        data = Data(e)
        img = af_subtraction(data.GFP, data.AF, settings=s.N2s2)

        img2 = polycrop(img, data.ROI_orig, -20)
        mean = np.mean(img2[np.nonzero(img2)])

        plt.scatter(n + 1, mean)

plt.gca().set_ylim(bottom=0)
plt.gca().set_xlim(left=0)
plt.show()
