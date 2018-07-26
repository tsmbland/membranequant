from ImageAnalysis.IA import *
import shutil

direc1 = '../PAR6_Nelio_orig'
direc2 = '../PAR6_Nelio'

count = 0
for cond in dirslist(direc1):

    # os.makedirs('%s/%s' % (direc2, cond))
    # count = 0
    # for embryo in dirslist('%s/%s/Embryos' % (direc1, cond)):
    #     os.makedirs('%s/%s/%s' % (direc2, cond, count))
    #
    #     shutil.copyfile(glob.glob('%s/%s/Embryos/%s/NEBD_*_*_w4DIC SP Camera_t1.TIF' % (direc1, cond, embryo))[0],
    #                     '%s/%s/%s/_%s_w4DIC SP Camera.TIF' % (direc2, cond, count, count))
    #     shutil.copyfile(glob.glob('%s/%s/Embryos/%s/NEBD_*_*_w1488 SP 535-50 Nelio_t1.TIF' % (direc1, cond, embryo))[0],
    #                     '%s/%s/%s/_%s_w1488 SP 535-50 Nelio.TIF' % (direc2, cond, count, count))
    #     shutil.copyfile(glob.glob('%s/%s/Embryos/%s/NEBD_*_*_w2488 SP 630-75 Nelio-AF_t1.TIF' % (direc1, cond, embryo))[0],
    #                     '%s/%s/%s/_%s_w2488 SP 630-75 Nelio-AF.TIF' % (direc2, cond, count, count))
    #     shutil.copyfile(glob.glob('%s/%s/Embryos/%s/NEBD_*_*_w3561 SP 630-75 Nelio_t1.TIF' % (direc1, cond, embryo))[0],
    #                     '%s/%s/%s/_%s_w3561 SP 630-75 Nelio.TIF' % (direc2, cond, count, count))
    #     shutil.copyfile(glob.glob('%s/%s/Embryos/%s/NEBD_Cortex_ROI.roi' % (direc1, cond, embryo))[0],
    #                     '%s/%s/%s/NEBD_Cortex_ROI.roi' % (direc2, cond, count))
    #     shutil.copyfile(glob.glob('%s/%s/Embryos/%s/NEBD_*_*.nd' % (direc1, cond, embryo))[0],
    #                     '%s/%s/%s/_%s.nd' % (direc2, cond, count, count))
    #
    #     count += 1

    try:
        for n2 in dirslist('%s/%s/N2' % (direc1, cond)):
            os.makedirs('%s/N2/%s' % (direc2, count))

            shutil.copyfile(glob.glob('%s/%s/N2/%s/NEBD_*_*_w4DIC SP Camera_t1.TIF' % (direc1, cond, n2))[0],
                            '%s/N2/%s/_%s_w4DIC SP Camera.TIF' % (direc2, count, count))
            shutil.copyfile(glob.glob('%s/%s/N2/%s/NEBD_*_*_w1488 SP 535-50 Nelio_t1.TIF' % (direc1, cond, n2))[0],
                            '%s/N2/%s/_%s_w1488 SP 535-50 Nelio.TIF' % (direc2, count, count))
            shutil.copyfile(
                glob.glob('%s/%s/N2/%s/NEBD_*_*_w2488 SP 630-75 Nelio-AF_t1.TIF' % (direc1, cond, n2))[0],
                '%s/N2/%s/_%s_w2488 SP 630-75 Nelio-AF.TIF' % (direc2, count, count))
            shutil.copyfile(glob.glob('%s/%s/N2/%s/NEBD_*_*_w3561 SP 630-75 Nelio_t1.TIF' % (direc1, cond, n2))[0],
                            '%s/N2/%s/_%s_w3561 SP 630-75 Nelio.TIF' % (direc2, count, count))
            shutil.copyfile(glob.glob('%s/%s/N2/%s/NEBD_*_*.nd' % (direc1, cond, n2))[0],
                            '%s/N2/%s/_%s.nd' % (direc2, count, count))
            shutil.copyfile(glob.glob('%s/%s/N2/%s/NEBD_*_*.nd' % (direc1, cond, n2))[0],
                            '%s/N2/%s/_%s.nd' % (direc2, count, count))

            count += 1

    except:
        pass

print(count)
