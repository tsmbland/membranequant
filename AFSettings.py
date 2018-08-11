from IA import *

N2s0 = Settings()

# N2, Tom3, 15, pfsin
# n2_analysis(['N2/180227_n2_wt_tom3,15,pfsin', 'N2/180417_n2_wt_tom3,15,pfsin'], 0)
N2s1 = Settings(m=2.12371235, c=-331.09130133)

# N2, Tom3, 15, pfsout
# n2_analysis(['N2/180302_n2_wt_tom3,15,pfsout', 'N2/180412_n2_wt_tom3,15,pfsout'], 1)
N2s2 = Settings(m=2.02186652, c=542.73511969)

# N2, Tom3, 5, pfsout
# n2_analysis(['N2/180417_n2_wt_tom3,5,pfsout'], 1)
N2s3 = Settings(m=1.92054323, c=431.56363822)

# N2, PAR2 Nelio
# n2_analysis(['PAR2_Nelio/N2'], 1)
N2s4 = Settings(m=0.85121289, c=193.89370874)

# N2, PAR6 Nelio
# n2_analysis(['PAR6_Nelio/N2'], 1)
N2s5 = Settings(m=0.651245797, c=2395.56862)

# OD70, Tom5, 30, pfsout
# n2_analysis(['180501/180501_od70_wt_tom4,5,30', '180420/180420_od70_wt_tom4,5,30'], 1)
OD70s1 = Settings(m=1.57152176, c=1212.72024584)

# N2, Florent 180110
# n2_analysis(['PKC_rundown_Florent_MP/180110/180110_N2'], 1)
N2s7 = Settings(m=2.9323386, c=343.80295124)

# N2, Florent 180221
# n2_analysis(['PKC_rundown_Florent_MP/180221/180221_N2'], 1)
N2s8 = Settings(m=4.05550647, c=-828.51647995)

# N2, Florent 180301
# n2_analysis(['PKC_rundown_Florent_MP/180301/180301_N2'], 1)
N2s9 = Settings(m=1.72050066, c=858.62079693)

# N2, Tom4, 15, 30, pfsout, after microscope move
# n2_analysis(['N2/180716_n2_wt_tom4,15,30'], 1)
# N2s10 = Settings(m=1.81985846, c=1923.41578)

# N2, Tom4, 15, 30, pfsout, after microscope move
# n2_analysis(['N2/180730_n2_wt_tom4,15,30'], 1)
N2s10 = Settings(m=1.82883199, c=1997.95716, m2=0.90837544, c2=344.00076897)

# N2, Tom4, 15, 30, pfsout, after microscope move, with bleach
# n2_analysis(['N2/180730_n2_wt_tom4,15,30+bleach'], 1)
N2s11 = Settings(m=1.34077353, c=4856.91160)

# plt.show()

