import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

from IA_lite import MembraneQuant, load_image, def_ROI, af_subtraction

# Load images
gfp = load_image('ExampleDataset/488 SP 535-50.TIF')
af = load_image('ExampleDataset/488 SP 630-75.TIF')

# AF subtract
img = af_subtraction(gfp, af, m=2.078827744358308, c=53.083216054020845)
plt.imshow(img, cmap='gray', vmin=0)
plt.show()

# Load background curve
cytbg = np.loadtxt('ExampleDataset/cytbg.txt')

# Specify ROI
coors = def_ROI(img, spline=True)

# Specify destination folder
dest = 'ExampleDataset/Quantification'
# os.mkdir(dest)

# Set up quantifier
q = MembraneQuant(img=img, cytbg=cytbg, coors=coors, thickness=50, rol_ave=20, mem_sigma=2, itp=10, freedom=0.3,
                  end_fix=10, parallel=True, destination=dest)

# Run
t = time.time()
q.run()
print('%s seconds' % (time.time() - t))

# Plot cortical distribution
mems = np.loadtxt(dest + '/mems_1000.txt')
plt.plot(mems)
plt.ylim(bottom=0)
plt.xlabel('Position')
plt.ylabel('Cortical concentration')
plt.show()
