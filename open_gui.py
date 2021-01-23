import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import matplotlib

matplotlib.use("TkAgg")
from membranequant.gui import ImageQuantGUI
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.ion()
    ImageQuantGUI()
