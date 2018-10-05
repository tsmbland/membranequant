from Model.Simulation import *


########## EQUATIONS


# Nullcline 1
def func1(x):
    y = 0.006 * x / 0.005
    return y


# Nullcline 2
def func2(x):
    y = 0.002 * x / (0.005 - 0.005 * x)
    return y


# Conservation

def func3(x):
    y = (1 - x) / (0.3 * 1)
    return y


def func4(x):
    y = (0.5 - x) / (0.3 * 1)
    return y


def func5(x):
    y = (1 - x) / (0.3 * 0.5)
    return y


######### LEFT PANEL

ax1 = plt.subplot2grid((1, 3), (0, 0))

x = np.linspace(0, 1, 100)
y = func1(x)
ax1.plot(x, y, c='k')

# x = np.linspace(0, 1, 100)
# y = func(x)
# ax1.plot(x, y, linestyle='--')
#
# x = np.linspace(0, 1, 100)
# y = func4(x)
# ax1.plot(x, y, linestyle='--')
#
# x = np.linspace(0, 1, 100)
# y = func5(x)
# ax1.plot(x, y, linestyle='--')

ax1.set_xlim([0, 1])
ax1.set_ylim([0, 2])
ax1.set_xlabel(r'$P_{cyto}$ (a.u.)')
ax1.set_ylabel(r'$P^*$ (a.u.)')
# ax1.set_title(r'$\frac{\partial P}{\partial t} = k_{on} P_{cyto} -  k_{off} P^*$ = 0   (1)', y=1.05)
ax1.set_title('Model without positive feedback', y=1.05)
# ax1.set_title('A', loc='left', y=1.1, fontweight="bold")

# y = 0.006 / 0.0068
# x = 1 - 0.3 * y
# ax1.vlines(x, ymin=0, ymax=y, color='k', linestyle=':', linewidth=1)
# ax1.hlines(y, xmin=0, xmax=x, color='k', linestyle=':', linewidth=1)
#
# y = 0.003 / 0.0068
# x = 0.5 - 0.3 * y
# ax1.vlines(x, ymin=0, ymax=y, color='k', linestyle=':', linewidth=1)
# ax1.hlines(y, xmin=0, xmax=x, color='k', linestyle=':', linewidth=1)
#
# y = 0.006 / 0.0059
# x = 1 - 0.3 * 0.5 * y
# ax1.vlines(x, ymin=0, ymax=y, color='k', linestyle=':', linewidth=1)
# ax1.hlines(y, xmin=0, xmax=x, color='k', linestyle=':', linewidth=1)

######### RIGHT PANEL

ax2 = plt.subplot2grid((1, 3), (0, 1))

x = np.linspace(0, 1, 1000)
y = func2(x)
ax2.plot(x, y, c='k')

# x = np.linspace(0, 1, 100)
# y = func(x)
# ax2.plot(x, y, linestyle='--')
#
# x = np.linspace(0, 1, 100)
# y = func4(x)
# ax2.plot(x, y, linestyle='--')
#
# x = np.linspace(0, 1, 100)
# y = func5(x)
# ax2.plot(x, y, linestyle='--')

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 2])
ax2.set_xlabel(r'$P_{cyto}$ (a.u.)')
ax2.set_ylabel(r'$P^*$ (a.u.)')
# ax2.set_title(r'$\frac{\partial P}{\partial t} = k_{on} P_{cyto} + k_{pos} P_{cyto} P -  k_{off} P^*$ = 0   (2)', y=1.05)
ax2.set_title('Model with positive feedback', y=1.05)
# ax2.set_title('B', loc='left', y=1.1, fontweight="bold")

# y = 0.2836
# x = 0.4149
# ax2.vlines(x, ymin=0, ymax=y, color='k', linestyle=':', linewidth=1)
# ax2.hlines(y, xmin=0, xmax=x, color='k', linestyle=':', linewidth=1)
#
# y = 0.9719
# x = 0.7084
# ax2.vlines(x, ymin=0, ymax=y, color='k', linestyle=':', linewidth=1)
# ax2.hlines(y, xmin=0, xmax=x, color='k', linestyle=':', linewidth=1)
#
# y = 1.4452
# x = 0.7832
# ax2.vlines(x, ymin=0, ymax=y, color='k', linestyle=':', linewidth=1)
# ax2.hlines(y, xmin=0, xmax=x, color='k', linestyle=':', linewidth=1)

########## LEGEND

# ax3 = plt.subplot2grid((1, 3), (0, 2))
# ax3.plot([], [], linestyle='--', label=r'$\rho_P = 1, \psi = 0.3, \omega = 1$')
# ax3.plot([], [], linestyle='--', label=r'$\rho_P = 0.5, \psi = 0.3, \omega = 1$')
# ax3.plot([], [], linestyle='--', label=r'$\rho_P = 1, \psi = 0.3, \omega = 0.5$')
# ax3.axis('off')
# ax3.legend()

###################
sns.despine()
plt.show()
