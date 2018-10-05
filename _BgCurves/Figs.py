from IA import *

# Loop through dictionary

# Loop through colours

# Plot cortical circumference, average


r = ImportAllBatch2(adirec + '/BgCurves_Control/')

"""
Check bg curve fitting

"""
for d in r:
    for e in vars(r[d]):
        if getattr(getattr(r[d], e), 'pro') is not None:
            plt.plot(getattr(getattr(r[d], e), 'pro').T, alpha=0.1, c='k')
            plt.plot(getattr(getattr(r[d], e), 'pro').mean(axis=0), c='k')
            plt.plot(getattr(getattr(r[d], e), 'fbc').T, alpha=0.1, c='r')
            plt.plot(getattr(getattr(r[d], e), 'fbc').mean(axis=0), c='r')
            plt.title(d + '_' + e)
            plt.show()
