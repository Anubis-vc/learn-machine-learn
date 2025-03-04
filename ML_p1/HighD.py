from math import *
import random
from numpy import *
import matplotlib.pyplot as plt
import datasets

waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]

def computeExampleDistance(x1, x2):
    dist = 0.0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)

def computeDistances(data):
    N = len(data)
    D = len(data[0])
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append( computeExampleDistance(data[n],data[m])  / sqrt(D))
    return dist

# N    = 200                   # number of examples
# Dims = [784]   # dimensionalities to try
# Cols = ['#FF0000']
# Bins = arange(0.8, 1, 0.005)

# plt.xlabel('distance / sqrt(dimensionality)')
# plt.ylabel('# of pairs of points at that distance')
# plt.title('dimensionality versus uniform point distances')

# for i,d in enumerate(Dims):
#     distances = computeDistances(datasets.DigitData.Y)
#     print("D=%d, average distance=%g" % (d, mean(distances) * sqrt(d)))
#     plt.hist(distances,
#              Bins,
#              histtype='step',
#              color=Cols[i])
#     if waitForEnter:
#         plt.legend(['%d dims' % d for d in Dims])
#         plt.show(False)
#         x = raw_input('Press enter to continue...')


# plt.legend(['%d dims' % d for d in Dims])
# plt.savefig('fig.pdf')
plt.hist(computeDistances(datasets.DigitData.X))
plt.show()