from math import *
import random
from numpy import *
import matplotlib.pyplot as plt
import util
import datasets

waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]

def computeExampleDistance(x1, x2, d):
    dist = 0.0
    arr = [] # create array to hold all numbers
    for i in range(0, d):
        arr += [i]
    util.permute(arr) # permute array for random sample

    for j in range(d): # now do the same distance calc on the first d samples
        dist += (x1[arr[j]] - x2[arr[j]]) * (x1[arr[j]] - x2[arr[j]])
    return sqrt(dist)

def computeDistancesSubdim(data, d):
    N = len(data)
    print(N)
    D = len(data[0])
    print(D)
    dist = []
    for n in range(200):
        for m in range(n):
            dist.append(computeExampleDistance(data[n], data[m], d) / sqrt(D))
    return dist

N= 200
Digits = [2, 8, 32, 128, 512]   # dimensionalities to try
Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
Bins = arange(0, 1, 0.02)

plt.xlabel('distance / sqrt(dimensionality)')
plt.ylabel('# of pairs of points at that distance')
plt.title('dimensionality versus uniform point distances')

for i,d in enumerate(Digits):
    distances = computeDistancesSubdim(generateUniformDataset(d, N), d)
    print("D=%d, average distance=%g" % (d, mean(distances) * sqrt(d)))
    plt.hist(distances,
             Bins,
             histtype='step',
             color=Cols[i])
    if waitForEnter:
        plt.legend(['%d dims' % d for d in Digits])
        plt.show(False)
        x = raw_input('Press enter to continue...')


plt.legend(['%d dims' % d for d in Digits])
plt.savefig('fig.pdf')
plt.show()