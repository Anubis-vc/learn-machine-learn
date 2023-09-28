import util
import datasets
import binary
import perceptron
import runClassifier
from numpy import *
from pylab import *

print(runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.TennisData))
print(runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.TennisData))

print('\n')
runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h = perceptron.Perceptron({'numEpoch': 200})
h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
print(h)
runClassifier.plotClassifier(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y, array([ 7.3, 18.9]), 0.0)

print('\n')
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.SentimentData)
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.SentimentData)