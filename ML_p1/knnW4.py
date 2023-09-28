import util
import datasets
import binary
import knn
import runClassifier
from numpy import *
from pylab import *

curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':True}), 'K', 
                                         [1,2,3,4,5,6,7,8,9,10], datasets.DigitData)
runClassifier.plotCurve('KNN Hyperparam on Digit Data', curve)

curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':False}), 'eps', 
                                         [4,5,6,7,8,9,10,11,12,13], datasets.DigitData)
runClassifier.plotCurve('KNN Epsilon Hyperparam on Digit Data', curve)

curve = runClassifier.learningCurveSet(knn.KNN({'isKNN':True, 'K':5}), datasets.DigitData)
runClassifier.plotCurve('KNN on Digit Data', curve)