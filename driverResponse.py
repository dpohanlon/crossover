import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib as mpl
mpl.use('Agg')

plt.style.use(['seaborn-whitegrid', 'seaborn-ticks'])
import matplotlib.ticker as plticker
rcParams['figure.figsize'] = 12, 8
rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['figure.facecolor'] = 'FFFFFF'

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'

rcParams.update({'figure.autolayout': True})

import csv

import numpy as np

from pprint import pprint

from scipy.interpolate import CubicSpline

class DriverResponse(object):

    def __init__(self, fileName):

        super(DriverResponse, self).__init__()

        self.fileName = fileName

        self.readFRD(fileName)
        self.makeSpline()

    def readFRD(self, fileName):

        with open(fileName, 'r') as f:
            reader = csv.reader(f, dialect='excel-tab')
            self.frequencyResponse = [tuple(map(float, r)) for r in reader]

        self.frequencies = [x[0] for x in self.frequencyResponse]
        self.response = [x[1] for x in self.frequencyResponse]

        self.minFreq = min(self.frequencies)
        self.maxFreq = max(self.frequencies)

    def makeSpline(self):

        self.spline = CubicSpline(self.frequencies, self.response)

    def __call__(self, freqs):

        return self.evaluateSpline(freqs)

    def evaluateSpline(self, freqs):

        splineVals = self.spline(freqs)
        splineVals[freqs > self.maxFreq] = 0.
        splineVals[freqs < self.minFreq] = 0.

        return splineVals

    def plotResponse(self, name):

        freqs = np.linspace(self.minFreq, self.maxFreq, 1000)
        plt.plot(freqs, self.spline(freqs), lw = 1.0)
        plt.plot(self.frequencies, self.response, lw = 1.0)
        plt.xscale('log')
        plt.savefig(f'{name}.pdf')
        plt.clf()

if __name__ == '__main__':

    driverT = DriverResponse('/Users/MBP/Downloads/AN25F-4_data/FRD/AN25F-4@0.frd')
    driverT.plotResponse('AN25')

    driverW = DriverResponse('/Users/MBP/Downloads/TCP115-8_data/FRD/TCP115-8@0.frd')
    driverW.plotResponse('TCP115')

    print(driverT(np.array([1.0, 10000., 1E9])))
