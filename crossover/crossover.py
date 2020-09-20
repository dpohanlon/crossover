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

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam

from tqdm import tqdm

import argparse

import crossover.components
from crossover.driverResponse import DriverResponse

class Crossover(object):

    # Just two way for now

    def __init__(self, driverLow, driverHigh, topologyLow, topologyHigh):
        super(Crossover, self).__init__()

        self.driverLow = driverLow
        self.driverHigh = driverHigh

        self.topologyLow = topologyLow
        self.topologyHigh = topologyHigh

        self.minOverlap = max(driverLow.minFreq, driverHigh.minFreq)
        self.maxOverlap = min(driverLow.maxFreq, driverHigh.maxFreq)

        self.frequencies = jnp.linspace(self.minOverlap, self.maxOverlap, 1000)
        self.angularFrequencies = 2. * np.pi * self.frequencies + 5E4

    def applyCrossover(self, res, cap, highRes):

        self.topologyLow.components[0].resistance = res
        self.topologyLow.components[1].capacitance = cap

        self.topologyHigh.components[0].capacitance = cap
        self.topologyHigh.components[1].resistance = res

        absLow = jnp.abs(self.topologyLow.transferFunction(self.angularFrequencies))
        absHigh = (1./highRes) * jnp.abs(self.topologyHigh.transferFunction(self.angularFrequencies))

        lowResponse = absLow * self.driverLow(self.frequencies)
        highResponse = absHigh * self.driverHigh(self.frequencies)

        response = lowResponse + highResponse

        return response, lowResponse, highResponse

    def noCrossover(self):

        return self.driverLow(self.frequencies) + self.driverHigh(self.frequencies)

    def flatness(self, logRes, logCap, logHighRes):

        res = jnp.exp(logRes)
        cap = jnp.exp(logCap)
        highRes = jnp.exp(logHighRes)

        response = self.applyCrossover(res, cap, highRes)[0]

        if np.mean(response) < 10.:
            return 1E10

        flat = jnp.sum((response - jnp.mean(response)) ** 2)

        return flat

def setupDriverCrossover(highFileName = 'data/AN25F-4@0.frd',
                         highDriverName = 'AN25',
                         lowFileName = 'data/TCP115-8@0.frd',
                         lowDriverName = 'TCP115'):

    driverT = DriverResponse(highFileName, highDriverName)
    driverT.plotResponse()

    driverW = DriverResponse(lowFileName, lowDriverName)
    driverW.plotResponse()

    resVal = np.log(5E6)
    capVal = np.log(2E-12)

    resLP = components.Resistor(resVal)
    capLP = components.Capacitor(capVal)

    filterLP = components.Rx([resLP, capLP])

    resHP = components.Resistor(resVal)
    capHP = components.Capacitor(capVal)

    filterHP = components.Rx([capHP, resHP])

    crossover = Crossover(driverW, driverT, filterLP, filterHP)

    return crossover

def optimiseCrossover(highFileName = 'data/AN25F-4@0.frd',
                      highDriverName = 'AN25',
                      lowFileName = 'data/TCP115-8@0.frd',
                      lowDriverName = 'TCP115',
                      dataDir = 'data',
                      plotName = 'opt',
                      learningRate = 1E-2,
                      epochs = 25):

    crossover = setupDriverCrossover(highFileName, highDriverName, lowFileName, lowDriverName)

    flatGrad = jax.grad(crossover.flatness, argnums = [0, 1, 2])

    init_fun, update_fun, get_params = adam(learningRate)

    res = np.log(5E6)
    cap = np.log(2E-12)
    highRes = np.log(1)

    state = init_fun((res, cap, highRes))

    losses = []

    for i in tqdm(range(epochs)):

        grads = flatGrad(res, cap, highRes)

        state = update_fun(i, grads, state)

        res, cap, highRes = get_params(state)

        flatness = crossover.flatness(res, cap, highRes)
        losses.append(flatness)

    availableComponents = components.AvailableComponents(f'{dataDir}/resistors.json', f'{dataDir}/capacitors.json')

    nearestRes = availableComponents.nearestRes(np.exp(res))
    nearestHighRes = availableComponents.nearestRes(np.exp(highRes))
    nearestCap = availableComponents.nearestCap(np.exp(cap))

    co, hi, lo = crossover.applyCrossover(nearestRes[1], nearestCap[1], nearestHighRes[1])

    plt.plot(crossover.frequencies, co, label = 'Total')
    plt.plot(crossover.frequencies, hi, label = 'High')
    plt.plot(crossover.frequencies, lo, label = 'Low')

    plt.xscale('log')

    plt.legend(loc = 0, fontsize = 18)
    plt.xlabel('Frequency [Hz]', fontsize = 16)
    plt.ylabel('Sound pressure level [dB]', fontsize = 16)

    plt.savefig(f'{plotName}.pdf')
    plt.clf()

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-lf", "--lowFileName", type = str, dest = "lowFileName", default = 'data/TCP115-8@0.frd', help = 'Low frequency driver FRD file location.')
    argParser.add_argument("-ld", "--lowDriverName", type = str, dest = "lowDriverName", default = 'TCP115', help = 'Low frequency driver name.')

    argParser.add_argument("-hf", "--highFileName", type = str, dest = "highFileName", default = 'data/AN25F-4@0.frd', help = 'High frequency driver FRD file location.')
    argParser.add_argument("-hd", "--highDriverName", type = str, dest = "highDriverName", default = 'AN25', help = 'High frequency driver name.')

    argParser.add_argument("--dataDir", type = str, dest = "dataDir", default = 'data', help = 'Component data directory.')

    argParser.add_argument("--plotName", type = str, dest = "plotName", default = 'opt', help = 'Frequency plot name.')

    argParser.add_argument("-lr", "--learningRate", type = float, dest = "learningRate", default = 1E-2, help = 'Adam learning rate.')
    argParser.add_argument("-e", "--epochs", type = int, dest = "epochs", default = 25, help = 'Number of epochs to train for.')

    args = argParser.parse_args()

    optimiseCrossover(highFileName = args.highFileName,
                      highDriverName = args.highDriverName,
                      lowFileName = args.lowFileName,
                      lowDriverName = args.lowDriverName,
                      dataDir = args.dataDir,
                      plotName = args.plotName,
                      learningRate = args.learningRate,
                      epochs = args.epochs)
