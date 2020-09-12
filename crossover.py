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

# from jax.config import config
# config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam

from tqdm import tqdm

import components

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
        absHigh = highRes * jnp.abs(self.topologyHigh.transferFunction(self.angularFrequencies))

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

        if res < 0:
            return 1E10
        if cap < 0:
            return 1E10

        response = self.applyCrossover(res, cap, highRes)[0]

        if np.mean(response) < 10.:
            return 1E10

        # return jnp.std(response)

        return jnp.sum((response - jnp.mean(response)) ** 2)

if __name__ == '__main__':

    from driverResponse import DriverResponse

    driverT = DriverResponse('/Users/MBP/Downloads/AN25F-4_data/FRD/AN25F-4@0.frd', 'AN25')
    driverT.plotResponse()

    driverW = DriverResponse('/Users/MBP/Downloads/TCP115-8_data/FRD/TCP115-8@0.frd', 'TCP115')
    driverW.plotResponse()

    resVal = np.log(5E6)
    capVal = np.log(2E-12)

    # resVal = np.log(7E6)
    # capVal = np.log(0.1E-12)
    highResVal = np.log(1)

    resLP = components.Resistor(resVal)
    capLP = components.Capacitor(capVal)

    filterLP = components.Rx([resLP, capLP])

    resHP = components.Resistor(resVal)
    capHP = components.Capacitor(capVal)

    filterHP = components.Rx([capHP, resHP])

    crossover = Crossover(driverW, driverT, filterLP, filterHP)

    plt.plot(crossover.frequencies, crossover.noCrossover())
    plt.xscale('log')
    plt.savefig('test.pdf')
    plt.clf()

    co, hi, lo = crossover.applyCrossover(np.exp(resVal), np.exp(capVal), np.exp(highResVal))

    plt.plot(crossover.frequencies, co)
    plt.plot(crossover.frequencies, hi)
    plt.plot(crossover.frequencies, lo)
    plt.xscale('log')
    plt.savefig('testCrossover.pdf')
    plt.clf()

    flatGrad = jax.grad(crossover.flatness, argnums = [0, 1, 2])

    # print(flatGrad(resVal, capVal))

    lr = 1E-2
    init_fun, update_fun, get_params = adam(lr)

    res = resVal
    cap = capVal
    highRes = highResVal

    state = init_fun((res, cap, highRes))

    losses = []

    for i in tqdm(range(250)):

        # resGrad, capGrad = flatGrad(res, cap)
        grads = flatGrad(res, cap, highRes)

        state = update_fun(i, grads, state)

        res, cap, highRes = get_params(state)

        flatness = crossover.flatness(res, cap, highRes)
        losses.append(flatness)

    print(crossover.flatness(res, cap, highRes))
    print(np.exp(res))
    print(np.exp(cap))
    print(np.exp(highRes))

    co, hi, lo = crossover.applyCrossover(np.exp(res), np.exp(cap), np.exp(highRes))
    coBefore, _, _ = crossover.applyCrossover(np.exp(resVal), np.exp(capVal), np.exp(highResVal))

    flatNew = crossover.flatness(res, cap, highRes)
    flatOld = crossover.flatness(resVal, capVal, highResVal)

    plt.plot(crossover.frequencies, co)
    # plt.plot(crossover.frequencies, coBefore)
    plt.plot(crossover.frequencies, hi)
    plt.plot(crossover.frequencies, lo)
    plt.xscale('log')
    plt.savefig('crossoverOpt.pdf')
    plt.clf()

    plt.plot(losses)
    plt.savefig('losses.pdf')
    plt.clf()
