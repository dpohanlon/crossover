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

from tqdm import tqdm

class Component(object):

    def __init__(self):
        super(Component, self).__init__()

    def impedence(self):
        pass

class Resistor(Component):

    def __init__(self, resistance):
        super(Resistor, self).__init__()
        self.resistance = float(resistance)

    def impedence(self, omega):

        return jax.lax.complex(self.resistance * jnp.ones((omega.shape)), 0.)

class Inductor(Component):

    def __init__(self, inductance):
        super(Inductor, self).__init__()
        self.inductance = float(inductance)

    def impedence(self, omega):
        return jax.lax.complex(0., omega * self.inductance)

class Capacitor(Component):

    def __init__(self, capacitance):
        super(Capacitor, self).__init__()
        self.capacitance = float(capacitance)

    def impedence(self, omega):

        return jax.lax.complex(0., -1./(omega * self.capacitance))

class Topology(object):

    def __init__(self, components):
        super(Topology, self).__init__()
        self.components = components

    def transferFunction(self, omega):
        pass

class SallenKey(Topology):

    def __init__(self, components):
        super(SallenKey, self).__init__()
        self.components = components

    def transferFunction(self, omega):

        impedences = [c.impedence(omega) for c in components]

        numerator = impedences[2] * impedences[3]

        denominator = impedences[0] * impedences[1]
        denominator += impedences[2] * (impedences[0] + impedences[1])
        denominator += impedences[2] * impedences[3]

        return numerator / denominator

class RLC(Topology):

    def __init__(self, components):
        super(RLC, self).__init__()
        self.components = components

    def transferFunction(self, omega):

        impedences = [c.impedence(omega) for c in components]

        return impedences[2] / (jax.sum(impedences))

class Rx(Topology):

    def __init__(self, components):
        super(Rx, self).__init__(components)

    def transferFunction(self, omega):

        impedences = jnp.array([c.impedence(omega) for c in self.components])

        return impedences[1, :] / jnp.sum(impedences, axis = 0)

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

    def applyCrossover(self, res, cap):

        self.topologyLow.components[0].resistance = res
        self.topologyLow.components[1].capacitance = cap

        self.topologyHigh.components[0].capacitance = cap
        self.topologyHigh.components[1].resistance = res

        absLow = jnp.abs(self.topologyLow.transferFunction(self.angularFrequencies))
        absHigh = jnp.abs(self.topologyHigh.transferFunction(self.angularFrequencies))

        lowResponse = absLow * self.driverLow(self.frequencies)
        highResponse = absHigh * self.driverHigh(self.frequencies)

        response = lowResponse + highResponse

        return response, lowResponse, highResponse

    def noCrossover(self):

        return self.driverLow(self.frequencies) + self.driverHigh(self.frequencies)

    def flatness(self, res, cap):

        return jnp.std(self.applyCrossover(res, cap)[0])

# if __name__ == '__main__':
#
#     # low pass
#
#     resLP = Resistor(160.)
#     capLP = Capacitor(1E-9)
#
#     filterLP = Rx([resLP, capLP])
#
#     # high pass
#
#     resHP = Resistor(2.4E5)
#     capHP = Capacitor(1E-12)
#
#     filterHP = Rx([capHP, resHP])
#
#     freqs = jnp.array(np.linspace(10, 10E6, 1000))
#     omegas = 2 * np.pi * freqs
#
#     outputHP = filterHP.transferFunction(omegas)
#     outputLP = filterLP.transferFunction(omegas)
#
#     # Need to 'marginalise' over omegas, inputs are component values
#     gradLP = jax.grad(filterHP.stdDev)
#
#     # g = jax.grad(filter.transferFunction, holomorphic = True)
#
#     plt.plot(freqs, np.abs(outputHP))
#     # plt.plot(freqs, np.abs(outputLP))
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.savefig('test.pdf')
#     plt.clf()
#
#     # plt.plot(freqs, np.angle(outputHP))
#     # plt.plot(freqs, np.angle(outputLP))
#     # plt.xscale('log')
#     # plt.savefig('testPhase.pdf')
#     # plt.clf()

if __name__ == '__main__':

    from driverResponse import DriverResponse

    driverT = DriverResponse('/Users/MBP/Downloads/AN25F-4_data/FRD/AN25F-4@0.frd', 'AN25')
    driverT.plotResponse()

    driverW = DriverResponse('/Users/MBP/Downloads/TCP115-8_data/FRD/TCP115-8@0.frd', 'TCP115')
    driverW.plotResponse()

    resVal = 5E6
    capVal = 2E-12

    resLP = Resistor(resVal)
    capLP = Capacitor(capVal)

    filterLP = Rx([resLP, capLP])

    resHP = Resistor(resVal)
    capHP = Capacitor(capVal)

    filterHP = Rx([capHP, resHP])

    crossover = Crossover(driverW, driverT, filterLP, filterHP)

    plt.plot(crossover.frequencies, crossover.noCrossover())
    plt.xscale('log')
    plt.savefig('test.pdf')
    plt.clf()

    co, hi, lo = crossover.applyCrossover(resVal, capVal)

    plt.plot(crossover.frequencies, co)
    plt.plot(crossover.frequencies, hi)
    plt.plot(crossover.frequencies, lo)
    plt.xscale('log')
    plt.savefig('testCrossover.pdf')
    plt.clf()

    flatGrad = jax.grad(crossover.flatness, argnums = [0, 1])

    # print(flatGrad(resVal, capVal))

    lr = 0.01

    res = resVal
    cap = capVal

    for i in range(10):

        resGrad, capGrad = flatGrad(res, cap)

        print(resGrad, capGrad)
        print()

        res += lr * resGrad
        cap += lr * capGrad

        flatness = crossover.flatness(res, cap)

        print(res, cap, flatness)
        print()

    co, hi, lo = crossover.applyCrossover(res, cap)

    # plt.plot(crossover.frequencies, co)
    plt.plot(crossover.frequencies, hi)
    # plt.plot(crossover.frequencies, lo)
    plt.xscale('log')
    plt.savefig('crossoverOpt.pdf')
    plt.clf()
