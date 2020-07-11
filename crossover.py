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

    def stdDev(self, omega):
        return jnp.std(self.transferFunction(omega))

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

    driver = DriverResponse('/Users/MBP/Downloads/AN25F-4_data/FRD/AN25F-4@0.frd')

    # high pass

    resHP = Resistor(5E6)
    capHP = Capacitor(2E-12)

    filterHP = Rx([capHP, resHP])

    # Shift the frequencies just to test
    freqs = np.array(driver.frequencies) + 5E4
    omegas = 2 * np.pi * freqs

    outputHP = filterHP.transferFunction(omegas)

    response = outputHP * np.array(driver.response)

    plt.plot(freqs, response)
    # plt.plot(freqs, driver.response)
    # plt.plot(freqs, outputHP)
    plt.xscale('log')
    # plt.yscale('log')
    plt.savefig('test.pdf')
    plt.clf()
