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

from tqdm import tqdm

class Component(object):

    def __init__(self):
        super(Component, self).__init__()

    def impedence(self):
        pass

class Resistor(Component):

    def __init__(self, resistance):
        super(Resistor, self).__init__()
        self.resistance = resistance

    def impedence(self, omega):
        return jax.lax.complex(self.resistance, 0)

class Inductor(Component):

    def __init__(self, inductance):
        super(Inductor, self).__init__()
        self.inductance = inductance

    def impedence(self, omega):
        return jax.lax.complex(0, omega * self.inductance)

class Capacitor(Component):

    def __init__(self, capacitance):
        super(Capacitor, self).__init__()
        self.capacitance = capacitance

    def impedence(self, omega):
        return jax.lax.complex(0, -1./(omega * self.capacitance))

class Topology(object):

    def __init__(self, components):
        super(Topology, self).__init__()
        self.components = components

    def transferFunction(self, omega):
        pass

class SallenKey(Topology):

    def __init__(self, components):
        super(Topology, self).__init__()
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
        super(Topology, self).__init__()
        self.components = components

    def transferFunction(self, omega):

        impedences = [c.impedence(omega) for c in components]

        return impedences[2] / (jax.sum(impedences))

class Rx(Topology):

    def __init__(self, components):
        super(Topology, self).__init__()
        self.components = components

    def transferFunction(self, omega):

        impedences = [c.impedence(omega) for c in components]

        return impedences[1] / (jax.sum(impedences))
