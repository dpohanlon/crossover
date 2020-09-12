import json
import functools

import numpy as np

import jax
import jax.numpy as jnp

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

class AvailableComponents(object):

    def __init__(self, resFile, capFile):

        self.resNames, self.resValues = self.readComponentFile(resFile)
        self.capNames, self.capValues = self.readComponentFile(capFile)

    def readComponentFile(self, fileName):

        components = json.load(open(fileName, 'r'))
        componentsList = [(n, v) for n, v in components.items()]

        sortedComponents = sorted(componentsList, key = lambda c : c[1])

        names = [c[0] for c in sortedComponents]
        values = np.array([c[1] for c in sortedComponents])

        return names, values

    def nearest(self, names, values, testValue):

        idx = np.searchsorted(values, testValue)

        if abs(values[idx - 1] - testValue) < abs(values[idx] - testValue):
            idx -= 1

        return names[idx], values[idx]

    def nearestRes(self, testValue):
        return self.nearest(self.resNames, self.resValues, testValue)

    def nearestCap(self, testValue):
        return self.nearest(self.capNames, self.capValues, testValue)

if __name__ == '__main__':

    ac = AvailableComponents('resTest.json', 'capTest.json')

    print(ac.nearest(ac.resNames, ac.resValues, 13000.))
    print(ac.nearestRes(13000.))
