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

from tqdm import tqdm

from crossover.driverResponse import DriverResponse
from crossover.crossover import Crossover

import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam

import numpy as np

import crossover.components

def testComponents():

    driverT = DriverResponse('data/AN25F-4@0.frd', 'AN25')
    driverT.plotResponse()

    driverW = DriverResponse('data/TCP115-8@0.frd', 'TCP115')
    driverW.plotResponse()

    resVal = np.log(5E6)
    capVal = np.log(2E-12)

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

    lr = 1E-2
    init_fun, update_fun, get_params = adam(lr)

    res = resVal
    cap = capVal
    highRes = highResVal

    state = init_fun((res, cap, highRes))

    losses = []

    for i in tqdm(range(25)):

        # resGrad, capGrad = flatGrad(res, cap)
        grads = flatGrad(res, cap, highRes)

        state = update_fun(i, grads, state)

        res, cap, highRes = get_params(state)

        flatness = crossover.flatness(res, cap, highRes)
        losses.append(flatness)

    # print("flat", crossover.flatness(res, cap, highRes))
    # print("res", np.exp(res))
    # print("cap", np.exp(cap))
    # print(np.exp(highRes))

    co, hi, lo = crossover.applyCrossover(np.exp(res), np.exp(cap), np.exp(highRes))
    coBefore, _, _ = crossover.applyCrossover(np.exp(resVal), np.exp(capVal), np.exp(highResVal))

    flatNew = crossover.flatness(res, cap, highRes)
    flatOld = crossover.flatness(resVal, capVal, highResVal)

    # print(flatNew)

    plt.plot(crossover.frequencies, co)
    plt.plot(crossover.frequencies, coBefore)
    plt.plot(crossover.frequencies, hi)
    plt.plot(crossover.frequencies, lo)
    plt.xscale('log')
    plt.savefig('crossoverOpt.pdf')
    plt.clf()

    componentsAvailable = components.AvailableComponents('data/resistors.json', 'data/capacitors.json')

    nearestRes = componentsAvailable.nearestRes(np.exp(res))
    nearestHighRes = componentsAvailable.nearestRes(np.exp(highRes))
    nearestCap = componentsAvailable.nearestCap(np.exp(cap))

    # print("res", np.exp(res), nearestRes[1])
    # print("cap", np.exp(cap), nearestCap[1])
    # print("highRes", np.exp(highRes), nearestHighRes[1])

    co, hi, lo = crossover.applyCrossover(nearestRes[1], nearestCap[1], nearestHighRes[1])

    plt.plot(crossover.frequencies, co, label = 'Total')
    plt.plot(crossover.frequencies, hi, label = 'High')
    plt.plot(crossover.frequencies, lo, label = 'Low')
    plt.xscale('log')
    plt.legend(loc = 0, fontsize = 18)
    plt.savefig('crossoverOptNearest.pdf')
    plt.clf()

    plt.plot(losses)
    plt.savefig('losses.pdf')
    plt.clf()
