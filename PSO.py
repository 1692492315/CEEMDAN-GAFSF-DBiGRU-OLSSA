import numpy as np
import random
import copy



''' Population initialization function '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
    return X


'''Boundary checking function'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''Function for calculating fitness'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''Fitness sorting'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''Sort positions according to fitness'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''Particle swarm optimization'''
def PSO(pop, dim, lb, ub, MaxIter, fun, Vmin, Vmax):
    w = 0.9
    c1 = 2
    c2 = 2
    X = initial(pop, dim, ub, lb)
    V = initial(pop, dim, Vmax, Vmin)
    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    Pbest = copy.copy(X)
    fitnessPbest = copy.copy(fitness)
    for i in range(MaxIter):
        for j in range(pop):
            V[j, :] = w * V[j, :] + c1 * np.random.random() * (Pbest[j, :] - X[j, :]) + c2 * np.random.random() * (
                        GbestPositon - X[j, :])
            for ii in range(dim):
                if V[j, ii] < Vmin[ii]:
                    V[j, ii] = Vmin[ii]
                if V[j, ii] > Vmax[ii]:
                    V[j, ii] = Vmax[ii]
            X[j, :] = X[j, :] + V[j, :]
            for ii in range(dim):
                if X[j, ii] < lb[ii]:
                    V[j, ii] = lb[ii]
                if X[j, ii] > ub[ii]:
                    V[j, ii] = ub[ii]
            fitness[j] = fun(X[j, :])
            if fitness[j] < fitnessPbest[j]:
                Pbest[j, :] = copy.copy(X[j, :])
                fitnessPbest[j] = copy.copy(fitness[j])
            if fitness[j] < GbestScore[0]:
                GbestScore[0] = copy.copy(fitness[j])
                GbestPositon = copy.copy(X[j, :])

        Curve[i] = GbestScore

    return GbestScore, GbestPositon, Curve