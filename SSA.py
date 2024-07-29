import random
import copy
import numpy as np



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


'''Producers update'''
def PDUpdate(X, PDNumber, ST, Max_iter, dim):
    X_new = copy.copy(X)
    R2 = random.random()
    for j in range(PDNumber):
        if R2 < ST:
            X_new[j, :] = X[j, :] * np.exp(-j / (random.random() * Max_iter))
        else:
            X_new[j, :] = X[j, :] + np.random.randn() * np.ones([1, dim])
    return X_new


'''Scroungers update'''
def JDUpdate(X, PDNumber, pop, dim):
    X_new = copy.copy(X)
    for j in range(PDNumber + 1, pop):
        if j > (pop - PDNumber) / 2 + PDNumber:
            X_new[j, :] = np.random.randn() * np.exp((X[-1, :] - X[j, :]) / j ** 2)
        else:
            A = np.ones([dim, 1])
            for a in range(dim):
                if (random.random() > 0.5):
                    A[a] = -1
            AA = np.dot(A, np.linalg.inv(np.dot(A.T, A)))
            X_new[j, :] = X[0, :] + np.abs(X[j, :] - X[0, :]) * AA.T
    return X_new


'''Vigilantes update'''
def SDUpdate(X, pop, SDNumber, fitness, BestF):
    X_new = copy.copy(X)
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[0:SDNumber]
    for j in range(SDNumber):
        if fitness[SDchooseIndex[j]] > BestF:
            X_new[SDchooseIndex[j], :] = X[0, :] + np.random.randn() * np.abs(X[SDchooseIndex[j], :] - X[0, :])
        elif fitness[SDchooseIndex[j]] == BestF:
            K = 2 * random.random() - 1
            X_new[SDchooseIndex[j], :] = X[SDchooseIndex[j], :] + K * (
                        np.abs(X[SDchooseIndex[j], :] - X[-1, :]) / (fitness[SDchooseIndex[j]] - fitness[-1] + 10E-8))
    return X_new


'''Sparrow search algorithm'''
def SSA(pop, dim, lb, ub, Max_iter, fun):
    ST = 0.6
    PD = 0.7
    SD = 0.2
    PDNumber = int(pop * PD)
    SDNumber = int(pop * SD)
    X = initial(pop, dim, ub, lb)
    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])
    for i in range(Max_iter):
        BestF = fitness[0]
        X = PDUpdate(X, PDNumber, ST, Max_iter, dim)
        X = JDUpdate(X, PDNumber, pop, dim)
        X = SDUpdate(X, pop, SDNumber, fitness, BestF)
        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = CaculateFitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        if (fitness[0] <= GbestScore):
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore

    return GbestScore, GbestPositon, Curve