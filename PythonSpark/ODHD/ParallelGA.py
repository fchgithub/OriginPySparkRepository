'''
Created on Jan 21, 2016

@author: Fatemeh 
'''
from __future__ import  print_function
import random
import numpy as np
import functools
from operator import add
import math
import itertools
import time
import CustomKey
from cmath import polar

# Global Variables
percentage_of_ellite = 0.5
popNum = 10 # choose it always an even number
crossover_rate = 0.7
mutation_rate = 0.2
dimLength = 0  #takes the number of dimension of high dimension data set 
numofGeneration = 10
k = 20


def Parallel_GA_main(rdd, sc):
    logInConsole(1, "main method started!")
    rngdivision = [5]
    prjSizes = [3]
    logInConsole(2, "REDUCE IN PROGRESS: Getting Min and Max of all data dimensions: START!")
    # pre-proccessing step to find min and max of attributes just once not everytime in running code 
#     all_attr_maxs = rdd.reduce(maxFunc)
#     save_to_file(all_attr_maxs, 'max')
#     logInConsole(2, "REDUCE IN PROGRESS: Getting done Max of all data dimensions: Done!")
#     all_attr_mins = rdd.reduce(minFunc)
#     save_to_file(all_attr_mins, 'min')

    all_attr_maxs = np.loadtxt("max.out", delimiter = ',')
    all_attr_mins = np.loadtxt('min.out', delimiter = ',')
    logInConsole(2, "REDUCE IN PROGRESS: Getting Min and Max of all data dimensions: FINISH!")
    # to change the value of a global variable
    global dimLength
    dimLength = len(all_attr_maxs)
    sizeOfDataset = 500 #rdd.count()
    rdd.cache()
    logInConsole(3, "Computing fitness function for all genes: START!")
    topKElegant = []
    solutions = []
    for psize in prjSizes:
        for rng in rngdivision:
            population = generatePop(dimLength, psize)
#             population = tempGeneratePop(dimLength, psize)
#             population = [[0,2,5], [0,2,3],[1,2,5]]   
            itr = 0
    # while not stopCondition(1, itr):
            while itr < numofGeneration:
                print ('\nCurrent iterations:', itr)
                rankedPopulation = rankedPop(population[:popNum], rdd, all_attr_maxs, all_attr_mins, sizeOfDataset, rng,sc)
                topKElegant = remove_duplicate(merge(rankedPopulation, topKElegant))
                if(len(topKElegant) > k):
                    topKElegant = topKElegant[:k]
                print('\nFinal list of elegant after ', itr, ' is: ', topKElegant)  
                tempPrintrankedPopulation(rankedPopulation)
                logInConsole(4, "Done: tempPrintrankedPopulation")
                population = iteratePop(rankedPopulation)
                logInConsole(5, 'Done : crossover and mutation')
                
                itr += 1
            
            solutions.append(topKElegant)  
            topKElegant = []
    #rankedPopulation = rankedPop(population, rdd, all_attr_maxs, all_attr_mins, sizeOfDataset, rngdivision[0])           

    print('\nSet of Solutions: ', solutions)
    
    
def rankedPop(population, rdd, all_attr_maxs, all_attr_mins, sizeOfDataset, prjrng, sc):
    #rankedPopulation = fitnessFunc_integrated(rdd, population, all_attr_maxs, all_attr_mins, sizeOfDataset, prjrng, sc)
    rankedPopulation = fintessFunc_perInd(rdd, population, all_attr_maxs, all_attr_mins, sizeOfDataset, prjrng)
    rankedPopulation.sort(key=lambda tup: tup[1], reverse=True)  
    return rankedPopulation

def fintessFunc_perInd(rdd, population, all_attr_maxs, all_attr_mins, sizeOfDataset, prjrng):
        rankedPopulation = list()
        for individual in population:
            fitness = fitnessFunc(rdd, individual, all_attr_maxs, all_attr_mins, sizeOfDataset, prjrng)
            individual.sort()
            rankedPopulation.append((individual, fitness))
        return rankedPopulation
    
def iteratePop(rankedPopulation):
    fitnessScores = [item[1] for item in rankedPopulation]
    rankedIndividuals = [item[0] for item in rankedPopulation]
    newpop = []
    ellitNum = math.ceil(popNum * percentage_of_ellite) 
    ellite = rankedPopulation[:ellitNum]
    for element in ellite:
        newpop.append(element[0])  # known as elitism, conserve the best solutions to new population
    while len(newpop) <= popNum:
        ch1, ch2 = [], []
        ch1, ch2 = selectFittest (fitnessScores, rankedIndividuals)  # select two of the fittest chromos
        
        ch1, ch2 = breed(ch1, ch2)  # breed them to create two new chromosomes 
        newpop.append(ch1)  # and append to new population
        newpop.append(ch2)
    return newpop
  
def breed (individual1, individual2):
    newCh1, newCh2 = [], []
    if random.random() < crossover_rate:  # rate dependent crossover of selected chromosomes
        newCh1, newCh2 = crossover(individual1, individual2)
    else:
        newCh1, newCh2 = individual1, individual2
    newnewCh1 = mutation(newCh1)  # mutate crossovered chromos
    newnewCh2 = mutation(newCh2)
  
    return newnewCh1, newnewCh2

def fitnessFunc(rdd, individual, all_attr_maxs, all_attr_mins, sizeOfDataset, prjRng):
    maxs = [all_attr_maxs[x] for x in individual]
    mins = [all_attr_mins[x] for x in individual]
    
    num_of_cells = prjRng ** len(individual)
   
    map2CellRDD = rdd.map(lambda point: (assign2Cell(point, individual, \
                                                     maxs, mins, prjRng), 1))
    sumPointsInCellRDD = map2CellRDD.reduceByKey(lambda a, b: a + b)
    cellsWithPoint = sumPointsInCellRDD.count()
    
    aver = float(sizeOfDataset) / num_of_cells 
    emptyCells = num_of_cells - cellsWithPoint
    zigma = 150
    '''
        Fitness values are sum of cells less than average
    '''
#     percetageLA = (sumPointsInCellRDD.map(lambda x: 1 if x[1]> aver else 0).reduce(add)/sizeOfDataset)*100
#     percetageBA = 100 - percetageLA
#     fitnessValue = sumPointsInCellRDD.map(lambda x: math.e**(-((15*x[1])^2)/(2*(zigma^2))) if x[1]<=aver else 0). \
#                                          reduce(add) + emptyCells 
    fitnessValue = emptyCells + sumPointsInCellRDD.map(lambda x: ((aver - x[1]) / aver) ** 3 if x[1] <= aver else 0).reduce(add) 
    return fitnessValue
def fitnessFunc_integrated(rdd, population, all_attr_maxs, all_attr_mins, sizeOfDataset, prjRng, sc):
    
    unionRdd = sc.emptyRDD()
    num_of_cells = prjRng ** len(population[0])
    for individual in population:
        maxs = [all_attr_maxs[x] for x in individual]
        mins = [all_attr_mins[x] for x in individual]
        curRdd = rdd.map(lambda point: (CustomKey.CustomKey(individual, assign2Cell(point, individual, \
                                                         maxs, mins, prjRng)), 1))    
        unionRdd = unionRdd.union(curRdd)
    sumPointsInCellRDD = unionRdd.reduceByKey(lambda a, b: a + b)
    fitnessValues = [0] * len(population)
    for ind in range(len(population)):
        fitnessValues[ind] = (population[ind], num_of_cells - \
                              sumPointsInCellRDD.filter(lambda x : (x[0].individual == population[ind])).count()) 
    return fitnessValues

def selectFittest (fitnessScores, rankedIndividuals):
    while 1 == 1: 
        index1 = roulette_Selection(fitnessScores)
        index2 = roulette_Selection(fitnessScores)
        if index1 == index2:
            continue
        else:
            break

    ch1 = rankedIndividuals[index1]  # select  and return chromosomes for breeding 
    ch2 = rankedIndividuals[index2]
    return ch1, ch2
        
def crossover(individual1, individual2):
    '''
        common dimensions left unchanged, the remaining are selected randomly
        To Do: take some greedy approaches in selecting new dimensions 
    '''
    cmmFields = list(set(individual1) & set(individual2))
    notCmmFields = list((set(individual1) - set(individual2)) | (set(individual2) - set(individual1))) 
    notCmmFields.sort()
    newChild1 = cmmFields[:]
    newChild2 = cmmFields[:]
    prjdim = len(individual1)
    while len(newChild1) != prjdim:
        newChild1.append(notCmmFields.pop(random.randint(0, len(notCmmFields) - 1)))
        newChild2.append(notCmmFields.pop(random.randint(0, len(notCmmFields) - 1))) 
        
    return newChild1, newChild2

def mutation(individual):
    for field in individual:
        if random.random() < mutation_rate:
            while True:
                r1 = random.randint(0, dimLength - 1)
                r2 = random.randint(0, len(individual) - 1)
                if(r1 not in individual):
#                     print('\nr1: ',r1)
#                     print('\nr2: ',r2)
#                     print('\n individual size: ',len(individual))
                    individual[r2] = r1
                    break
                else:
                    continue
    return individual


def roulette_Selection (fitnessScores):
    # Source: http://www.obitko.com/tutorials/genetic-algorithms/selection.php
    cumulativeFitness = 0.0
    for i in range(len(fitnessScores)): 
        cumulativeFitness += fitnessScores[i]
    r = random.uniform(0, cumulativeFitness)
   
    cumulativeFitness = 0
    for i in range(len(fitnessScores)):  # for each chromosome's fitness score
        cumulativeFitness += fitnessScores[i]  # add each chromosome's fitness score to cumulative fitness
        if cumulativeFitness > r:  # in the event of cumulative fitness becoming greater than r, return index of that chromo
            return i

def rank_Selection(fitnessScores):
    while True:
        first = roulette_Selection(fitnessScores)
        second = roulette_Selection(fitnessScores)
        if first == second:
            continue
        else: 
            break
    return first, second
    
def steady_State_Selection(fitnessScores):
        k = 2;
        selected = list()
        for i in range(k):
            i = i + 1 - 1  # to get ride of warning message of not used variable
            max_index, max_value = max(enumerate(fitnessScores), key=operator.itemgetter(1))
            fitnessScores.remove(max_value)
            selected.append(max_index)
        return selected, k
    
def generatePop(dimension, projectionSize):
    ''' generates initial population with number of popNum(global variable)
        with the dimension projectionSize. 
        randomly picks the positions and sort them then add to a list and return it
    '''
    population, individual = [], []
    rangeDimension = range(dimension)
    while len(population) != popNum:
        individual = random.sample(rangeDimension, projectionSize)
        individual.sort()
        population.append(individual)
        population = [list(t) for t in set(map(tuple, population))]
    return population

def tempGeneratePop(dimension, projectionSize):
    ''' generatePop(dimension, projectionSize) -> list of population
    '''
    lst = [1] * dimension
    for i in range(dimension):
        lst[i] = i
    
    return [list(x) for x in set(itertools.combinations(lst, projectionSize))]

def tempPrintrankedPopulation(rankedPopulation):
    print("\n")
    for ind in rankedPopulation:
        print(ind[0], ' - ', ind[1])

def stopCondition(stop_type, itr, convergens=0.5):
    alpha = 0.5
    end_itr = 1
    if stop_type == 1:
        return False if itr < end_itr else True
    else: 
        return False if convergens < alpha else True
    
def maxFunc(a, b):
    if(len(a)!=len(b)):
        print('len a: ', len(a), ' -- len b:  ', len(b))
    lenA = len(a)
    maxs = [None] * lenA
    for i in range(lenA):
        maxs[i] = a[i] if a[i] >= b[i] else b[i]
    return maxs
    
def minFunc(a, b):
    mins = [None] * len(a)
    for i in range(len(a)):
        mins[i] = a[i] if a[i] <= b[i] else b[i]
    return mins

def assign2Cell(point, projectionDims, mins, maxs, divisionsNum):
    cellLoc = []
    ind = 0
    projectionSize = len(projectionDims)
    for dim in projectionDims:
        divRange = np.linspace(0, 1, divisionsNum + 1)
        try:
            ''' Normalizing data between 0 and 1'''
            nrmPoint = (point[dim] - mins[ind]) / (maxs[ind] - mins[ind])
            for i in range(1, len(divRange)):  # to avoid -1 when point[i]==0
                if nrmPoint <= divRange[i]:
                    cellLoc.append(i - 1)
                    break
        except:
            print('\npoint: ', point)
            print('\nIndividual: ', projectionDims)
            cellLoc.append(mins[ind])
        ind = ind + 1   
    # (A,B,C) , (i,j,k) -> i + (j*A) + (k*A*B)

    sum_ = cellLoc[0]
    for i in range(1, projectionSize):
        l = cellLoc[i]
        for j in range(0, i):
            l = l * divisionsNum
        sum_ = sum_ + l 
    return sum_

def getBinary(number, k):
    lst = list()
    # lst = [[None]*k for i in range(number)]
    for j in range(number):
        num = j
        for i in reversed(range(0, k)):
            l = list()
            r = num % 2
            l.append(r)
            # lst[j][i] = r
            num = num // 2
        lst.append(l)
    return lst    
    
def permutation(binaryList, p1, p2, R):
    lst = [[None] * len(R) for i in range(len(binaryList))]
    for i in range(len(binaryList)):
        for j in range(len(R)):
            if binaryList[i][j] == 0:
                lst[i][j] = p1[R[j]]
            else:
                lst[i][j] = p2[R[j]]
            
    return lst

def code(individual):
    code = 0
    base = max(individual) + 1
    for i in range(len(individual)):
        code = int(code + individual[i] * math.pow(base, i))
    return code

def decode(individual, dictionary):
    if(len(individual) == 0):
        print("okh okh")
    return dictionary[code(individual)]

def getUniqueChilderen(p1, p2, R, lenR):
    binaryList = getBinary((2 ** lenR) * (0 if lenR == 0 else 1), lenR)
    tmpChilderen = permutation(binaryList, p1, p2, R)
    # To remove doubles
    binarySet = set(tuple(x) for x in tmpChilderen)
    tmpChilderen = [list(x) for x in binarySet]
    return tmpChilderen

def findBestChildInR(rdd, bestSparceChild, child, all_attr_maxs, all_attr_mins, sizeOfDataset, dictionary):
    if bestSparceChild == None:
        bestSparceChild = child
        if(code(child) not in dictionary.keys()):
            bestFitnessValue = fitnessFunc(rdd, bestSparceChild, all_attr_maxs, all_attr_mins, sizeOfDataset)
            dictionary[code(child)] = bestFitnessValue
        else: 
            bestFitnessValue = decode(child, dictionary)
    else:
        if(code(child) not in dictionary.keys()):
            s = fitnessFunc(rdd, child, all_attr_maxs, all_attr_mins, sizeOfDataset)
            dictionary[code(child)] = s
        else:
            s = decode(child, dictionary)
        
        if s > bestFitnessValue:
            bestSparceChild = child
            bestFitnessValue = s
            
    return bestSparceChild

def findBestGreedyChildInQ(rdd, bestSparceChild, p1, p2, k, Q, all_attr_maxs, all_attr_mins, sizeOfDataset, dictionary, dimension):
    while (k) != 0:
        tmpBestChild = bestSparceChild if bestSparceChild != None else [0] * dimension
        bestFitnessValue = float('-inf')
        for i in range(len(Q)):
            tmpBestChild[Q[i]] = p1[Q[i]] if p1[Q[i]] != 0 else p2[Q[i]]
            if (code(tmpBestChild) not in dictionary.keys()):
                s = fitnessFunc(rdd, tmpBestChild, all_attr_maxs, all_attr_mins, sizeOfDataset)
                dictionary[code(tmpBestChild)] = s
            else:
                s = decode(tmpBestChild, dictionary)
            if s > bestFitnessValue:
                bestFitnessValue = s
                pos = Q[i]
                bestSparceChild = tmpBestChild[:]
            tmpBestChild[Q[i]] = 0  # return to not selected to test others for minimum
        k -= 1
        Q.remove(pos)     
    return bestSparceChild

def createSibling(child, p1, p2):
    child2 = [None] * len(child)
    for i in range(len(child)):
        child2[i] = p2[i] if child[i] == p1[i] else p1[i]
    return child2

def save_to_file(all_attr_max, name):
    x = np.savetxt(name+'.out', all_attr_max, delimiter = ',') 

def logInConsole(step , message):
    now = time.strftime('%H:%M:%S')
    print("\n","LOG STEP ", step, " time: ", now, " : ", message)
    
def merge(a, b):
    """
    a and b are two sorted lists
    """
    i = 0
    j = 0
    c =  []
    while i < len(a) and j < len(b) and len(c) < k:
        if a[i][1] > b[j][1]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    if i < len(a):
        while i < len(a)and len(c) < k:
            c.append(a[i])
            i += 1
    if j < len(b):
        while j < len(b)and len(c) < k:
            c.append(b[j])
            j += 1
    return c

def remove_duplicate(rankedList):
    output = []
    seen = []
    for value in rankedList:
        if value[0] not in seen:
            output.append(value)
            seen.append(value[0])
    return output       