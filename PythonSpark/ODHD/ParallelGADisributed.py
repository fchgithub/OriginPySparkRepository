'''
Created on Jan 21, 2016

@author: Fatemeh 
'''
from __future__ import  print_function, division
import random
import numpy as np
import functools
from operator import add
import math
import itertools
import time
import os
import time
import datetime
from pyspark.mllib.stat import Statistics
from pyspark import SparkContext, SparkConf
from _ast import Subscript


def parallel_GA_main(sc, rdd, divisionRange):
    populationSize = 50
    generationsNumber = 50
     
    ellitePercentage = 0.2
    crossover_rate = 0.7
    mutation_rate = 0.3
    new_individual_feed_rate = 0.2
    
    dimensionSize = 30  
    datasetSize = 367
    
    projectionSizes = [3]
    rng = divisionRange
    samples_in_cell_cutoff = 1
    
    
               
    hash_dict = dict()
    topKElegant = list()
    solution = list()
    
    for prjSize in projectionSizes:
        seenBefore = dict()
        population = generatePopulation(dimensionSize, prjSize, seenBefore, hash_dict, populationSize)
        #population = tempgeneratePopulation(dimensionSize, prjSize)
        itr = 0
        while itr < generationsNumber: 
            print('iteration is : ', itr)
            rankedPopulation = population_ranking(population[:populationSize], datasetSize, rng, hash_dict, rdd ,sc)
            solution.append(rankedPopulation)
            rankedPopulation_unhashed = unhash_population(rankedPopulation, hash_dict)
            #write_to_S3(rankedPopulation_unhashed, itr, sc, "/home/fatemeh/Data/Datasets/output/ODHD")
            write2File(itr, rankedPopulation_unhashed, "/home/fatemeh/Data/Datasets/output/ODHD")
            population = iterate_population(rankedPopulation, populationSize, ellitePercentage, new_individual_feed_rate, crossover_rate,
                                            mutation_rate, hash_dict, seenBefore, dimensionSize, prjSize)
            itr += 1
     
    return None
 
def unhash_population(population, hash_dict):
    return [(hash_dict[ind[0]],ind[1]) for ind in population]

def population_ranking(population, datasetSize, prjrng, hash_dict, rdd, sc):
    """
        (list)->(list(hash value of subspace, fitness value))
    """
    #rankedPopulation = fitnessFunc_integrated(rdd, population, all_attr_maxs, all_attr_mins, datasetSize, prjrng, sc)
    #rankedPopulation = fintessFunc_perInd(rdd, population, all_attr_maxs, all_attr_mins, datasetSize, prjrng)
    rankedPopulation = fitnessFuncPerPopulation(rdd, population, datasetSize, prjrng, hash_dict)
    rankedPopulation.sort(key=lambda tup: tup[1], reverse=True)  
    return rankedPopulation

def fitnessFuncPerPopulation(rdd, population, datasetSize, projectionRange, hash_dict):
    num_of_cells = projectionRange ** len(population[0])
    # point => [((subspace_1, cell), 1), ((subspace_2, cell), 1), ..., ((subspace_50, cell), 1)]=> [((subspace_1, cell_1), sum1), ((subspace_1, cell_2), sum), ... ] => [(subspace_1, [sum,sum,...]), ...]
    points_in_subspace_RDD = rdd.flatMap(lambda point: (assign2Cell(point, population, projectionRange)))
    points_in_subspace_RDD = points_in_subspace_RDD.reduceByKey(lambda a, b: a+b).map(lambda x: (x[0][0], x[1])).reduceByKey(merge_vectors)
    subspace_stat_RDD = points_in_subspace_RDD.map(lambda (subspace, list_): (subspace, chi_test(list_, num_of_cells)))
    return subspace_stat_RDD.collect()
    
def fintessFunc_perInd(rdd, population, all_attr_maxs, all_attr_mins, datasetSize, prjrng):
        rankedPopulation = list()
        for individual in population:
            fitness = fitnessFunc(rdd, individual, all_attr_maxs, all_attr_mins, datasetSize, prjrng)
            rankedPopulation.append((individual, fitness))
        return rankedPopulation
    

def iterate_population(rankedPopulation, populationSize, ellitePercentage, new_individual_feed_rate, crossover_rate, mutation_rate, 
                       hash_dict, seenBefore, dimensionSize, projectionSize):
    fitnessScores = [item[1] for item in rankedPopulation]
    rankedIndividuals = [item[0] for item in rankedPopulation]
    newpop = []
    rand_invididual_number = int(math.ceil(populationSize * new_individual_feed_rate))
    ellitNum = int(math.ceil(populationSize * ellitePercentage))
    ellite = rankedPopulation[:ellitNum]
    for element in ellite:
        seenBefore[element[0]] = False
        newpop.append(list(hash_dict[element[0]]))  
    if len(rankedPopulation) > 1:
        while len(newpop) < populationSize-rand_invididual_number:
            child1, child2 = [], []
            parent1, parent2 = selectFittest(fitnessScores, rankedIndividuals)     
            child1, child2 = breed(parent1, parent2, crossover_rate, mutation_rate, hash_dict, dimensionSize)  
            hashableChild1 = hash(tuple(child1)) 
            hash_dict[hashableChild1] = tuple(child1)
            hashableChild2 = hash(tuple(child2)) 
            hash_dict[hashableChild2] = tuple(child2)
            ch1BeenSeen = hashableChild1 in seenBefore
            ch2BeenSeen = hashableChild2 in seenBefore
            if (not ch1BeenSeen or (ch1BeenSeen and not seenBefore[hashableChild1])) and child1 not in newpop:
                if not ch1BeenSeen:
                    seenBefore[hashableChild1] = True
                newpop.append(child1) 
            if (not ch2BeenSeen or (ch2BeenSeen and not seenBefore[hashableChild2] )) and len(newpop) < populationSize and child2 not in newpop:
                if not ch2BeenSeen:
                    seenBefore[hashableChild2] = True
                newpop.append(child2)
        newpop = newpop + generatePopulation(dimensionSize, projectionSize, seenBefore, hash_dict, rand_invididual_number)
    else:
        #newpop = generatePopulation(dimensionSize, prjSizes[random.randint(0, len(prjSizes)-1)], seenBefore, hash_dict, populationSize)
        newpop = generatePopulation(dimensionSize, projectionSize, seenBefore, hash_dict, populationSize)
    return newpop
  
def breed (individual1, individual2, crossover_rate, mutation_rate, hash_dict, dimensionSize):
    newCh1, newCh2 = [], []
    individual1 = list(hash_dict[individual1])
    individual2 = list(hash_dict[individual2])
    if random.random() < crossover_rate:  # rate dependent crossover of selected chromosomes
        newCh1, newCh2 = crossover(individual1, individual2)
        
    else:
        newCh1, newCh2 = individual1, individual2
    newnewCh1 = mutation(newCh1, mutation_rate, dimensionSize)  # mutate crossovered chromos
    newnewCh2 = mutation(newCh2, mutation_rate, dimensionSize)
    newnewCh1.sort()
    newnewCh2.sort()
  
    return newnewCh1, newnewCh2

def fitnessFunc(rdd, individual, all_attr_maxs, all_attr_mins, datasetSize, prjRng):
    maxs = [all_attr_maxs[x] for x in individual]
    mins = [all_attr_mins[x] for x in individual]
    
    num_of_cells = prjRng ** len(individual)
   
   
    map2CellRDD = rdd.map(lambda point: (assign2Cell(point[individual], individual, \
                                                     mins, maxs, prjRng), 1))
    sumPointsInCellRDD = map2CellRDD.reduceByKey(lambda a, b: a + b)
    cellsWithPoint = sumPointsInCellRDD.count()
    
#    aver = float(datasetSize) / num_of_cells
    aver = float(datasetSize) / cellsWithPoint 
    emptyCells = num_of_cells - cellsWithPoint
    zigma = 150
    '''
        Fitness values are sum of cells less than average
    '''
#     percetageLA = (sumPointsInCellRDD.map(lambda x: 1 if x[1]> aver else 0).reduce(add)/datasetSize)*100
#     percetageBA = 100 - percetageLA
#     fitnessValue = sumPointsInCellRDD.map(lambda x: math.e**(-((15*x[1])^2)/(2*(zigma^2))) if x[1]<=aver else 0). \
#                                          reduce(add) + emptyCells 
    fitnessValue = emptyCells + sumPointsInCellRDD.map(lambda x: ((aver - x[1]) / aver) ** 3 if x[1] <= aver else 0).reduce(add) 
    return fitnessValue

def fitnessFunc_integrated(rdd, population, all_attr_maxs, all_attr_mins, datasetSize, prjRng, sc):
    
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



def chi_test(list_, num_of_cells):
    grid_cells = list_ + [0]*(num_of_cells - len(list_))
    avrg = sum(list_)/len(grid_cells)
    vect = [(x-avrg)**2 for x in grid_cells]
    return round((sum(vect)/avrg), 2)
    
def write_to_S3(information, itr, sc, dir="s3://kddlabs3bucket/output/odhd/"):
    #st = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d%H%M%S')
    path_ = dir + 'run'# + st
    path_ +=  str(itr)
    sc.parallelize(information).saveAsTextFile(path_)
    return

def write_to_console(rankedPopulation, itr, hash_dict):
    print("Iteration "+str(itr)+": ")
    for ind in rankedPopulation:
        print(hash_dict[ind[0]])
        print(" : "+str(ind[1]))

def write2File(itration, subspaceList, path):
    currentRun = path + "/" + str(itration)+ ".txt"
    file = open(currentRun, 'w+')
    file.write("Iteration %s:\n" % itration)
    for subspace in subspaceList:
        file.write("%s\n" % str(subspace))
    return

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

def normalize_1(input):
    input_array = np.array(input)
    mn = np.min(input_array)
    mx = np.max(input_array)
    return (input_array-mn)/(mx-mn)

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

def mutation(individual, mutation_rate, dimensionSize):
    for field in individual:
        if random.random() < mutation_rate:
            while True:
                r1 = random.randint(0, dimensionSize - 1)
                r2 = random.randint(0, len(individual) - 1)
                if(r1 not in individual):
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
        ind = 2;
        selected = list()
        for i in range(ind):
            i = i + 1 - 1  # to get ride of warning message of not used variable
            max_index, max_value = max(enumerate(fitnessScores), key=operator.itemgetter(1))
            fitnessScores.remove(max_value)
            selected.append(max_index)
        return selected, ind
    
def generatePopulation(totalDimension, projectionSize, seenBefore, hash_dict, number_of_inds):
    population, individual = [], []
    rangeDimension = range(totalDimension)#(0,1,2,3,...,d) 
    while len(population) < number_of_inds:       
        individual = random.sample(rangeDimension, projectionSize)
        individual.sort()
        hashableInd = hash(tuple(individual))
        hash_dict[hashableInd] = tuple(individual)
        indBeenSeen = hashableInd in seenBefore
        if not indBeenSeen or (indBeenSeen and not seenBefore[hashableInd]) :
            seenBefore[hashableInd] = True
            population.append(individual)
    return population

def tempgeneratePopulation(dimension, projectionSize):
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

def merge_vectors(a, b):
    ret = list()
    
    if(type(a).__name__  == 'int'):
        ret.append(a)
    else:
        ret = ret + a
    if(type(b).__name__  == 'int'):
        ret.append(b)
    else:
        ret = ret + b
        
    return ret
    
def minFunc(a, b):
    mins = [None] * len(a)
    for i in range(len(a)):
        mins[i] = a[i] if a[i] <= b[i] else b[i]
    return mins

def assign2Cell(point, population, divisionsNum):
    spotted_cells_in_subSpaces = []
    for individual in population:
        cell_locations = []
        divisions = np.linspace(0, 1, divisionsNum + 1)
        for ind in range(len(individual)):
            try:
                point_in_single_dim = point[individual[ind]]
            
            except IndexError:
                print('length point: ', len(point), 'individual: ', individual, 'index: ', ind)
            for i in range(1, len(divisions)): 
                if point_in_single_dim <= divisions[i]:
                    cell_locations.append(i - 1)
                    break
    
    # (A,B,C) , (i,j,ind) -> i + (j*A) + (ind*A*B)
    
        cell = cell_locations[0]
        for k in range(1, len(cell_locations)):
            l = cell_locations[k]
            for j in range(0, k):
                l = l * divisionsNum
            cell = cell + l
        hashed_individual = hash(tuple(individual))
        spotted_cells_in_subSpaces.append(((hashed_individual, cell), 1))
        #hash_dict[hashed_individual] = tuple(individual) 
    return spotted_cells_in_subSpaces

def getBinary(number, ind):
    lst = list()
    # lst = [[None]*ind for i in range(number)]
    for j in range(number):
        num = j
        for i in reversed(range(0, ind)):
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

def findBestChildInR(rdd, bestSparceChild, child, all_attr_maxs, all_attr_mins, datasetSize, dictionary):
    if bestSparceChild == None:
        bestSparceChild = child
        if(code(child) not in dictionary.keys()):
            bestFitnessValue = fitnessFunc(rdd, bestSparceChild, all_attr_maxs, all_attr_mins, datasetSize)
            dictionary[code(child)] = bestFitnessValue
        else: 
            bestFitnessValue = decode(child, dictionary)
    else:
        if(code(child) not in dictionary.keys()):
            s = fitnessFunc(rdd, child, all_attr_maxs, all_attr_mins, datasetSize)
            dictionary[code(child)] = s
        else:
            s = decode(child, dictionary)
        
        if s > bestFitnessValue:
            bestSparceChild = child
            bestFitnessValue = s
            
    return bestSparceChild

def findBestGreedyChildInQ(rdd, bestSparceChild, p1, p2, ind, Q, all_attr_maxs, all_attr_mins, datasetSize, dictionary, dimension):
    while (ind) != 0:
        tmpBestChild = bestSparceChild if bestSparceChild != None else [0] * dimension
        bestFitnessValue = float('-inf')
        for i in range(len(Q)):
            tmpBestChild[Q[i]] = p1[Q[i]] if p1[Q[i]] != 0 else p2[Q[i]]
            if (code(tmpBestChild) not in dictionary.keys()):
                s = fitnessFunc(rdd, tmpBestChild, all_attr_maxs, all_attr_mins, datasetSize)
                dictionary[code(tmpBestChild)] = s
            else:
                s = decode(tmpBestChild, dictionary)
            if s > bestFitnessValue:
                bestFitnessValue = s
                pos = Q[i]
                bestSparceChild = tmpBestChild[:]
            tmpBestChild[Q[i]] = 0  # return to not selected to test others for minimum
        ind -= 1
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
    while i < len(a) and j < len(b) and len(c) < ind:
        if a[i][1] > b[j][1]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    if i < len(a):
        while i < len(a)and len(c) < ind:
            c.append(a[i])
            i += 1
    if j < len(b):
        while j < len(b)and len(c) < ind:
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


def toVector(line, splitter):
    tokenized = line.split(splitter)
    return  np.array([float(tokenized[x]) for x in range(0,len(tokenized)-2)])      

def load_data(sc):
    #rdd = sc.textFile("s3://kddlabs3bucket/Data/wdbcundersampledmnormalized.csv",  minPartitions= 500, use_unicode=False)
    rdd = sc.textFile("/home/fatemeh/Data/Datasets/wdbc_UndersampledM_Normalized.csv", use_unicode=False)
    vectorRDD = rdd.map(lambda line: toVector(line, splitter = ' '))
    return vectorRDD

def main():
    spark_conf = SparkConf().setAppName("Different-Sampling data")
    spark_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sc = SparkContext(conf= spark_conf)
    rdd = load_data(sc)  
    print(rdd.getNumPartitions())
    parallel_GA_main(sc, rdd, 5)
    
    sc.stop()
    
if __name__ == "__main__":
    main()