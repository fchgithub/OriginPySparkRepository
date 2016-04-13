'''
Created on Jan 5, 2016

@author: Fatemeh
'''
from __future__ import print_function
from pyspark import SparkContext, SparkConf
import numpy as np



def main(sc):
    #rdd = sc.parallelize([[2,4],[10,5], [-4,6], [3,0], [7,100], [-13,45], [798,25], [12,3], [2,-100], [46,9], [6, 500]], 6)
    
    rdd = sc.textFile('D:\dataset - Copy.txt', minPartitions = 10)#print(rdd.map(toVector))
    rdd2 = rdd.map(toVector)
    print(rdd2.reduce(minMax))

def toVector(line):
    return  np.array([float(x) for x in line.split('\t')])

def minMax(a, b):
    maxs = list()
    mins = list()
    for i in range(len(a)):
        if a[i] >= b[i]:
            maxs.append(a[i])
            mins.append(b[i])
        else: 
            maxs.append(b[i])
            mins.append(a[i])
    return maxs, mins

if __name__ == "__main__":
    sparkconf = SparkConf().setAppName("K-MEANS").setMaster("local[*]")
    sc = SparkContext(conf = sparkconf)
    minPartition = 2;
    main(sc)
    
    sc.stop()


