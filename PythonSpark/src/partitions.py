'''
Created on Oct 29, 2015

@author: Fatemeh
'''

from __future__ import print_function
from operator import add
from pyspark import SparkContext, SparkConf

def f(iterator): 
    for x in iterator:
        print (x, end= " ")
    print ()
    yield None
    
def x(z):
    print (z, end = " ")
    
if __name__ == "__main__":
    
    sparkConf = SparkConf().setAppName("All-About-Partition").setMaster("local[3]")
    sc = SparkContext(conf = sparkConf)
    
    # Applies on all elements of RDD  
    sc.parallelize(range(1, 9), 2).foreach(x) 
    
    # Applies on each partition of RDD separately
    sc.parallelize(range(1, 9), 2).foreachPartition(f)
    
    # Using "add" operator
    print(sc.parallelize([1, 5, 6, 7, 8, 9, 3]).reduce(add))
    
    