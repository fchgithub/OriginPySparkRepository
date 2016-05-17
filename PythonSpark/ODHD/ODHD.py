'''
Created on Nov 9, 2015

@author: Fatemeh
'''

from pyspark import SparkContext, SparkConf
import numpy as np
from scipy.stats import chisquare
import ParallelGA as GA
import profile
import pstats

class ODHD_Ensemble(object):
    '''
        ODHD_Ensemble is a class to find the subspaces in a high-dimensional by benefit of GA on some sample of main data (forming an ensemble)
        The Goal of the class: list of subspaces with the most chance of finding a  
    '''
    subspace_lst = []
    def __init__(self):
        '''
        Constructor
        '''
        # static member of the class; accessible out side of class via ODHD_Ensemble.subSpaceList
        ODHD_Ensemble.subSpaceList = []  
             
    def _X_distribution(self, attr):
        ''' calculate X-distribution for the selected attribute
            check R packages that how to use existed method in R (avoiding to implement by myself) 
        ''' 
        chisquare(attr)
        return 10
    
    def attr_uniform_disribution(self, attr):
        '''
        check the uniformity of attribute
        '''
        alfa = 0
        if self._X_distribution(attr) < alfa:
            return True
        
        return False
    
    def special_sampling_each_partion(self, rdd, number_of_partition, number_of_samples, percentage):
   
        '''
        There should be in each partition (does it make a difference?)
           To_Do:  
            - change the number of partitions and see how it will effect/influence
            - my idea was to find out the attributes in data with lower uniformity distribution. 
            - create a data with 3 attributes
            - scatter them based on just two of them into 2-3 groups  (20 samples)
            - keep third attribute as uniform as possible
            - do the calculations to assign numbers to these attributes based on their uniformity
            - take samples based on that
            - check out if the samples come out from dense parts or not
            - there should be a compare with spark random sampling
            - qq
        '''
        
        return None
    
    def special_sampling_whole_data(self, rdd, percentage):
        '''
        This method is different from special_sampling_each_partion method in terms of sampling whole data
        not just in each partition.
        '''
        return None
    
    def spark_self_Sampling(self, rdd, percentage):
    
        '''rdd.takeSample(False, number_of_samples, seed = 10)
            Return a fixed-size sampled subset of this RDD.
            it's a list of sample data; not another RDD (not applicable since we need an rdd)
            Note: rdd.sample returns PipelinedRDD which is a subclass of RDD and inherits all APIs which RDD has
        '''
        return rdd.sample(False, percentage, seed = 10)
    
    
    
    def genetic_algorithm(self, sampleData):
    
        return
    # End of Class ODHD    
    
def data_reduction():
    
    return None
    
def cleaning_data(data):
    ''' It's a pre-processing step to clean data: Dealing with missing data, finding correlations, reduce the size of data
    '''
    return None

def toVector(line):
        return  np.array([float(x) for x in line.split(',')])
        
    
def main():
    '''
        All setup with Git
        Instruction:
        - Changes from laptop captioned: "Laptop-Date" like Laptop-April-14-2016
        - Changes from PC captioned: "PC-Date" like PC-April-14-2016
    '''
    spark_conf = SparkConf().setAppName("Different-Sampling data").setMaster("local[*]")
    sc = SparkContext(conf= spark_conf)
    #rdd = sc.textFile('/user/kddhadoop/inputs/pre-data.txt')#, minPartitions = 7)
    GA.logInConsole(0, "input file read!")
    rdd = sc.textFile('/fatemeh/inputs/FMA-1.csv', minPartitions=10)
    vectorRDD = rdd.map(toVector)
    print(vectorRDD.count())
    GA.logInConsole(0 , "Data Vectorized!")
    #print(vectorRDD.first())
    #rdd.unpersist()
    #myODHD = ODHD_Ensemble()
    #percetage_of_sample = 0.3
    #sizeOfDataset = vectorRDD.count()  
    #sampleRDD = myODHD.spark_self_Sampling(vectorRDD, percetage_of_sample)
    #data_dimension = len(vectorRDD.first())
    #print('data dimension: ', data_dimension)
    #maxall = vectorRDD.reduce(GA.maxFunc)
    #GA.Parallel_GA_main(vectorRDD,sc)#, sizeOfDataset, data_dimension)
    print('this is a test from laptop May 17')
    sc.stop()
    
if __name__ == "__main__":
    main()
    #profile.run("main()")
    
