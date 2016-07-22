'''
Created on Nov 9, 2015

@author: Fatemeh
'''
from __future__ import  print_function
from pyspark import SparkContext, SparkConf
import numpy as np
# from scipy.stats import chisquare
import ParallelGA as GA
import profile
from pyspark.mllib.feature import Normalizer
import SSD
class ODHD_Ensemble(object):
    '''
        ODHD_Ensemble is a class to find the subspaces in a high-dimensional by benefit of GA on some sample of main data (forming an ensemble)
        The Goal of the class: list of subspaces with the most chance of finding a  
    '''
    subspace_lst = []
    def __init__(self):
        '''
        Constructor of the class
        '''
        # static member of the class; accessible out side of class via ODHD_Ensemble.subSpaceList
        ODHD_Ensemble.subSpaceList = []  
             
    def _X_distribution(self, attr):
        ''' calculate X-distribution for the selected attribute
            check R packages that how to use existed method in R (avoiding to implement by myself) 
        ''' 
#         chisquare(attr)
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
    
    def spark_self_sampling(self, rdd, percentage):
    
        '''rdd.takeSample(False, number_of_samples, seed = 10)
            Return a fixed-size sampled subset of this RDD.
            it's a list of sample data; not another RDD (not applicable since we need an rdd)
            Note: rdd.sample returns PipelinedRDD which is a subclass of RDD and inherits all APIs which RDD has
        '''
        return rdd.sample(False, percentage, seed = 10)    # End of Class ODHD    
    
def toVector(line, splitter):
    tokenized = line.split(splitter)
    return  np.array([float(tokenized[x]) for x in range(0,len(tokenized)-2)])    

def voted_subsapces():
    pass

def pre_process_normalize(rdd):
    
    all_attr_maxs = rdd.reduce(maxFunc)
    all_attr_mins = rdd.reduce(minFunc)
    '''
    np.savetxt('max.out', all_attr_max, delimiter = ',') 
    all_attr_maxs = np.loadtxt("max.out", delimiter = ',')
    '''
    normalizeDataRDD = rdd.map(lambda point: pre_normalize(point, all_attr_mins, all_attr_maxs, ln))
    f = normalizeDataRDD.collect()
    np.savetxt('/home/fatemeh/Data/saveData.txt', f)
    
def pre_normalize(input, mins, maxs, ln):
    for i in range(ln):
       input[i] = (input[i]-mins[i])/(maxs[i]-mins[i])   
    return (input)   

def main():
    spark_conf = SparkConf().setAppName("Different-Sampling data").setMaster('local[*]')
    spark_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sc = SparkContext(conf= spark_conf)
    GA.logInConsole(0, "input file read!")
    rdd = sc.textFile("/home/fatemeh/Data/saveData.txt",  minPartitions= 500, use_unicode=False)
    rdd.unpersist()
#     print('\nNumber of Partitions for this run: ', rdd.getNumPartitions())
    vectorRDD = rdd.map(lambda line: toVector(line, splitter = ' '))
    
    GA.logInConsole(0 , "Data Vectorized!")
    ss = list()
    GA.logInConsole(-1, 'Start the ensemble')
    GA.logInConsole(-10, "GA with range 3")
    ss.append(GA.parallel_GA_main(vectorRDD,sc, 5))
#     GA.logInConsole(-10, "GA with range 4")
#     ss.append(GA.parallel_GA_main(norm,sc, 4))
#     GA.logInConsole(-10, "GA with range 5")
#     ss.append(GA.parallel_GA_main(norm,sc, 5))
#     GA.logInConsole(-10, "GA with range 3 and Sampled data set")
#    sampleRDD = norm.sample(False, 0.6, seed=10)
#    ss.append(GA.parallel_GA_main(sampleRDD,sc, 3))
    print(ss)
    #selectedSS = voted_subsapces(ss)
#     SSD.outlierDetection(vectorRDD, ss)
    GA.logInConsole(100, "\nend of program")
    sc.stop()
    
if __name__ == "__main__":
    main()
    profile.run("main()")
    
