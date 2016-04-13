'''
Created on Nov 28, 2015

@author: Fatemeh
'''

from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    sparkconf = SparkConf().setAppName("arash").setMaster("local[*]")
    sc = SparkContext(conf = sparkconf)
    lines = sc.textFile("testFile1.txt")
    lines.count()
