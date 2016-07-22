import numpy as np
# I want this changes on VM
def outlierDetection(vectorRDD, subspaces):
    all_attr_maxs = np.loadtxt("max.out", delimiter = ',')
    all_attr_mins = np.loadtxt('min.out', delimiter = ',')
   
    avrg = 5
    for ss in subspaces:
        maxs = [all_attr_maxs[x] for x in ss[0]]
        mins = [all_attr_mins[x] for x in ss[0]]
        map2CellRDD = vectorRDD.map(lambda point: (assign2Cell(point[ss[0]], ss[0], \
                                                     maxs, mins, len(ss[0])), (point[ss[0]], 1)))
        sumPointsInCellRDD = map2CellRDD.map(lambda cell, (point, one): (cell, one)).reduceByKey(lambda a, b: a + b)
        tmp = sumPointsInCellRDD.filter(lambda x, y: y <= avrg) 
    return 