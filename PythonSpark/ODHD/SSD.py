def outlierDetection(vectorRDD, subspaces):
    avrg = 5
    for ss in subspaces:
        map2CellRDD = rdd.map(lambda point: (assign2Cell(point[ss], ss, \
                                                     maxs, mins, len(ss)), (point[ss], 1)))
        sumPointsInCellRDD = map2CellRDD.map(lambda cell, (point, one): (cell, one)).reduceByKey(lambda a, b: a + b)
        tmp = sumPointsInCellRDD.filter(lambda x, y: y <= avrg) 
    return 