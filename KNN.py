import numpy as np
import operator

def createDataSet():
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['愛情片','愛情片','動作片','動作片']
    return group,labels

def classify0(inX, dataSet,labels, k):
    #返回dataS行數
    dataSetSize = dataSet.shape[0]
    #在列向量方向複製inX共一次(橫向),行向量方向上複製inX共dataSetSize次(縱向)
    diffMat = np.tile(inX,(dataSetSize, 1)) - dataSet
    #相減平方
    sqDiffMat = diffMat**2
    #sum(0)列相加,sum(1)行相加
    sqlDistances = sqDiffMat.sum(axis=1)
    #開根號
    distances = sqlDistances**0.5
    #由小到大排序
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        #取前k個類別
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get()方法,返回指定鍵的值,如果值不在字典中返回默認值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #items代表iteritems
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的鍵进行排序
    #reverse降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    #返回最多類別
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group, labels = createDataSet()
    test = [101,20]
    test_class = classify0(test, group, labels, 3)
    print(test_class)