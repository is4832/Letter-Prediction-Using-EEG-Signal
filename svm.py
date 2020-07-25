import csv
import random
import math
import numpy as np

def loadDataset(filename,split,trainingSet_x=[],trainingSet_y=[],testSet_x=[],testSet_y=[]):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)):
            for y in range(0,13):
                if y<11:
                    dataset[x][y] = float(dataset[x][y])
            if random.random()<split:
                trainingSet_x.append(dataset[x][0:11])
                trainingSet_y.append(dataset[x][12])
            else:
                testSet_x.append(dataset[x][0:11])
                testSet_y.append(dataset[x][12])

def main():
    trainingSet_x=[]
    trainingSet_y=[]
    testSet_x=[]
    testSet_y=[]
    split=0.67
    loadDataset('EEG_Signal_Alphabets(A-J).csv',split,trainingSet_x,trainingSet_y,testSet_x,testSet_y)
    X = np.array(trainingSet_x)
    Y = np.array(trainingSet_y)
    print(X,Y)
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(X, Y)
    prediction = clf.predict(testSet_x)
    print("doing")
    print(prediction)    
main()
