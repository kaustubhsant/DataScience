'''
Created on Nov 16, 2014

@author: Kaustubh Sant
@copyright: Copyright (c) 2014 Kaustubh Sant
@summary: Uses decision tree to learn and predict class label on the breast-cancer data
          taken from UCI repository (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
          Training dataset: 16% of complete data
          Prediction accuracy: 96.42%
'''

from sklearn import tree 

inputdataset = "breast-cancer-wisconsin.data"

def gettrainingdata(inFile):
    trainingdata=[]
    classlabels=[]
    with open(inFile,'r') as fin:
        lines= fin.readlines()
        for i in range(0,len(lines),6):
            if("?" not in lines[i]):
                line=[]
                line = lines[i].strip().split(",")
                trainingdata.append(line[:-1])
                classlabels.append(line[-1])
    return trainingdata,classlabels

def applydecisiontree(data,labels):
    clf= tree.DecisionTreeClassifier(min_samples_leaf=2)
    clf.fit(data,labels)
    return clf

def predictonmodel(clf,InFile):
    with open("result.csv",'w') as fout:
        with open(InFile,'r') as fin:
            for lines in fin:
                line = lines.strip().split(",")[:-1]
                if("?" in line):
                    line[line.index("?")] = "0"
                for item in line:
                    fout.write(str(item).strip("'") + ",")
                fout.write(str(clf.predict([line])[0]) + "\n")

def calaccuracy(inFile):
    count=0
    with open(inFile,'r') as fin:
        l1=fin.readlines()                
    with open("result.csv",'r') as fin:
        l2=fin.readlines()
    for i in range(len(l1)):
        if(l1[i].split(",")[-1] != l2[i].split(",")[-1]):
            count += 1
    print("accuracy:" + str((len(l1)-count)*100.00/len(l1)))  
                                 
def main(inFile):
    trainingdata,classlabels = gettrainingdata(inFile)
    clf = applydecisiontree(trainingdata,classlabels)
    predictonmodel(clf,inFile)
    calaccuracy(inFile)
    
if __name__ == '__main__':
    main(inputdataset)