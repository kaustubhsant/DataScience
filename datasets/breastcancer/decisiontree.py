'''
Created on Nov 16, 2014

@author: Kaustubh Sant
@copyright: Copyright (c) 2014 Kaustubh Sant
@summary: Uses decision tree to learn and predict class label on the breast-cancer data
          taken from UCI repository (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
          Training dataset: 16% of complete data
          Missing values: substituted with middle value of range of values for attribute
          Prediction accuracy: 96.28%
'''

from sklearn import tree 
from sklearn.externals.six import StringIO  
import pydot

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
    count=0
    totallines=0
    with open("result.csv",'w') as fout:
        with open(InFile,'r') as fin:
            for lines in fin:
                totallines += 1
                line = lines.strip().split(",")[:-1]
                if("?" in line):
                    line[line.index("?")] = "5" 
                for item in line:
                    fout.write(str(item).strip("'") + ",")
                fout.write(str(clf.predict([line])[0]) + "\n")
                if(clf.score([line],[lines.strip().split(",")[-1]]) == 1.0 ):
                    count += 1
                    
    print("accuracy:" + str(count*100.00/totallines))
    
def plottree(clf):
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_png("decisiontree.png")     
                                 
def main(inFile):
    trainingdata,classlabels = gettrainingdata(inFile)
    clf = applydecisiontree(trainingdata,classlabels)
    plottree(clf)
    predictonmodel(clf,inFile)
    
if __name__ == '__main__':
    main(inputdataset)