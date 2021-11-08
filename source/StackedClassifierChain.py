#Author : Mukunthan Tharmakulasingam (mukunthan1989@gmail.com)

import collections
from sklearn.base import clone
import numpy as np
import pandas as pd
from Utilils import *

delta=0.000000000000000000001

class StackedClassifierChain:
    def __init__(self,basemodel):
        '''
        :param basemodel: Basemodel to be used in ClassiferChain
        '''
        self.ClassifierList = []
        self.model=basemodel
        self.originallabelorder = []
        self.X=None
        self.Y=None
        self.LabelHashDic={}
        self.LabelCount={}

    def trainSCC(self,X,Y):
        '''
        :param X: Pandas Data Frame Features
        :param Y: Pandas Data Frame Labels
        :return: List of model
        '''
        self.X = X
        self.Y=Y
        Categories = list(Y.columns.values)
        for categories in Categories:
            self.ClassifierList.append(clone(self.model))
            self.originallabelorder.append(categories)
        
        k=0
        for category in list(Categories):   ###Updated if needed
            #print("Category: " + category)
            modelc = self.ClassifierList[k]
            Xtrain = X
            Ytrain =Y[category].fillna(0) # , inplace=True)
           
            modelc.fit(Xtrain, Ytrain.values.ravel())
            Ypred=modelc.predict(Xtrain)
            YPredict = pd.DataFrame(Ypred, columns=[self.originallabelorder[k]])
            X = X.join(YPredict)
        
            k=k+1
    
        return self.ClassifierList

    def predictSCC(self, X):
        '''
        :param X:
        :return:
        '''
        i = 0
        Y = pd.DataFrame(index=X.index)
        for classifer in self.ClassifierList:
            prediction = classifer.predict(X)
            YPredict = pd.DataFrame(prediction, columns=[self.originallabelorder[i]])
            X = X.join(YPredict)
            Y = Y.join(YPredict)
            i = i + 1

        return Y

    def predictProbRCC(self, X):
        '''
        :param X:
        :return:
        '''
        i = 0
        Y = pd.DataFrame(index=X.index)
        for classifer in self.ClassifierList:
            prediction = classifer.predict(X)
            predprob=classifer.predict_proba(X)
            predprobValue=(predprob[:,1]-predprob[:,0])
            #print(predprobValue)
            YPredict = pd.DataFrame(prediction, columns=[self.originallabelorder[i]])
            Ypredprob=pd.DataFrame(predprobValue, columns=[self.originallabelorder[i]])
            X = X.join(YPredict)
            Y = Y.join(Ypredprob)
            i = i + 1

        return Y

    def ModifiedHammingAccuracyscore(self, y_pred, y_true):
        '''
        :param y_pred:
        :param y_true:
        :return:
        '''
        scorelist=[]
        i = 0
        for value in ((y_true.values)):
            #print(i)
            match = 0.0
            total = 0.0
            predictedlabel = y_pred.iloc[i]
            j = 0
            for singlelabel in value:
                if (singlelabel == predictedlabel[j]):
                    match = match + 1
                    total = total + 1
                elif (not (np.isnan(singlelabel))):
                    total = total + 1
                j=j+1
            i = i + 1
            scorelist.append(match/total)
        return np.mean(scorelist)

    def ModifiedF1score(self, y_pred, y_true):
        '''
        :param y_pred:
        :param y_true:
        :return:
        '''
        Fscorelist=[]
        for i, value in enumerate((y_true.values)):
            TP, TN, FP, FN = 0.0000000001, 0.0000000001, 0.0000000001, 0.0000000001  ##Delta to avoid Zero division
            predictedlabel = y_pred.iloc[i]
            for j, singlelabel in enumerate(value):
                if ((np.isnan(singlelabel))):
                    continue
                elif (singlelabel ==1 and predictedlabel[j]==1):
                    TP = TP + 1
                elif (singlelabel ==1 and predictedlabel[j]==0):
                    FN = FN + 1
                elif (singlelabel ==0 and predictedlabel[j]==0):
                    TN = TN + 1
                elif (singlelabel ==0 and predictedlabel[j]==1):
                    FP = FP + 1
            precision = TP/(TP + FP)
            recall = TP/(TP + FN)
            fscore = (2 * precision * recall)/(precision + recall)
            Fscorelist.append(fscore)
        return np.mean(Fscorelist)

    def Evaluate(self, y_true, y_predict):
        '''
        :param y_true:
        :param y_predict:
        :return:
        '''
        return self.ModifiedHammingAccuracyscore(y_predict,y_true),self.ModifiedF1score(y_predict,y_true)



    def getFeature(self, NoOfFeature=10, type=0, full=False):
        sim_all_df = pd.DataFrame()
        count=0
        Data=self.X
        for classifer in self.ClassifierList:

            if(count >0):
                Data=Data.join(self.Y[self.originallabelorder[count-1]])
            if (type == 0):
                sim_t_df = pd.DataFrame([classifer.coef_[0]], columns=Data.columns)
            else:  ###RF
                sim_t_df = pd.DataFrame([classifer.feature_importances_], columns=Data.columns)

            sim_all_df = pd.concat([sim_all_df, sim_t_df], ignore_index=True)


            count=count+1
        if (full==True):
            return sim_all_df
        else:
            sim_all_df_T = sim_all_df.transpose().copy()
            sim_all_df_T["feature_weight_sum"] = sim_all_df_T.apply(lambda x: abs(x).sum(), axis=1)
            sim_all_df_T_top = sim_all_df_T.sort_values("feature_weight_sum", ascending=False)[:NoOfFeature]
            return sim_all_df_T_top
    
    
    def AddExisitingLabels(self,Labels):
        print('Labeladd')
        for lab in Labels:
            hashvalue=hash(tuple(lab))
            if hashvalue in self.LabelHashDic:
                self.LabelCount[hashvalue]=self.LabelCount[hashvalue]+1
            else:
                self.LabelHashDic[hashvalue]=lab
                self.LabelCount[hashvalue]=1
    
    def LabelSubsetCorrection(self, Labelset,columnsname):
        NewLabelset=pd.DataFrame(columns=columnsname)
        Closetset=[]
        for lab in Labelset:
            if hash(tuple(lab)) in self.LabelHashDic:
                Closetset=lab
                #print('Exisiitng. Ok')
            else:
                print('Not found' +str(lab))
                Distance=max(len(lab)*2,1000)
                count=0
                for key in self.LabelHashDic:
                    HammingDistance=sum(c1 != c2 for c1, c2 in zip(self.LabelHashDic[key],lab))
                    if (HammingDistance < Distance):
                        Distance = HammingDistance
                        Closetset=self.LabelHashDic[key]
                        count=self.LabelCount[key]
                        print('Found' + str(HammingDistance) +str(Closetset))
                    elif (HammingDistance == Distance and count < self.LabelCount[key]):
                        Distance = HammingDistance
                        Closetset=self.LabelHashDic[key]
                        count=self.LabelCount[key]
                        print('Found Another' +str(Closetset))

            labelseries=pd.Series(Closetset,columnsname)
            NewLabelset=NewLabelset.append(labelseries,ignore_index=True)
        
        return NewLabelset





'''
import pandas as pd
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold

df = pd.read_csv('Finalplfam_iddataset_Phenotype.csv',index_col=0)
Y=df[['ciprofloxacin','ampicillin','amoxicillin','gentamicin']]
droppeddf=df.drop(['genome_id', 'genome_name','phenotype','ciprofloxacin','ampicillin','amoxicillin','gentamicin'],axis=1)
X=droppeddf
columnnanme=['ciprofloxacin','ampicillin','amoxicillin','gentamicin']
X=X.fillna(0)
print("Before removing low variance: ", X.shape)
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
selector.fit_transform(X)
X = X[X.columns[selector.get_support()]].copy()
print("After removing low variance: ", X.shape)

model= SVC(class_weight='balanced',C=0.01,kernel='linear',gamma=1e-06,probability=True)
kfold = RepeatedKFold(n_splits=5,n_repeats=1, random_state=328228)

scorelist=[]
subsetresult=[]
for train_index, test_index in kfold.split(X, Y):
    x_train_tfidf = X.iloc[train_index]
    y_train_tfidf = Y.iloc[train_index]
    x_test_tfidf = X.iloc[test_index]
    y_test_tfidf = Y.iloc[test_index]

    scc=StackedClassiferChain(model)

    
    classifierlist=scc.trainSCC(X,Y)
    x_test_tfidf=x_test_tfidf.reset_index(drop=True)
    y_test_tfidf=y_test_tfidf.reset_index(drop=True)
    prediction=scc.predictSCC(x_test_tfidf)
    print (prediction)
    score=scc.Evaluate(y_test_tfidf,prediction) 
    scorelist.append(score)
    scc.AddExisitingLabels(y_train_tfidf.fillna(0).values)
    subsetcorrectpredict=scc.LabelSubsetCorrection(prediction.values,columnnanme)
    #print(subsetcorrectpredict)
    scoressc=scc.Evaluate(y_test_tfidf,subsetcorrectpredict) 
    subsetresult.append(scoressc)
print (np.mean(scorelist))
print(np.mean(subsetresult))

'''