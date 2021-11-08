#Author : Mukunthan Tharmakulasingam (mukunthan1989@gmail.com)

from sklearn.base import clone
import numpy as np
import pandas as pd
from Utilils import *
import shap

delta=0.000000000000000000001

class RectifiedClassiferChain:
    def __init__(self,basemodel,type=0, optimized=False, optimizedmethod='CrossEntropy'):
        '''
        :param basemodel: Basemodel to be used in ClassiferChain
        '''
        self.ClassifierList = []
        self.model=basemodel
        self.updatedlabelorder=[]
        self.originallabelorder = []
        self.optimized=optimized
        self.optimizedmethod=optimizedmethod
        self.X=None
        self.Y=None
        self.type = type

    def trainRCC(self,X,Y):
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

        if(self.optimized==True):
            Y=self.OptimizeLabelOrder(Y)
            Categories = list(Y.columns.values)
        else:
            self.updatedlabelorder=self.originallabelorder
        k = 0
        for category in list(Categories):   ###Updated if needed
            #print("Category: " + category)
            modelc = self.ClassifierList[k]
            Xtrain = X
            Ytrain = Y
            Ytrain = Ytrain.dropna(subset=[category])  # , inplace=True)
            droppedindex = Y[~Y.index.isin(Ytrain.index)].index.values
            Xtrain = Xtrain.drop(droppedindex, axis=0)
            Ytrain = Ytrain[category]
            Ytrain = Ytrain.astype(int)
            modelc.fit(Xtrain, Ytrain.values.ravel())
            if (len(droppedindex) > 0):
                prediction = modelc.predict(X.iloc[droppedindex])
                i = 0
                for value in prediction:
                    Y.iloc[droppedindex[i]][category] = value
                    i = i + 1
            X = X.join(Y[category])
            k = k + 1
        return self.ClassifierList

    def predictRCC(self, X):
        '''
        :param X:
        :return:
        '''
        i = 0
        Y = pd.DataFrame(index=X.index)
        global TotalSumvalue
        TotalSumvalue = [[0 for x in range(X.shape[0])] for y in range(X.shape[1])]
        for classifer in self.ClassifierList:
            if (self.type==3):
                KErnalExplnanier = shap.TreeExplainer(classifer)
                class_shap_values = KErnalExplnanier.shap_values(X)
                if (i == 0):
                    TotalSumvalue = np.absolute(class_shap_values)
                else:
                    TotalSumvalue = TotalSumvalue + np.absolute(class_shap_values[:, :-i])

            prediction = classifer.predict(X)
            YPredict = pd.DataFrame(prediction, columns=[self.updatedlabelorder[i]])
            X = X.join(YPredict)
            Y = Y.join(YPredict)
            i = i + 1

        if (self.optimized == True):
            Y_rearranged = pd.DataFrame(index=Y.index)
            for index, element  in enumerate(self.originallabelorder):
                Y_rearranged = Y_rearranged.join(Y[element])
            Y=Y_rearranged

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
            YPredict = pd.DataFrame(prediction, columns=[self.updatedlabelorder[i]])
            Ypredprob=pd.DataFrame(predprobValue, columns=[self.updatedlabelorder[i]])
            X = X.join(YPredict)
            Y = Y.join(Ypredprob)
            i = i + 1

        if (self.optimized == True):
            Y_rearranged = pd.DataFrame(index=Y.index)
            for index, element  in enumerate(self.originallabelorder):
                Y_rearranged = Y_rearranged.join(Y[element])
            Y=Y_rearranged

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

    def getOptimizedLabelOrder(self):
        '''
        :param y_true:
        :param y_predict:
        :return:
        '''
        return self.updatedlabelorder

    def OptimizeLabelOrder(self, Y):
        '''
        :param Y:
        :return:
        '''
        Valuelist = []
        Yoptimized = pd.DataFrame(index=Y.index)

        for i in range(len(self.originallabelorder)):
            TotalValue = 0.0
            count=0.0
            if self.optimizedmethod == 'MissingRatio':
                ratio=getMissingratio(Y[self.originallabelorder[i]])
                Valuelist.append(ratio)
            else:
                for j in range(len(self.originallabelorder)):
                    if (i == j):
                        continue
                    Ydropped = Y.dropna(subset=[self.originallabelorder[i], self.originallabelorder[j]])
                    Ydropped = Ydropped.reset_index()
                    if(len(Ydropped) >= 50):
                        if self.optimizedmethod== 'CrossEntropy':
                            value = cEntropy(Ydropped[self.originallabelorder[i]], Ydropped[self.originallabelorder[j]])
                        else:
                            value=getConditionalProbability(Ydropped[self.originallabelorder[i]], Ydropped[self.originallabelorder[j]])

                        TotalValue = TotalValue + value
                        count=count+1
                Valuelist.append(TotalValue/(count+delta))
        Indexlist = np.argsort(Valuelist)
        for index in Indexlist:
            Yoptimized = Yoptimized.join(Y[self.originallabelorder[index]])
            self.updatedlabelorder.append(self.originallabelorder[index])
            #print(Yoptimized)
        return Yoptimized

    def getFeature(self, NoOfFeature=10, type=0, full=False):
        sim_all_df = pd.DataFrame()
        count=0
        Data=self.X
        for classifer in self.ClassifierList:

            if(count >0):
                Data=Data.join(self.Y[self.updatedlabelorder[count-1]])
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

    def getShapFeatures(self):
        if (self.type==3):
            return TotalSumvalue
        else:
            return 'Not Implemented'

'''
import pandas as pd
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold

df = pd.read_csv('Finalplfam_iddataset_Phenotype.csv',index_col=0)
Y=df[['ciprofloxacin','ampicillin','amoxicillin','gentamicin']]
droppeddf=df.drop(columns=['genome_id', 'genome_name','phenotype','ciprofloxacin','ampicillin','amoxicillin','gentamicin'])
X=droppeddf
model= SVC(class_weight='balanced',C=0.01,kernel='linear',gamma=1e-06,probability=True)
kfold = RepeatedKFold(n_splits=5,n_repeats=1, random_state=328228)
X=X.fillna(0)
scorelist=[]
for train_index, test_index in kfold.split(X, Y):
    x_train_tfidf = X.iloc[train_index]
    y_train_tfidf = Y.iloc[train_index]
    x_test_tfidf = X.iloc[test_index]
    y_test_tfidf = Y.iloc[test_index]
    rcc=RectifiedClassiferChain(model)
    classifierlist=rcc.trainRCC(X,Y)
    x_test_tfidf=x_test_tfidf.reset_index(drop=True)
    y_test_tfidf=y_test_tfidf.reset_index(drop=True)
    prediction=rcc.predictRCC(x_test_tfidf)
    print (prediction)
    score=rcc.Evalute(y_test_tfidf,prediction)
    scorelist.append(score)
print (np.mean(scorelist))
'''
