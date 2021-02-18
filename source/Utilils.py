import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

delta=0.000000000000000000001
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count / len(Y)
    en = np.sum((-1) * prob * np.log2(prob))
    return en

# Joint Entropy
def jEntropy(Y, X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y, X]
    #print(YX)
    #print(len(YX))
    return entropy(YX)

# Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)

def getConditionalProba4Positive(A,B):
    count=0.0
    total=0.0
    i=0
    for value in A:
        if(value==1 and B.at[i] ==1):
            count=count+1
            total=total+1
        elif (value==1 and B.at[i]==0):
            total=total+1
        i=i+1
    return count/(total+delta),total

def getConditionalProba4Negative(A,B):
    count=0.0
    total=0.0
    i=0
    for value in A:
        if(value==0 and B.at[i] ==0):
            count=count+1
            total=total+1
        elif (value==0 and B.at[i]==1):
            total=total+1
        i=i+1
    return count/(total+delta), total

def getConditionalProbability(Y1,Y2):
    pfsP, lengthP = getConditionalProba4Positive(Y1, Y2)
    pfsN, lengthN = getConditionalProba4Negative(Y1, Y2)
    conditionalProb=(lengthP/(lengthP+lengthN))*(1-pfsP)+(lengthN/(lengthP+lengthN))*(1-pfsN)
    return conditionalProb


def getMissingratio(Y):
    Ydropped = Y.dropna()  # , inplace=True)
    droppedindex = Y[~Y.index.isin(Ydropped.index)].index.values
    #print(len(droppedindex)/len(Y))
    return (len(droppedindex)/len(Y))