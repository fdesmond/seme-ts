#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:01:32 2020

@author: jubidan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subsample import *
from models import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import metrics


#lecture données
data_A = pd.read_csv('energydata_complete.csv')  
del data_A['rv2']

N = len(data_A)


## SUBSAMPLING
print("Voulez vous conserver un nombre d'observations (1) ou une proportion d'observations (2)")
choix = int(input())

if choix == 1:
    print('Combien d''observations voulez-vous conserver ?')
    n = int(input())
    #selection aléatoire sur la dataframe de n observations
    index_B1 = subsample().sampleObservation(N, n=n)
    data_B1 = data_B2 = data_A.iloc[index_B1]
    #selection d'une portion d'observations consécutives de taille n
    index_B2 = subsample().sampleSequence(N, n=n)
    data_B2 = data_A.iloc[index_B2]
    
if choix == 2:
    print('Quelle proportion des données voulez-vous conserver ?')
    p = float(input())
    #selection aléatoire d'une proportion p d'observations
    index_B1 = subsample().sampleObservation(N, prop = p)
    data_B1 = data_B2 = data_A.iloc[index_B1]
    #selection d'une proportion p d'observations consécutives 
    index_B2 = subsample().sampleSequence(N, prop = p)
    data_B2 = data_A.iloc[index_B2]
    
X_data = data_A.loc[:, data_A.columns != 'rv1']
del X_data['date']

Y_data = data_A.loc[:, 'rv1']
    
modelisation = models()
modelisation.estimate(y = Y_data, X = X_data, proptrain = 0.7)
modelisation.evaluate()

plt.bar(x = np.arange(4), height=np.sqrt(modelisation.evaluate()), tick_label = ['Regression', 'Tree', 'Bagging', 'KNN'])










