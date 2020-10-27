# Chargement des librairies
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor



class models():
  def __init__(self):
    self.ytrain = None
    self.ytest = None
    self.Xtrain = None
    self.Xtest = None
    self.trainIndex = None
    self.testIndex = None

       


  def estimate(self, y, X, proptrain): 
        lentrajectory = X.shape[0]
        nb_train = int(lentrajectory*proptrain)
        self.trainIndex = np.arange(nb_train)
        self.testIndex = np.arange(nb_train, lentrajectory)
        
        self.ytrain = y.loc[self.trainIndex.tolist()]
        self.ytest = y.loc[self.testIndex.tolist()]
        self.Xtrain = X.loc[self.trainIndex.tolist(), :]
        self.Xtest = X.loc[self.testIndex.tolist(), :]

        
        ##  régression linéaire
        self.reg = LinearRegression().fit(self.Xtrain, self.ytrain)
        
        ## arbre de régression
        self.regtree = DecisionTreeRegressor().fit(self.Xtrain, self.ytrain)       
        
        ## bagging 
        self.regbg = BaggingRegressor().fit(self.Xtrain, self.ytrain)

        ## knn
        self.regknn = KNeighborsRegressor().fit(self.Xtrain, self.ytrain)


  def prediction(self):
    return (self.reg.predict(self.Xtest), self.regtree.predict(self.Xtest),
               self.regbg.predict(self.Xtest),
               self.regknn.predict(self.Xtest))
  
  
  def evaluate(self):
    return (np.mean((self.reg.predict(self.Xtest) - self.ytest)**2),
                np.mean((self.regtree.predict(self.Xtest) - self.ytest)**2),
                np.mean((self.regbg.predict(self.Xtest) - self.ytest)**2),
                np.mean((self.regknn.predict(self.Xtest) - self.ytest)**2),
                )     

  
 