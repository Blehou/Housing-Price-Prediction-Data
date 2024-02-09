# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 23:37:56 2023

@author: konain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



""" Exportation des données """

data = pd.read_csv('housing_price_dataset.csv')
df = data.copy() 

"""Observation des valeurs manquantes"""

#sns.heatmap(df.isna(), cbar = False)

""" Affichage """

#sns.pairplot(df)

""" Matrice de correlation """

df.drop(['Neighborhood'],axis = 1,inplace=False)
correlation = df.corr()
print(correlation)

""" Récupération des données pour le modèle """

Y = df['Price'].values.reshape(50000,1)
X = df['SquareFeet'].values.reshape(50000,1)


""" Train set and test set """

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)


""" GridSearchCV """

param_grid = {'alpha': np.linspace(0,3,15)
             }

grid_search = GridSearchCV(Lasso(), param_grid, cv = 10)

grid_search.fit(X_train, Y_train)

print("Best parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

Y_prediction = best_model.predict(X_test)

r2 = r2_score(Y_test, Y_prediction)

print('erreur entre y_test et y_pred : ', r2)










