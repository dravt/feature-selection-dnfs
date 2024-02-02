#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys 

sys.path.append('/home/nhgedoer/nhgedoer/icml24/feature-selection')

from feature_selector import FeatureSelector

#%% Prepare dataset

#Add path here
LOAD_PATH = ''

a = 1
b = 251
trials = 4
N = 11
lambdas = np.linspace(a, b, N)
factor_loss = 20
epochs = 600

#%%

df = pd.read_csv(f'{LOAD_PATH}/AmesHousing.csv')

df = df.fillna(0)

X = df.drop(['SalePrice', 'PID', 'Order'], axis=1)
X = pd.get_dummies(X, drop_first=True)

y = df['SalePrice']

y = (y-y.min())/(y.max()-y.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

std_scaler = preprocessing.StandardScaler()
X_train = std_scaler.fit_transform(X_train.astype(float))
X_test = std_scaler.transform(X_test.astype(float))


#%%

selector = FeatureSelector(selector_nodes = [276,256,276],
               task_nodes = [276, 128, 64, 32, 1])

selector.compile(factor_loss = factor_loss)

history = selector.fit(epochs, X_train, y_train, validation_data=(X_test, y_test), verbose=2)

#%%

selector.eval()