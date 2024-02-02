#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from synthetic_generator import generate_data
from feature_selector import FeatureSelector

NUM = 1500
EPOCHS = 100
FACTOR_LOSS = 30
EPOCHS = 400

selector = FeatureSelector(task_nodes = [10,20,100,50,50,25,1],
               selector_nodes = [10,100,75,50,10],
               task_activation='sigmoid')

#%%

selector.compile(factor_loss=FACTOR_LOSS)

#%%

df = generate_data(linear=True)
X = df.drop(['y'], axis=1)
y = df['y']

y = (y-y.min())/(y.max()-y.min())

eighty = int(2/3*NUM)
X_train = X[:eighty]
X_test = X[eighty:]
y_train = y[:eighty]
y_test = y[eighty:]

history = selector.fit(EPOCHS, X_train, y_train, validation_data=(X_test, y_test))

#%% 

selector.eval()
