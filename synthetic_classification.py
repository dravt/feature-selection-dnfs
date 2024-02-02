#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.utils import to_categorical

from synthetic_generator import generate_data
from feature_selector import FeatureSelector

NUM = 1500
EPOCHS = 100
FACTOR_LOSS = 3
THRESHOLD = 0.01
EPOCHS = 300

selector = FeatureSelector(task_nodes = [10,20,60,30,20,10,2],
               selector_nodes = [10,100,75,50,10],
               task_activation='sigmoid')

#%%

selector.compile(factor_loss=FACTOR_LOSS,
            threshold=THRESHOLD,
            regression = False,
            loss = 'categorical_crossentropy',
            metric = 'categorical_accuracy')

df = generate_data(linear=True, classification=True)
X = df.drop(['y'], axis=1)
y = to_categorical(df['y'].to_numpy())

#%%

eighty = int(2/3*NUM)
X_train = X[:eighty]
X_test = X[eighty:]
y_train = y[:eighty]
y_test = y[eighty:]

history = selector.fit(EPOCHS, X_train, y_train, validation_data=(X_test, y_test))

#%% 

selector.eval()
