#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys 

sys.path.append('/home/nhgedoer/nhgedoer/icml24/feature-selection')

from feature_selector import FeatureSelector

#%% Define constants 

# Add path here
# LOAD_PATH = sys.argv[1]
LOAD_PATH = ''

factor_loss = 20
epochs = 600

#%% Prepare dataset. 

df = pd.read_csv(f'{LOAD_PATH}/AmesHousing.csv')

df = df.fillna(0)

X = df.drop(['SalePrice', 'PID', 'Order'], axis=1)
X = pd.get_dummies(X, drop_first=True)

y = df['SalePrice']

# Normalize target data
y = (y-y.min())/(y.max()-y.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

std_scaler = preprocessing.StandardScaler()
X_train = std_scaler.fit_transform(X_train.astype(float))
X_test = std_scaler.transform(X_test.astype(float))


#%% Perform selection.

selector = FeatureSelector(selector_nodes = [276,256,276],
               task_nodes = [276, 128, 64, 32, 1])

selector.compile(factor_loss = factor_loss)

history = selector.fit(epochs, X_train, y_train, validation_data=(X_test, y_test), verbose=2)

#%% Print feature selection. Not that just the feature indices are printed. 

selected_features = selector.eval()
print(f'Selected: {selected_features}. Size: {len(selected_features)}')