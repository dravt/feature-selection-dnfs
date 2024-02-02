import numpy as np
from keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from feature_selector import FeatureSelector

EPOCHS = 150

a = 1
b = 251
interval = 2
trials = 4
N = 11
lambdas = np.linspace(a, b, N)
factor_loss = 0.5

# Prepare the dataset.
mnist = fetch_openml('mnist_784', version=1, cache=True)

samples = np.random.choice(len(mnist.target), int(0.1*len(mnist.target)), replace=False)

X = mnist.data.to_numpy()[samples].astype(float)
y = mnist.target[samples].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.reshape(X_train, (-1, 784))
X_test = np.reshape(X_test, (-1, 784))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#%% 

selector = FeatureSelector(selector_nodes = [784, 256 ,784],
               task_nodes = [784, 242, 64,10], 
               task_activation='softmax')


selector.compile(regression=False, loss='categorical_crossentropy', factor_loss = factor_loss,
            metric = 'categorical_accuracy')

#%%

history = selector.fit(EPOCHS, X_train, y_train, 
                       validation_data=(X_test, y_test),
                       batch_size = 64)

#%% 

selector.eval()