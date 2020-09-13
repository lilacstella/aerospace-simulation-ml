from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

from kmeans_initializer import InitCentersKMeans
from rbflayer import RBFLayer


def csv_to_np_array(file_path):
    return np.array(pd.read_csv(file_path))


def shuffle_split(a, shuffle_seed):
    np.random.seed(shuffle_seed)
    np.random.shuffle(a)
    temp = np.split(a, indices_or_sections=[0, int(0.2 * len(a) + 1)])
    return temp[1], temp[2]


x = csv_to_np_array('parameters.csv')
y = csv_to_np_array('solver_settings.csv')
seed = random.randrange(0, 99999)

xtest, xtrain = shuffle_split(x, seed)
ytest, ytrain = shuffle_split(y, seed)

rbflayer = RBFLayer(10, initializer=InitCentersKMeans(xtrain), betas=3.0, input_shape=(2,))

model = tf.keras.models.Sequential()
model.add(rbflayer)
model.add(tf.keras.layers.Dense(5, activation='linear', name='output'))

tb_cb = tf.keras.callbacks.TensorBoard(log_dir="logs/" + datetime.datetime.now().strftime("%H:%M"), histogram_freq=1)

tf.keras.optimizers.Adam(learning_rate=0.5)
model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_error')
model.fit(x=xtrain, y=ytrain, epochs=4000, verbose=2, callbacks=[tb_cb], initial_epoch=0,
          validation_data=(xtest, ytest))

model.save('model.h5')
model.predict(xtest)
invtest = csv_to_np_array('solver_settings.csv')[0:18]
ypred = model.predict(xtest)
print(ypred)
mae = np.array(
    [metrics.mean_absolute_error(invtest[:, 0], ypred[:, 0]), metrics.mean_absolute_error(invtest[:, 1], ypred[:, 1]),
     metrics.mean_absolute_error(invtest[:, 2], ypred[:, 2]), metrics.mean_absolute_error(invtest[:, 3], ypred[:, 3]),
     metrics.mean_absolute_error(invtest[:, 4], ypred[:, 4])])
print(mae)

print(np.average(mae))
print(metrics.mean_squared_error(invtest, ypred))

examples = [[7.0, 0.87], [8.0, 0.95]]
print(model.predict(examples))
