#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    This code trains a convolutional network to find flaws in 
    ultrasonic data. See https://arxiv.org/abs/1903.11399 for details.
'''
from gc import callbacks

import keras
from keras import backend as K
from keras import Input, layers
from keras import Model

import csv

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import uuid
import time
import tensorflow as tf

from utils import data_generator

SEED = 1337
keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

w,h = 256,256                       # initial data size
window = 7                          # window for the first max-pool operation

run_uuid = uuid.uuid4()             #unique identifier is generated for each run

vpath = "data/validation/"       #validation data path

'''     The data_generator reads raw binary UT data from the pre-processed files
        and preconditions it for ML training. '''


input_tensor = Input(shape=(w,h,1))

# start with max-pool to envelop the UT-data
ib = layers.MaxPooling2D(pool_size=(window,1),  padding='valid' )(input_tensor) # MaxPooling1D would work, but we may want to pool adjacent A-scans in the future

#build the network
cb = layers.Conv2D(96,3,padding='same', activation='relu')(ib)
cb = layers.Conv2D(64,3,padding='same', activation='relu')(cb)
cb = layers.MaxPooling2D( (2,8), padding='same')(cb)

cb = layers.Conv2D(48,3,padding='same', activation='relu')(cb)
cb = layers.Conv2D(32,3,padding='same', activation='relu')(cb)
cb = layers.MaxPooling2D( (3,4), padding='same' )(cb)
cb = layers.Flatten()(cb)
cb = layers.Dense(14, activation='relu', name='RNN')(cb)
iscrack = layers.Dense(1, activation='sigmoid', name='output')(cb)


model = Model(input_tensor, iscrack)
opt = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.)

model.compile(optimizer=opt, loss='binary_crossentropy' , metrics=['acc'])

xs = np.empty(0, dtype='float32')
ys = np.empty((0,2), dtype='float32')
input_files = [f for f in listdir(vpath) if isfile(join( vpath,f)) and f.endswith('.bins') ]
print("Validation files:")
for i in input_files:
    print(i)
    bxs = np.fromfile(vpath+i, dtype=np.uint16 ).astype('float32')
    bxs -= bxs.mean()
    bxs /= bxs.std()+0.0001
    xs = np.concatenate((xs, bxs))
    bys = np.loadtxt(vpath + i[:-5] +'.labels')
    ys = np.concatenate((ys, bys))
xs =np.reshape( xs, (-1,256,256,1), 'C')

callbacks = [keras.callbacks.ModelCheckpoint( 'modelcpnt'+str(run_uuid)+'.keras', monitor='val_loss', verbose=1, save_best_only=True)
            ]

aug_num = 40
gen = data_generator(path="data/training_100/", batch_size=70, augmentation=True, multiaug=False, pure_aug=True,
                     rotation_num=aug_num, noise_num=aug_num, flip_num=aug_num,shift_num=aug_num, amp_change_num=aug_num)

history = model.fit(gen,
                    epochs=50,
                    validation_data= (xs,ys[:,0]),
                    workers=8,
                    steps_per_epoch=60,
                    callbacks=callbacks,)


plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.legend(['val_acc', 'acc'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig("acc_history.png")
plt.show()

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss', 'loss'])
plt.title('Validation loss history')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig("loss_history.png")
plt.show()

np.savetxt('data.csv',(history.history['val_loss'],history.history['val_acc']), delimiter=',')