#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    This code trains a convolutional network to find flaws in 
    ultrasonic data. See https://arxiv.org/abs/1903.11399 for details.
'''

import keras
from Tools.i18n.makelocalealias import optimize
from keras import backend as K
from keras import Input, layers
from keras import Model
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import uuid
import time

from utils import data_generator



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
opt = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.)

# class_weights = [1.47,0.76]  # class weights for 66 percent of data containing defect
class_weights = [0.5, 0.5]

model.compile(optimizer=opt, loss='binary_crossentropy' , metrics=['acc'], loss_weights=class_weights)
model.summary()

xs = np.empty(0, dtype='float32')
ys = np.empty((0,2), dtype='float32')
input_files = [f for f in listdir(vpath) if isfile(join( vpath,f)) and f.endswith('.bins') ]
for i in input_files:
    bxs = np.fromfile(vpath+i, dtype=np.uint16 ).astype('float32')
    bxs -= bxs.mean()
    bxs /= bxs.std()+0.0001
    xs = np.concatenate((xs, bxs))
    bys = np.loadtxt(vpath + i[:-5] +'.labels')
    ys = np.concatenate((ys, bys))
xs =np.reshape( xs, (-1,256,256,1), 'C')

callbacks = [keras.callbacks.ModelCheckpoint( 'modelcpnt'+str(run_uuid)+'.keras', monitor='val_loss', verbose=1, save_best_only=True)
            ]

# RECOMENDED: rotation 1, noise_num 2, noise_level < 50, flip=0!
gen = data_generator(batch_size=70, augmentation=True,
                     rotation=1000, noise_num=1000, noise_level=40, flip=0)
history = model.fit(gen,
                    epochs=100,
                    validation_data= (xs,ys[:,0]),
                    workers=8,
                    steps_per_epoch=60,
                    callbacks=callbacks,)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'val'])
plt.title('Accuracy history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig("acc_history.png")
plt.show()

# BEST VAL_LOSS = 0.16342 AUG
# BEST VAL_LOSS =


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'val'])
plt.title('Loss history')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig("loss_history.png")
plt.show()