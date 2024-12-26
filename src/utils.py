
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.layers import noise
from keras.preprocessing.image import ImageDataGenerator
from numpy.core.numeric import newaxis

INPUT_LIMIT = 30


def data_generator(path = "data/training/", batch_size = 10, fitting=True, augmentation=True, rotation=0,
                   noise_num=0, noise_level=25, flip=0):
    print("Loading data from files")
    input_files = [f for f in listdir(path) if isfile(join( path,f)) and f.endswith('.bins') ]
    np.random.shuffle(input_files)          # we'll take random set from available data files
    input_files = input_files[0:INPUT_LIMIT]        # limit to INPUT_LIMIT files per epoch
    xs = np.empty(0, dtype='float32')    #  input data
    ys = np.empty((0,2), dtype='float32')   #  label data
    for j,i in enumerate(input_files):
        print(j,"/",len(input_files),"\r",end='')
        bxs = np.fromfile(path+i, dtype=np.uint16).astype('float32')
        bxs -= bxs.mean()
        bxs /= bxs.std() +0.00001           #avoid division by zero
        xs = np.concatenate((xs,bxs))
        bys = np.loadtxt(path + i[:-5] +'.labels')
        ys = np.concatenate((ys,bys) )

    xs = np.reshape(xs, (-1,256,256,1), 'C')

    print(len(xs))

    if augmentation:
        print("Augmentation started")
        if rotation > 0 : print("\nCreating rotation of the data", rotation, "times")
        xs_rot, ys_rot = random_rotation(xs,ys, rotation)
        if noise_num > 0 : print("\nCreating noisy data with strength", noise_level)
        xs_noise, ys_noise = gauss_noise(xs, ys, noise_level, noise_num)
        if flip: print("\nCreating flipped data")
        xs_flipped, ys_flipped = flip_image(xs,ys,flip)

        print('\n',len(xs) ,len(xs_rot), len(xs_noise), len(xs_flipped))
        xs = np.concatenate((xs, xs_rot, xs_noise, xs_flipped))
        ys = np.concatenate((ys, ys_rot, ys_noise, ys_flipped))
        print(len(xs))

        rand_idxs = np.random.permutation(len(xs))
        xs = xs[rand_idxs]
        ys = ys[rand_idxs]
        print("Augmentation finished")
    print("The training dataset contains ",xs.shape[0], "images")
    if not fitting:
        yield xs, ys
    else:
        rows = xs.shape[0]
        cursor = 0
        while True:
            start = cursor
            cursor += batch_size
            if cursor > rows:
                cursor = 0
            bxs = xs[start:cursor,:,:,:]
            bys = ys[start:cursor,0]
            yield bxs, bys

def random_rotation(xs,ys,num):
    xs_aug = []
    ys_aug = []
    i = 0
    while i < num:
        for j, x in enumerate(xs):
            print(i,"/",num,'\r',end='')
            datagen = ImageDataGenerator(rotation_range=15,fill_mode='constant')
            x = next(datagen.flow(x[newaxis, :, :, :],batch_size=1))
            xs_aug.append(x[0])
            ys_aug.append(ys[j])
            i += 1
            if i >= num:
                break
    ys_aug = np.array(ys_aug)
    xs_aug = np.array(xs_aug)
    return xs_aug, ys_aug

def gauss_noise(xs,ys,strength,num):
    xs_aug = []
    ys_aug = []
    i=0
    while i < num:
        str_n = strength/(np.floor(i / len(xs))+1)
        for j, x in enumerate(xs):
            print(i,"/",num,'\r',end='')
            n = np.random.normal(0,str_n,x.shape).astype('float32')
            x = np.clip(x + n,0,255).astype('float32')
            xs_aug.append(x)
            ys_aug.append(ys[j])
            i += 1
            if i >= num:
                break
    ys_aug = np.array(ys_aug)
    xs_aug = np.array(xs_aug)
    return xs_aug, ys_aug

def flip_image(xs, ys, num):
    if num == 0:
        xs = np.empty((0, 256, 256, 1), dtype='float32')
        ys = np.empty((0, 2), dtype='float32')
        return xs, ys
    xs_aug = []
    i=0
    while i*2 < num or i <= len(xs):
        print(i, "/", num, '\r', end='')
        flipped_x1 = np.flipud(xs[i]).astype('float32')
        xs_aug.append(flipped_x1)
        flipped_x2 = np.fliplr(xs[i]).astype('float32')
        xs_aug.append(flipped_x2)
        i += 1
    xs_aug = np.array(xs_aug)
    ys_aug = np.concatenate((ys, ys))
    return xs_aug, ys_aug