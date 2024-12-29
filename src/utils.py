
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.layers import noise
from keras.preprocessing.image import ImageDataGenerator
from numpy.core.numeric import newaxis
import json

INPUT_LIMIT = 10


def data_generator(path = "data/training/", batch_size = 10, fitting=True, augmentation=True, multiaug=False, pure_aug=False,
                   rotation_num=0, noise_num=0, flip_num=0, shift_num=0, amp_change_num=0):
    print("Loading data from files")
    input_files = [f for f in listdir(path) if isfile(join( path,f)) and f.endswith('.bins') ]
    for i in input_files:
        print(i)
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

    if augmentation:
        if multiaug:
            print("Augmentation started",end='')
            if shift_num > 0:
                print("\nCreating", shift_num, " of shifted data")
                xs_shifted, ys_shifted =shift(xs, ys, shift_num)
                rand_idxs = np.random.permutation(len(xs_shifted))
                xs_shifted = xs_shifted[rand_idxs]
                ys_shifted = ys_shifted[rand_idxs]
            else:
                xs_shifted, ys_shifted = xs, ys
            if flip_num > 0:
                print("\nCreating", flip_num, " of flipped data")
                xs_flipped, ys_flipped = flip_image(xs_shifted, ys_shifted, flip_num)
                rand_idxs = np.random.permutation(len(xs_flipped))
                xs_flipped = xs_flipped[rand_idxs]
                ys_flipped = ys_flipped[rand_idxs]
            else:
                xs_flipped, ys_flipped = xs_shifted, ys_shifted
            if rotation_num > 0:
                print("\nCreating rotations", rotation_num, " of the data")
                xs_rot, ys_rot = random_rotation(xs_flipped, ys_flipped, rotation_num)
                rand_idxs = np.random.permutation(len(xs_rot))
                xs_rot = xs_rot[rand_idxs]
                ys_rot = ys_rot[rand_idxs]
            else:
                xs_rot, ys_rot = xs_flipped, ys_flipped
            if noise_num > 0:
                print("\nCreating", noise_num, " of noisy data with strength")
                xs_noise, ys_noise = gauss_noise(xs_rot, ys_rot, noise_num)
                rand_idxs = np.random.permutation(len(xs_noise))
                xs_noise = xs_noise[rand_idxs]
                ys_noise = ys_noise[rand_idxs]
            else:
                xs_noise, ys_noise = xs_rot, ys_rot
            if amp_change_num > 0:
                print("\nCreating", amp_change_num, " of data with smaller amplitude")
                xs_amp_changed, ys_amp_changed = amp_change(xs_noise, ys_noise, amp_change_num)
            else:
                xs_amp_changed, ys_amp_changed = xs_noise, ys_noise

            if pure_aug:
                xs = xs_amp_changed
                ys = ys_amp_changed
            else:
                xs = np.concatenate((xs, xs_amp_changed))
                ys = np.concatenate((ys, ys_amp_changed))
        else:
            print("\nAugmentation started")
            if rotation_num > 0 : print("\nCreating", rotation_num,"rotations of the data")
            xs_rot, ys_rot = random_rotation(xs,ys, rotation_num)
            rand_idxs = np.random.permutation(len(xs))
            xs = xs[rand_idxs]
            ys = ys[rand_idxs]
            if noise_num > 0 : print("\nCreating", noise_num ," of noisy data")
            xs_noise, ys_noise = gauss_noise(xs, ys, noise_num)
            rand_idxs = np.random.permutation(len(xs))
            xs = xs[rand_idxs]
            ys = ys[rand_idxs]
            if flip_num > 0 : print("\nCreating", flip_num ," of flipped data")
            xs_flipped, ys_flipped = flip_image(xs,ys,flip_num)
            rand_idxs = np.random.permutation(len(xs))
            xs = xs[rand_idxs]
            ys = ys[rand_idxs]
            if shift_num > 0 : print("\nCreating", shift_num ," of shifted data")
            xs_shifted, ys_shifted =shift(xs, ys,shift_num)
            rand_idxs = np.random.permutation(len(xs))
            xs = xs[rand_idxs]
            ys = ys[rand_idxs]
            if amp_change_num >0 : print("\nCreating",amp_change_num," of data with smaller amplitude")
            xs_amp_changed, ys_amp_changed = amp_change(xs, ys, amp_change_num)

            if pure_aug:
                xs = np.concatenate((xs_rot, xs_noise, xs_flipped, xs_shifted, xs_amp_changed))
                ys = np.concatenate((ys_rot, ys_noise, ys_flipped, ys_shifted, ys_amp_changed))
            else:
                xs = np.concatenate((xs, xs_rot, xs_noise, xs_flipped, xs_shifted, xs_amp_changed))
                ys = np.concatenate((ys, ys_rot, ys_noise, ys_flipped, ys_shifted, ys_amp_changed))
        print("\nAugmentation finished")
    print("The training dataset contains ",xs.shape[0], "images")
    if not fitting: # for testing purposes
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
    if num == 0:
        xs = np.empty((0, 256, 256, 1), dtype='float32')
        ys = np.empty((0, 2), dtype='float32')
        return xs, ys
    xs_aug = []
    ys_aug = []
    i = 0
    while i < num:
        for j, x in enumerate(xs):
            print(i+1,"/",num,'\r',end='')
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

def gauss_noise(xs,ys,num):
    if num == 0:
        xs = np.empty((0, 256, 256, 1), dtype='float32')
        ys = np.empty((0, 2), dtype='float32')
        return xs, ys
    xs_aug = []
    ys_aug = []
    i=0
    while i < num:
        for j, x in enumerate(xs):
            noise = np.max(x) * np.random.randint(0,10)/100
            print(i+1,"/",num,'\r',end='')
            n = np.random.normal(0,noise,x.shape).astype('float32')
            x = x+n
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
    ys_aug = []
    i=0
    while i*2 < num and i < len(xs):
        print((i+1)*2, "/", num, '\r', end='')
        flipped_x1 = np.flipud(xs[i]).astype('float32')
        xs_aug.append(flipped_x1)
        ys_aug.append(ys[i])
        flipped_x2 = np.fliplr(xs[i]).astype('float32')
        xs_aug.append(flipped_x2)
        ys_aug.append(ys[i])
        i += 1
    xs_aug = np.array(xs_aug)
    ys_aug = np.array(ys_aug)
    return xs_aug, ys_aug

def shift(xs,ys,num):
    if num == 0:
        xs = np.empty((0, 256, 256, 1), dtype='float32')
        ys = np.empty((0, 2), dtype='float32')
        return xs, ys
    xs_aug = []
    ys_aug = []
    i=0
    while i < num:
        for j, x in enumerate(xs):
            print(i+1,"/",num,'\r',end='')
            pos_shift = np.random.randint(-25,25)
            ax = np.random.randint(0,2)
            x = np.roll(x, pos_shift, axis=ax)
            xs_aug.append(x)
            ys_aug.append(ys[j])
            i += 1
            if i >= num:
                break
    ys_aug = np.array(ys_aug)
    xs_aug = np.array(xs_aug)
    return xs_aug, ys_aug

def amp_change(xs,ys,num):
    if num == 0:
        xs = np.empty((0, 256, 256, 1), dtype='float32')
        ys = np.empty((0, 2), dtype='float32')
        return xs, ys
    xs_aug = []
    ys_aug = []
    i = 0
    a_change = np.random.randint(50,100)/100
    while i < num:
        for j, x in enumerate(xs):
            print(i + 1, "/", num, '\r', end='')
            x = x * a_change
            xs_aug.append(x)
            ys_aug.append(ys[j])
            i += 1
            if i >= num:
                break
    ys_aug = np.array(ys_aug)
    xs_aug = np.array(xs_aug)
    return xs_aug, ys_aug
