
import keras

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import rot_x

from utils import data_generator

model_path = 'used_models/modelcpnt8be37c37-3771-4d89-917f-5c6fdfb4f63f.keras'
data_path  = 'data/training'

model = keras.models.load_model(model_path)

gen =[i for i in data_generator(fitting=False)]

rxs, rys = gen[0][0],gen[0][1]

for i in range(5):
    plt.imshow(rxs[i], interpolation='nearest')
    print(rys[i])
    plt.show()


# predictions = model.predict(rxs)
# print(predictions)
