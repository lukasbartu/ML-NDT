
import keras
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import rot_x

from utils import data_generator

# model_path = 'used_models/modelcpnt8be37c37-3771-4d89-917f-5c6fdfb4f63f.keras'
data_path  = 'data/training'
data = 'data/training/0A4A7A0F-A8A4-40A5-95A3-2D15AEC422E3'

# model = keras.models.load_model(model_path)

# gen =[i for i in data_generator(fitting=False, noise_num=10, noise_level=1)]
# xs, ys = gen[0][0],gen[0][1]

js = []
with open(data+'.jsons','r') as file:
    content = file.read().strip()
    objects = content.split('}{')
    objects[0] = objects[0][1:]
    objects[-1] = objects[-1][:-1]
    for obj in objects:
        j = json.loads("{"+obj+"}")
        js.append(j["flaws"][0])

bxs = np.fromfile(data + '.bins', dtype=np.uint16 ).astype('float32')
bxs -= bxs.mean()
bxs /= bxs.std()+0.0001
bys = np.loadtxt( data + '.labels')
xs = np.reshape( bxs, (-1,256,256,1), 'C')
p = np.empty((256, 256, 1), dtype='float32')

m_j = (js[2]["location"]).split("-")
m_j = [int(m_j[0]), int(m_j[1])]
t=0
for i in range(256):
    for j in range(256):
        t=t+1
        if m_j[0] < t < m_j[1]:
            p[i][j] = 10
        else:
            p[i][j] = 0

plt.imshow(xs[2]+p, interpolation='nearest')
print(m_j, bys[2])
plt.show()

# for i in range(1):
#     plt.imshow(xs[i], interpolation='nearest')
#     print(ys[i])
#     plt.show()


# predictions = model.predict(rxs)
# print(predictions)
