
import numpy as np
import matplotlib.pyplot as plt
import csv
from os import listdir
from os.path import isfile, join
import os

from keras.testing_infra.test_utils import run_without_tensor_float_32
from matplotlib.pyplot import legend

data_files = [f for f in listdir(os.getcwd()) if isfile(join( os.getcwd(),f)) and f.endswith('.csv') and f.startswith("data_500")]

legends=[]
for data_file in data_files:
    with open(data_file, newline='') as f:
      reader = csv.reader(f)
      for row in reader:
        break

    for i,r in enumerate(row):
        row[i] = float(row[i])

    plt.plot(row)
    legends.append(str(data_file[9:-4]))
plt.legend(legends)
plt.title('Validation loss of different training configurations')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig("loss results.svg", bbox_inches='tight', format='svg')
plt.show()

legends=[]
for data_file in data_files:
    i = 0
    with open(data_file, newline='') as f:
      reader = csv.reader(f)
      for row in reader:
        if i == 0:
            i+=1
            continue
        break

    for i,r in enumerate(row):
        row[i] = float(row[i])

    plt.plot(row)
    legends.append(str(data_file[9:-4]))
plt.legend(legends)
plt.title('Validation accuracy of different training configurations')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig("accuracy results.svg", bbox_inches='tight', format='svg')
plt.show()
