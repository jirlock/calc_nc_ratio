import matplotlib.pyplot as plt
import os
import pickle

with open('prediction.pickle', 'rb') as f:
    pred = pickle.load(f)

for x in pred:
    plt.imshow(x)
    plt.show()

