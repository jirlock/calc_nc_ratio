import matplotlib.pyplot as plt
import os
import numpy as np

ncratio = []

for x in os.listdir('test_images'):
    c = 'test_masks/' + x.replace('.jpg', '.mask.0.png')
    n = 'test_masks/' + x.replace('.jpg', '.mask.1.png')
    carray = plt.imread(c)
    narray = plt.imread(n)
    ncr = np.sum(narray) / np.sum(carray)
    ncratio.append(ncr)

print(ncratio)