from matplotlib import pyplot as plt
import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
chaise_pca = [8.685286, 15.67155513, 33.23345357]
sol_pca = [11.9412356, 17.567176]
chaise_pcl, sol_pcl = pickle.load(open('centroid_chaise_sol_ae.p', 'rb'))

box = plt.boxplot([chaise_pca, sol_pca, chaise_pcl, sol_pcl], patch_artist=True)

colors = ['red', 'green',
         'red', 'green']

for patch, color in zip(box['boxes'], colors):
   patch.set_facecolor(color)

hR, = plt.plot([1,1], 'r-')
hG, = plt.plot([1,1], 'g-')
plt.legend([hR, hG], ['Sur une chaise', 'Au sol'])
plt.xticks([1.5, 3.5],['2', '3'])
hR.set_visible(False)
hG.set_visible(False)
plt.ylabel('MAE (BPM)')
plt.xlabel('Méthode')
plt.pause(0.00001)

assert True