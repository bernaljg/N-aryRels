import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import *
import matplotlib
import numpy as np

style=['b:*','r:s','c:p']
prectage =  [0.001, 0.01, 0.1, 1.0]
lprectage = np.log10(prectage)
snli_bert =    [52.5, 78.1, 86.7, 91.0]
snli_mtl =    [82.1, 85.2, 88.4, 91.5]

scitail_bert =    [51.2, 82.2,90.5, 94.3]
scitail_mtl =    [81.9, 88.3, 91.1, 95.7]


plt.ylabel('Accuracy', fontsize=28)
plt.xlabel('Log10(Percentage of Training Data)', fontsize=28)


f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(lprectage, snli_bert, style[2], markersize=14, linewidth=3)
axarr[0].plot(lprectage, snli_mtl, style[1], markersize=14, linewidth=3)
axarr[0].axis([-3.2, 0.1, 50, 92])
axarr[0].legend(['BERT', 'MT-DNN'], loc="lower right", fontsize=28)
axarr[0].ticklabel_format(style='sci' ,axis='y', labelsize=28)
axarr[0].tick_params(labelsize=28)
axarr[0].set_title('(a) SNLI', fontsize=28)

axarr[1].plot(lprectage, scitail_bert, 'c-p', markersize=14, linewidth=3)
axarr[1].plot(lprectage, scitail_mtl, 'r-s', markersize=14, linewidth=3)
axarr[1].axis([-3.2, 0.1, 50, 100])
axarr[1].legend(['BERT', 'MT-DNN'], loc="lower right", fontsize=28)
axarr[1].ticklabel_format(style='sci' ,axis='y', labelsize=28)
axarr[1].tick_params(labelsize=28)
axarr[1].set_title('(b) SciTail', fontsize=28)
for ax in axarr.flat:
    ax.set(xlabel='Log10(Percentage of Training Data)', ylabel='Accuracy')
    ax.xaxis.label.set_size(28)
    ax.yaxis.label.set_size(28)

plt.show()
