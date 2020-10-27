#!/usr/bin/python
import os
import numpy as np
import cv2

#Get all files
pred_files = [f for f in os.listdir('.') if f[:4] == 'pred' and (f[-9:] == '16384.png' or f[-9:] == '32768.png' or f[-9:] == '65536.png') ]


for f in pred_files:
    print(f)
    source = f
    target = f.replace('.png', '_white.png')
    inp = cv2.imread(source)
    inp = np.rollaxis(inp, 2)
    out = inp.copy()
    out[0][np.sum(inp, 0) == 0] = 255
    out[1][np.sum(inp, 0) == 0] = 255
    out[2][np.sum(inp, 0) == 0] = 255
    out = np.rollaxis(np.rollaxis(out,2),2)
    cv2.imwrite(target, out)