#!/usr/bin/python
import os
import cv2

import os
from fnmatch import fnmatch

pattern = "*.png"

ims = []
targets = []

for path, subdirs, files in os.walk('.'):
    for name in files:
        if fnmatch(name, pattern):
            if name[:5] != 'small':
                ims.append(os.path.join(path, name))

                target_name = name
                # Only compress the image if it's too big
                if os.path.getsize(os.path.join(path, name)) > 150000:
                    name = ".".join(name.split(".")[:-1]) + ".jpg"

                targets.append(os.path.join(path, 'small_'+name))


for source, target in zip(ims, targets):
    print(source)
    #os.remove(target)
    inp = cv2.imread(source)
    #size = (inp.shape[1]//2, inp.shape[0]//2)
    #out = cv2.resize(inp, size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(target, inp, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
