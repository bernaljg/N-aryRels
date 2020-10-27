import glob
import os

files = glob.glob('*.tar.gz')

for file in files:
    dir = file.replace('.tar.gz','')

    os.system('mkdir {}'.format(dir))
    os.system('tar -xzvf {} -C {}/'.format(file, dir))
