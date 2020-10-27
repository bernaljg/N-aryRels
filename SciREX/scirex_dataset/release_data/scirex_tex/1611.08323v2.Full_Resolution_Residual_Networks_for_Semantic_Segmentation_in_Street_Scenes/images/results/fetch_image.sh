#!/bin/bash

#Get the image name

#Copy all the needed images to the folders
cp /work/hermans/cvpr/results/ims/$1 ims/$1
cp /work/hermans/cvpr/results/gt/$1 gt/$1
cp /work/hermans/cvpr/results/lrr/$1 lrr/$1
cp /work/hermans/cvpr/results/dilation/$1 dilation/$1
cp /work/hermans/cvpr/results/frrnd_halfres_no_coarse/$1 ours/$1
cp /work/hermans/cvpr/results/frrnd_halfres_coarse/$1 ours_coarse/$1
