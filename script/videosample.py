#!/usr/bin/env python
# [Description] A script to extract exact number of frames from a sequence video (including data augmentation)
# [Author] chaonan99
# [Date] 2016/07/17/
# [Last Modified] 2018/02/15 by Luowei Zhou
# [Email] luozhou@umich.edu

import subprocess as sp
import re
import optparse
import os
import math
import numpy as np
from PIL import Image
# import Image
import sys

parser = optparse.OptionParser()
parser.add_option('-i', '--input', dest='infile', help='input video file name')
parser.add_option('-o', '--output_folder', dest='outfolder', help='output folder name')
parser.add_option('-n', '--number', dest='fnum', help='number of frames', default=500, type=int)
parser.add_option('-r', '--repeat', dest='repeat', help='number of frame batches or temporal data augmentation (default=1; all=-1)', default=1, type=int)

(opts, args) = parser.parse_args()

def getInfo(filename):
    result = sp.Popen(["ffmpeg", "-i", filename],
        stdout = sp.PIPE, stderr = sp.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x or "yuv" in x]

def decode(filename):
    sp.call(["ffmpeg", "-i", filename, "temp.yuv"],stdout = sp.PIPE, stderr = sp.STDOUT)

def yuv_import(filename,dim):
    data = np.fromfile(filename,dtype=np.uint8).reshape((-1,6,dim[0]*dim[1]/4))
    y = data[:,0:4,:].reshape(-1,dim[1],dim[0])
    u = data[:,4,:].reshape(-1,dim[1]/2,dim[0]/2)
    v = data[:,5,:].reshape(-1,dim[1]/2,dim[0]/2)
    return y,u,v

def touch(indir):
    if not os.path.isdir(indir):
        os.mkdir(indir)

infoStr = getInfo(opts.infile)
durarr = re.findall(r'Duration:\s\d{2}:\d{2}:\d+.\d+',infoStr[0])
findstr = np.asarray(re.findall(r'(\d+)x(\d+) (?=\[)', infoStr[1]))
if len(findstr) == 0:
    findstr = np.asarray(re.findall(r'yuv420p.*? (\d+)x(\d+)', infoStr[1]))
if len(findstr) == 0:
    print("[ERROR] Size not found: "+ opts.infile)

else:
    decode(opts.infile)
    touch(opts.outfolder)
    dim = findstr[0].astype(np.int32)
    o = np.ones((2,2)).astype(np.uint8)

    # temp.yuv might be huge, sometimes you need to mannally delete it
    y,u,v = yuv_import(r"temp.yuv",dim)

    total_frames = y.shape[0]
    # jump = int(math.floor(total_frames/opts.fnum))+1
    jump = int(math.ceil(total_frames/opts.fnum))

    if opts.repeat < 0:
        opts.repeat = jump

    print("[INFO] " + opts.infile + " " + str(durarr[0]) + " " + "Frame size: %dx%d Total frames: %d"%(dim[0],dim[1],total_frames))
    if jump < opts.repeat:
        print("[Warning] Total frames is less than batch_size * number of batches. Output numver of batches may be less than %d"
            %(opts.repeat))
        opts.repeat = jump

    skip = int(math.floor(jump/opts.repeat))
    for i in range(opts.repeat):
        batch_folder = opts.outfolder + "/%04d/"%(i+1)
        touch(batch_folder)
        for j in range(opts.fnum):
            ind = min(j*jump + i*skip, total_frames - 1)
            im = Image.fromarray(np.dstack((y[ind,:,:], np.kron(u[ind,:,:],o), np.kron(v[ind,:,:],o))),mode="YCbCr")
            im.thumbnail([400,225], Image.ANTIALIAS)
            im.save(batch_folder + "%04d"%(j+1) + ".jpg")

    os.remove("temp.yuv")
