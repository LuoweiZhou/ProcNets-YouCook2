#!/usr/bin/env python
# A script to retrieve video info (duration, number of frames)

import subprocess as sp
import re
import optparse
import os
import math
import numpy as np
from PIL import Image
# import Image
import sys
import csv
import time

parser = optparse.OptionParser()
parser.add_option('-i', '--input', dest='infolder', help='input video file name')
parser.add_option('-s', '--split', dest='split', help='split')

(opts, args) = parser.parse_args()

def getInfo(filename):
    result = sp.Popen(["ffmpeg", "-i", filename],
        stdout = sp.PIPE, stderr = sp.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x or "yuv" in x]

def getLength(filename):
    result = sp.Popen(["ffprobe", filename],
        stdout = sp.PIPE, stderr = sp.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]

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

dur_frame_inf = {'vid':[], 'dur':[], 'tf':[]}
for rec in os.listdir(opts.infolder):
  rec_path = os.path.join(opts.infolder, rec)
  for vid in os.listdir(rec_path):
    vid_file = os.path.join(rec_path, vid)
    print vid_file
    infoStr = getInfo(vid_file)
    durarr = re.findall(r'Duration:\s\d{2}:\d{2}:\d+.\d+',infoStr[0])
    findstr = np.asarray(re.findall(r'(\d+)x(\d+) (?=\[)', infoStr[1]))
    if len(findstr) == 0:
        findstr = np.asarray(re.findall(r'yuv420p.*? (\d+)x(\d+)', infoStr[1]))
    if len(findstr) == 0:
        print("[ERROR] Size not found: "+ vid_file)
    else:
        decode(vid_file)
        dim = findstr[0].astype(np.int32)
        o = np.ones((2,2)).astype(np.uint8)
    
        y,u,v = yuv_import(r"temp.yuv",dim)
        total_frames = y.shape[0]
        duration = getLength(vid_file)
        duration_in_second = float(duration[0][15:17])*60+float(duration[0][18:23])
        video_name = vid_file

        dur_frame_inf['vid'].append(vid[:11])
        dur_frame_inf['dur'].append(duration_in_second)
        dur_frame_inf['tf'].append(total_frames)    
        print vid, duration_in_second, total_frames

        os.remove("temp.yuv")

# write to csv file
ofile = open("youcookii_"+opts.split+"_duration.csv", 'wb')
writer = csv.writer(ofile, delimiter=',')
# each column is a video... transpose it if you feel more comfortable the other way
writer.writerow(dur_frame_inf['vid'])
writer.writerow(dur_frame_inf['dur'])
writer.writerow(dur_frame_inf['tf'])
ofile.close()
