# A script to plot training loss and validation miou/jacc

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def getKey(item):
    return item[0]

def main(params):

    losses = json.load(open(params['loss_file'], 'r'))
    train_loss = losses['loss_history']
    val_miou = losses['val_miou_history']
    val_jacc = losses['val_jacc_history']

    # plot training loss
    l1 = []
    for i,lossT in train_loss.iteritems():
        l1.append([int(i),lossT])
    l1 = sorted(l1, key=getKey) #sorted the list
 
    ave_l1 = []
    t1 = []
  
    ave_point_in = 2000
    point_interval = 50
    for j in range(0,len(l1)*point_interval/ave_point_in):
        t1.append(j*ave_point_in)
        temp = 0
        for k in range(0,ave_point_in/point_interval):
            temp += l1[j*ave_point_in/point_interval+k][1]
        ave_l1.append(temp*point_interval/ave_point_in)
    plt.plot(t1,ave_l1,'g^', label='training loss')
    

    # plot validation miou/jacc
    t1 = []
    vl1 = []
    for i,lossT in val_miou.iteritems():
        t1.append(int(i))
        vl1.append(lossT)
    plt.figure(2)
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    ax = plt.axes()
    ax.yaxis.grid()
    plt.plot(t1,vl1,'bo', label='validation miou')
    plt.legend(loc='upper right', numpoints=1)

    t1 = []
    va1 = []
    for i,jaccT in val_jacc.iteritems():
         t1.append(int(i))
         va1.append(jaccT)
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('jacc', fontsize=14)
    ax = plt.axes()
    ax.yaxis.grid()
    plt.plot(t1,va1,'go', label='validation jacc')
    plt.legend(loc='lower right', numpoints=1)
    plt.show()
    


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--loss_file', required=True, help='input loss json file')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)
