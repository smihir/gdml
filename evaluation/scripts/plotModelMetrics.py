from __future__ import division
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

if len(sys.argv) == 1:
    print("enter log file path")
    sys.exit(1)

filepath = sys.argv[1]
accuracy = list()
ce = list()
rmse = list()
with open(filepath) as f:
    for line in f:
        if 'Train-accuracy' in line:
            groups = re.search(r'Train-accuracy=(\d+.\d+)', line)
            accuracy.append(float(groups.group(1)))
        elif 'Train-cross-entropy' in line:
            groups = re.search(r'Train-cross-entropy=(\d+.\d+)', line)
            ce.append(float(groups.group(1)))
        elif 'Train-rmse' in line:
            groups = re.search(r'Train-rmse=(\d+.\d+)', line)
            rmse.append(float(groups.group(1)))


if len(accuracy) != 0:
    x1 = np.arange(len(accuracy))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Model Accuracy on Validation Data vs Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Accuracy")
    rects1 = ax1.plot(x1,accuracy)

if len(ce) != 0:
    x1 = np.arange(len(ce))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Cross Entropy Error vs Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cross Entropy Error")
    rects2 = ax1.plot(x1,ce)

if len(rmse) != 0:
    x1 = np.arange(len(rmse))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Root Mean Square Error vs Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Root Mean Square Error")
    rects3 = ax1.plot(x1,rmse)
plt.show()
