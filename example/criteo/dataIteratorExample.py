import sys
import numpy as np
import pandas as pd
import mxnet as mx

import CriteoParser as parser

filepath = 'day_0_10000.txt'

# batchSize = 100
# trainSize = 9000

# criteoData = parser.ParseCriteoDataFile(filepath)
# print criteoData['labels'].shape
# print type(criteoData['labels'])
# print criteoData['features'].shape
# print type(criteoData['features'])

# X_train = criteoData['features'][0:trainSize,:]
# X_test = criteoData['features'][trainSize:,:]
# Y_train = criteoData['labels'][0:trainSize]
# Y_test = criteoData['labels'][trainSize:]

# print X_train.shape
# print X_test.shape

data_train = mx.io.CSVIter(data_csv="./data/data_3.csv", data_shape=(3,13,1),
                           label_csv="./data/labels_3.csv", label_shape=(3,),
                           batch_size=1)
