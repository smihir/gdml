# some_file.py
import sys
import numpy as np
sys.path.insert(0, '../tools')

import OneHotEncoderCOO
import numpy as np
import scipy as sp


x = np.array([['a', 'b', 'c'],['a', 'f', 'd'],['c', 'b', 'f']])
print x
encoder = OneHotEncoderCOO.OneHotEncoderCOO()
encoder.fit(x)
print encoder.keymap
print encoder.transform(x)
# print encoder.keymap
# x.shape = (6,1)
# print encoder.transform(x)



# Load label and integer features
use_cols_intft = np.arange(0,14)
click_logs_intft = np.genfromtxt("data/day_0_10000.txt",delimiter="	",usecols=use_cols_intft)
click_logs_intft = np.nan_to_num(click_logs_intft)
print click_logs_intft[0]
print click_logs_intft.shape

use_cols_cat = np.arange(15,40)
print use_cols_cat
click_logs_cat = np.loadtxt("data/day_0_10000.txt",delimiter="	",usecols=use_cols_cat)
print click_logs_cat[0]

