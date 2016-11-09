# some_file.py
import sys
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