# some_file.py
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '../tools')
import OneHotEncoderCOO

print "Starting hot encoder example"
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
click_logs_intft = pd.read_csv("data/day_0_10000.txt",delimiter="	",usecols=use_cols_intft,header= None)
click_logs_intft = click_logs_intft.fillna(value=0)
# click_logs_intft = np.nan_to_num(click_logs_intft)
print click_logs_intft.loc[[0]]

use_cols_cat = np.arange(14,40)
print use_cols_cat
click_logs_cat = pd.read_csv("data/day_0_10000.txt",delimiter="	",usecols=use_cols_cat,header=None)
click_logs_cat = click_logs_cat.fillna(value=0)
np_click_logs_cat = click_logs_cat.as_matrix()

print click_logs_cat.loc[[0]]

cat_encoder = OneHotEncoderCOO.OneHotEncoderCOO()
# print click_logs_cat.as_matrix()[0]
cat_encoder.fit(np_click_logs_cat)
cat_data = cat_encoder.transform(np_click_logs_cat)
print cat_data.shape
    	