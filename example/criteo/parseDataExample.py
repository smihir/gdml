# some_file.py
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '../../tools')
import OneHotEncoderCOO
import CriteoParser as parser

print "Starting hot encoder example"

filepath = 'day_0_10000.txt'
criteoData = parser.ParseCriteoDataFile(filepath)
print criteoData['labels'].shape
print criteoData['intFeatures'].shape
print criteoData['catFeatures'].shape

