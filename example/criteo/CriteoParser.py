import sys
import numpy as np
import pandas as pd
import mxnet as mx
sys.path.insert(0, '../../tools')
import OneHotEncoderCOO


# Parse a file with the criteo dataset format:
# <label> <integer feature 1><integer feature 13> <categorical feature 1><categorical feature 26>
def ParseCriteoDataFile(filepath):

	# # Load label and integer features
	intFeatureCols = np.arange(0,14)
	criteoIntFeatures = pd.read_csv(filepath, delimiter="	", usecols=intFeatureCols, header= None)
	criteoIntFeatures = criteoIntFeatures.fillna(value=0)
	# Get numpy matrix
	np_criteoIntFeatures = criteoIntFeatures.as_matrix()


	# Get categorical feature data
	catFeatureCols = np.arange(14,40)
	criteoCatFeatures = pd.read_csv(filepath,delimiter="	",usecols=catFeatureCols,header=None)
	criteoCatFeatures = criteoCatFeatures.fillna(value=0)
	np_criteoCatFeatures = criteoCatFeatures.as_matrix()

	# Decode categorical data into a sparse matrix
	catEncoder = OneHotEncoderCOO.OneHotEncoderCOO()
	catEncoder.fit(np_criteoCatFeatures)
	np_criteoCatFeatures = catEncoder.transform(np_criteoCatFeatures)
    	

	return {'labels': np_criteoIntFeatures[:,0], 
			'features': np.concatenate((np_criteoIntFeatures[:,1:],np_criteoCatFeatures.toarray()),axis=1)}

def SaveCriteoDataAsNDArray(filepath,outfile):
	criteoData = ParseCriteoDataFile(filepath)
	mx.nd.save(outfile, {'labels' : mx.nd.array(criteoData['labels']), 'features' : mx.nd.array(criteoData['features'])})
	return


# Parse a file with the criteo dataset format:
# <label> <integer feature 1><integer feature 13> <categorical feature 1><categorical feature 26>
def ParseCriteoDataFile_old(filepath):

	# # Load label and integer features
	intFeatureCols = np.arange(0,14)
	criteoIntFeatures = pd.read_csv(filepath, delimiter="	", usecols=intFeatureCols, header= None)
	criteoIntFeatures = criteoIntFeatures.fillna(value=0)
	# Get numpy matrix
	np_criteoIntFeatures = criteoIntFeatures.as_matrix()


	# Get categorical feature data
	catFeatureCols = np.arange(14,40)
	criteoCatFeatures = pd.read_csv("day_0_10000.txt",delimiter="	",usecols=catFeatureCols,header=None)
	criteoCatFeatures = criteoCatFeatures.fillna(value=0)
	np_criteoCatFeatures = criteoCatFeatures.as_matrix()

	# Decode categorical data into a sparse matrix
	catEncoder = OneHotEncoderCOO.OneHotEncoderCOO()
	catEncoder.fit(np_criteoCatFeatures)
	np_criteoCatFeatures = catEncoder.transform(np_criteoCatFeatures)
    	

	return {'labels': np_criteoIntFeatures[:,0], 
			'intFeatures': np_criteoIntFeatures[:,1:],
			'catFeatures': np_criteoCatFeatures.toarray()}