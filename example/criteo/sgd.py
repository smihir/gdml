import find_mxnet
import mxnet as mx
import inspect
# from numpy import *
import numpy as np
import matplotlib
import os, sys,time

def ComputeMSE(y,y_p):
	return ((y - y_p) ** 2).mean(axis=0)

def ComputeCostF(y,y_p):
	numElements = float(y.size)
	return (1/(2*numElements)) * ComputeMSE

def ComputeGrad(y,y_p):
	sumOfDifference = np.sum(y_p - y)
	numElements = float(y.size)
	grad = (1/numElements)*sumOfDifference

	return grad

# kv = mx.kv.create('dist_sync')
kv = mx.kv.create('local')
data_path = 'data/sample_data';
nu = 0.0001

rank = kv.rank
print "rank = {0}".format(rank)


points = np.genfromtxt(data_path+str(rank)+".csv", delimiter=",")
(numRows,numCols) = points.shape
print "numRows = {0}".format(numRows)
print "numCols = {0}".format(numCols)

# x = np.array([points[0:,0:], np.ones(numRows).transpose()])
# x = np.transpose(x)
x = np.concatenate((points[0:,0:1],points[0:,0:1],points[0:,0:1]),1)
y = points[0:,1]

(numPoints,dim) = x.shape  
# print x[:,1:2]

# Init weight vectors
w = np.zeros(dim)
w_id = 1
w_local = mx.nd.array(w)
kv.push(w_id, w_local)

batchSize = 1
currBatch = 0
for i in range(numPoints):	
	# Read mini-batch
	currX = x[currBatch * batchSize:(currBatch+1) * batchSize,:]
	currY = y[currBatch * batchSize:(currBatch+1) * batchSize]
	
	# Read weight vector
	kv.pull(w_id, out = w_local) 
	# Compute hypothesis output
	w_local_np = w_local.asnumpy()
	y_p = np.matmul(currX,w_local_np)
	# Compute gradient
	grad = ComputeGrad(currY,y_p)

	# Update weight vector
	w_local = mx.nd.array(w_local_np - nu * grad * x[i])
	kv.push(w_id,w_local)



# kv.pull(w_id,out = w_local)
# y_p = np.matmul(x,w_local.asnumpy())
# print ComputeMSE(y,y_p)
	









