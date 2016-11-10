import find_mxnet
import mxnet as mx
import inspect
from numpy import *
import os, sys,time


rate = 0.0001

points = genfromtxt("data/sample_data.csv", delimiter=",")


shape = (3,4)
kv = mx.kv.create('dist_sync')
kv.init(3, mx.nd.ones(shape)*1)
kv.init(4, mx.nd.ones(shape)*1)
a = mx.nd.ones(shape)

if(kv.rank == 0):
	l = list()
#else:
#kv.pull(3,out = a)
#print a.asnumpy()

a = a * 8
kv.push(3, a)
print "After push"
b = mx.nd.zeros(shape)
time.sleep(1)
kv.pull(3, out = b)
print a.asnumpy()

time.sleep(1)
c = a * 8
kv.push(4, c)
print "After push"
d = mx.nd.zeros(shape)
time.sleep(1)
kv.pull(4, out = d)
print a.asnumpy()

print "rank = {0}".format(kv.rank)

# kv.init('key1', mx.nd.ones((2,2)))
# xdata = points[:,0]
# ydata = points[:,1]
# numPoints = ydata.size
# print numPoints
# mx_xdata = mx.nd.ones(1) * xdata[0]
# mx_mdata = mx.nd.zeros(1)
# mx_bdata = mx.nd.zeros(1)

# grad_b = mx.nd.empty(1) 
# grad_m = mx.nd.empty(1) 

# executor = y.bind(ctx=mx.cpu(),
# 				  args={'x':mx_xdata,'m':mx_mdata,'b':mx_bdata},
# 				  args_grad={'b':grad_b,'m':grad_m})

# executor.forward()
# print executor.outputs[0].asnumpy()

# grad_y = (ydata[0] - executor.outputs[0]) * rate
# print grad_y.asnumpy()

# executor.backward(out_grads=grad_y)

# print executor.outputs[0].asnumpy()
