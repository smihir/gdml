import find_mxnet
import mxnet as mx
import inspect
from numpy import *

rate = 0.0001

points = genfromtxt("data/sample_data.csv", delimiter=",")

m = mx.sym.Variable('m')
b = mx.sym.Variable('b')
x = mx.sym.Variable('x')

y = (m * x) + b


xdata = points[:,0]
ydata = points[:,1]
numPoints = ydata.size
print numPoints
mx_xdata = mx.nd.ones(1) * xdata[0]
mx_mdata = mx.nd.zeros(1)
mx_bdata = mx.nd.zeros(1)

grad_b = mx.nd.empty(1) 
grad_m = mx.nd.empty(1) 

executor = y.bind(ctx=mx.cpu(),
				  args={'x':mx_xdata,'m':mx_mdata,'b':mx_bdata},
				  args_grad={'b':grad_b,'m':grad_m})

executor.forward()
print executor.outputs[0].asnumpy()

grad_y = (ydata[0] - executor.outputs[0]) * rate
print grad_y.asnumpy()

executor.backward(out_grads=grad_y)

print executor.outputs[0].asnumpy()
