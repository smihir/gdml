import os
from pprint import pprint

import numpy as np

import mxnet as mx
import logging


print "testing global optimizer"

class GlobalOptimizer(mx.optimizer.Optimizer):
    """A GlobalOptimizer for the master parameter server.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of GlobalOptimizer

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
    """
    def __init__(self, **kwargs):
        super(GlobalOptimizer, self).__init__(**kwargs)
        self.momentum = 0

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        if self.momentum == 0.0:
            return None
        else:
            return zeros(weight.shape, weight.context, dtype=weight.dtype)

    def update(self, index, curr_weights, incoming_weights, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        curr_weights : NDArray
            curr_weights ndarray

        incoming_weights : NDArray
            incoming_weights ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(curr_weights, mx.ndarray.NDArray))
        assert(isinstance(incoming_weights, mx.ndarray.NDArray))

        # lr: learning rate, wd: L2 regularization coeff add to all weights
        # lr = self._get_lr(index)
        # wd = self._get_wd(index)
        self._update_count(index)

        # We are not adding a gradient to no need
        # grad = grad * self.rescale_grad
        # if self.clip_gradient is not None:
        #     grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        curr_weights[:] = (curr_weights[:] + incoming_weights[:]) / 2

        # if state:
        #     mom = state
        #     mom[:] *= self.momentum
        #     mom[:] += -lr * (grad + wd * weight)
        #     weight[:] += mom
        # else:
        #     assert self.momentum == 0.0
        #     weight[:] += -lr * (grad + wd * weight)



mx.optimizer.Optimizer.register(GlobalOptimizer)

kv = mx.kv.create('local')
kv.set_optimizer(GlobalOptimizer())

num_weights = 10

w = np.arange(num_weights)
shape = w.shape
print "Initial w: {0}".format(w)

kv.init(0, mx.nd.array(w))
kv.init(1, mx.nd.array(w * 2))
a0 = mx.nd.zeros(shape)
a1 = mx.nd.zeros(shape)
kv.pull(0, out = a0)
kv.pull(1, out = a1)
print "Pulled w0: {0}".format(a0.asnumpy())
print "Pulled w1: {0}".format(a1.asnumpy())

kv.push(0, a0 * 10)
kv.push(1, a1 * 10)
kv.pull(0, out = a0) # pull out the value
kv.pull(1, out = a1) # pull out the value
print "Pulled w0*10: {0}".format(a0.asnumpy())
print "Pulled w1*10: {0}".format(a1.asnumpy())
