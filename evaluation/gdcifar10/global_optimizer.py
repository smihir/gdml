import mxnet as mx
import logging

class GlobalOptimizer(mx.optimizer.Optimizer):
    """A GlobalOptimizer for the master parameter server.
    """
    def __init__(self, **kwargs):
        super(GlobalOptimizer, self).__init__(**kwargs)

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        """
        return None

    def update(self, index, curr_weights, incoming_weights, state):
        """Update the parameters.
        """
        assert(isinstance(curr_weights, mx.ndarray.NDArray))
        assert(isinstance(incoming_weights, mx.ndarray.NDArray))

        self._update_count(index)

        curr_weights[:] = (curr_weights[:] + incoming_weights[:]) / 2
