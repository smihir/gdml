import zmq
import pickle
import mxnet as mx
import logging
import global_optimizer as opt

def main():

    go = mx.optimizer.Optimizer.register(opt.GlobalOptimizer)

    # ZeroMQ Context
    context = zmq.Context()

    # Define the socket using the "Context"
    sock = context.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5678")

    kv = mx.kvstore.create('dist_async')
    kv.set_optimizer(mx.optimizer.Test())
    kv.set_optimizer(opt.GlobalOptimizer())
    kv_initialized = dict()

    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('zmq_worker starting...')

    while True:
        message = sock.recv()
        index, weight = pickle.loads(message)
        if not index in kv_initialized:
            #logging.info('index {}, init weight size {}'.format(index, weight.shape))
            kv.init(index, weight)
            kv_initialized[index] = True
        else:
            #logging.info('index {}, push weight size {}'.format(index, weight.shape))
            kv.push(index, weight)
        new = mx.nd.zeros(weight.shape)
        kv.pull(index, new)
        #logging.info('new weight size {}'.format(new.shape))
        #logging.info('new weight {}'.format(new.asnumpy()))
        sock.send(pickle.dumps(new))

if __name__ == '__main__':
    main()
