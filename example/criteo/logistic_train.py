# import mxnet as mx
import numpy as np
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../"))
import find_mxnet
import mxnet as mx

import logging

# kvstore
kv = mx.kvstore.create('local')

head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

logging.info("start log")

datapath = "data/data.mx"

dataset = mx.nd.load(datapath)

trainSize = 1000
batchSize = 1

train = mx.io.NDArrayIter(data=dataset['features'][0:trainSize],label=dataset['labels'][0:trainSize],batch_size=batchSize,shuffle=True)
test = mx.io.NDArrayIter(data=dataset['features'][trainSize:],label=dataset['labels'][trainSize:],batch_size=batchSize,shuffle=True)

# # configure a two layer neuralnetwork
#data = mx.symbol.Variable('data')
#fc1  = mx.symbol.FullyConnected(data = data, num_hidden=64)
#act1 = mx.symbol.Activation(data = fc1, act_type="relu")
#fc2  = mx.symbol.FullyConnected(data = act1, num_hidden = 32)
#act2 = mx.symbol.Activation(data = fc2, act_type="relu")
#fc3  = mx.symbol.FullyConnected(data = act2, num_hidden=2)
#mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
data = mx.symbol.Variable('data')
#flat = mx.symbol.Flatten(name='flat', data=data)
fc1  = mx.symbol.FullyConnected(data = data, num_hidden=1)
mlp  = mx.symbol.LogisticRegressionOutput(data = fc1, name = 'softmax')

# create a model
model = mx.model.FeedForward(symbol = mlp,
                             num_epoch = 5,
                             learning_rate = .1)

eval_metrics = ['accuracy']
batch_end_callback = []
calcFrequency = 50
batch_end_callback.append(mx.callback.Speedometer(batchSize, calcFrequency))


print model.fit(X                  = train,
        kvstore = kv,
        eval_metric        = eval_metrics,
        batch_end_callback = batch_end_callback)

out = model.predict(X=test)

for i in out:
    print i

