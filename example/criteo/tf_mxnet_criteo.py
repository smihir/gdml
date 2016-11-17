import mxnet as mx
from mxnet import io
import tensorflow as tf
import numpy as np
import logging


class CriteoTFRecordIter(io.DataIter):
    def __init__(self, data=None, label=None, batch_size=1, shuffle=False, last_batch_handle='pad'):
        # pylint: disable=W0201
        super(CriteoTFRecordIter, self).__init__()

        self.num_data = 10000
        self.batch_size = 1
        self.cursor = 0

        # number of features in the criteo dataset after one-hot encoding
        self.num_features = 33762578

        # Here, we will show how to include reader operators in the TensorFlow graph.
        # These operators take as input list of filenames from which they read data.
        # On every invocation of the operator, some records are read and passed to the
        # downstream vertices as Tensors
        self.g = tf.Graph()
        with self.g.as_default():
            # We first define a filename queue comprising 5 files.
            input_data = list()
            if isinstance(data, str):
                input_data.append(data)
            #filename_queue = tf.train.string_input_producer([
            #    "./data/criteo-tfr-tiny/tfrecords00",
            #    #"./data/criteo-tfr-tiny/tfrecords01",
            #    #"./data/criteo-tfr-tiny/tfrecords02",
            #    #"./data/criteo-tfr-tiny/tfrecords03",
            #    #"./data/criteo-tfr-tiny/tfrecords04",
            #], num_epochs=None)
            filename_queue = tf.train.string_input_producer(input_data, num_epochs=None)

            # TFRecordReader creates an operator in the graph that reads data from queue
            reader = tf.TFRecordReader()

            # Include a read operator with the filenae queue to use. The output is a string
            # Tensor called serialized_example
            _, serialized_example = reader.read(filename_queue)

            # The string tensors is essentially a Protobuf serialized string. With the
            # following fields: label, index, value. We provide the protobuf fields we are
            # interested in to parse the data. Note, feature here is a dict of tensors
            features = tf.parse_single_example(serialized_example,
                                            features={
                                                'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                                'index' : tf.VarLenFeature(dtype=tf.int64),
                                                'value' : tf.VarLenFeature(dtype=tf.float32),
                                            }
                                            )
            self.label = features['label']
            index = features['index']
            value = features['value']

            # These print statements are there for you see the type of the following
            # variables
            #print self.label
            #print index
            #print value

            # since we parsed a VarLenFeatures, they are returned as SparseTensors.
            # To run operations on then, we first convert them to dense Tensors as below.
            self.dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
                                        [self.num_features,],
            #                               tf.constant([33762578, 1], dtype=tf.int64),
                                        tf.sparse_tensor_to_dense(value))

            # as usual we create a session.
            self.sess = tf.Session()
            self.sess.run(tf.initialize_all_variables())

            #print type(dense_feature.eval())
            # this is new command and is used to initialize the queue based readers.
            # Effectively, it spins up separate threads to read from the files
            tf.train.start_queue_runners(sess=self.sess)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        output =  self.sess.run(self.dense_feature)
        output = np.transpose(output.reshape(len(output), 1))
        return [('data', output.shape)]
        #print [('data', tuple([1] + [self.num_features]))]
        #return [('data', tuple([1] + [self.num_features]))]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        l = self.sess.run(self.label)
        return [('softmax_label', l.shape)]
        #print [('softmax_label', tuple([] + [self.batch_size]))]
        #return [('softmax_label', tuple([self.batch_size] + []))]
        #return [('softmax_label', tuple([] + [self.batch_size]))]
        #return [('softmax_label', tuple([33762578] + []))]


    def hard_reset(self):
        """Igore roll over data and set to start"""
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            return io.DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _getdata(self):
        """Load data from underlying arrays, internal use only"""
        #assert(self.cursor < self.num_data), "DataIter needs reset."
        output =  self.sess.run(self.dense_feature)
        #print type(output)
        output = np.transpose(output.reshape(len(output), 1))
        #print output
        #print output.shape
        self.cursor += 1
        mx_o = mx.nd.array(output)
        #print mx_o.shape
        return [mx_o]

    def _getlabel(self):
        o = list()
        l = self.sess.run(self.label)
        #print l
        #print l.shape
        mx_l = mx.nd.array(l)
        #print mx_l.shape
        #print type(mx_l.shape)
        return [mx_l]

    def getdata(self):
        return self._getdata()

    def getlabel(self):
        return self._getlabel()

    def getpad(self):
        return None


if __name__ == "__main__":
    train = CriteoTFRecordIter('./data/criteo-tfr-tiny/tfrecords00')
    test = CriteoTFRecordIter('./data/criteo-tfr-tiny/tfrecords01')
    data = mx.symbol.Variable('data')
    #flat = mx.symbol.Flatten(name='flat', data=data)
    fc1  = mx.symbol.FullyConnected(data = data, num_hidden=1)
    mlp  = mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')
    #mlp  = mx.symbol.LogisticRegressionOutput(data = fc3, name = 'softmax')
    #mlp = mx.symbol.SoftmaxOutput(data, name='softmax')

    #data = mx.symbol.Variable('data')
    #fc1  = mx.symbol.FullyConnected(data = data, num_hidden=64)
    #act1 = mx.symbol.Activation(data = fc1, act_type="relu")
    #fc2  = mx.symbol.FullyConnected(data = act1, num_hidden = 32)
    #act2 = mx.symbol.Activation(data = fc2, act_type="relu")
    #fc3  = mx.symbol.FullyConnected(data = act2, num_hidden=2)
    #mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    
    kv = mx.kvstore.create('local')
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)


    model = mx.model.FeedForward(symbol=mlp, num_epoch=1, learning_rate=0.01)

    eval_metrics = ['accuracy']
    batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(1, 10))


    model.fit(X                = train,
            kvstore            = kv,
            eval_metric        = eval_metrics,
            batch_end_callback = batch_end_callback)

    #model.fit(X=train)
    print "score"
    print model.score(X=test)
