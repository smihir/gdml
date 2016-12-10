
import tensorflow as tf

def read_np_csv(filename_queue):
    reader = tf.TextLineReader()
    key, record_string = reader.read(filename_queue)

    # dot needed to make sure they take a float value
    record_defaults = [[0.], [0.], [0.]]
    (label, feature0, feature1) = tf.decode_csv(record_string, record_defaults=record_defaults)
    features = tf.pack([feature0,feature1])
    # label = tf.to_int32(label,name='ToInt32')

    return label, features

def input_np_csv_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    label, features = read_np_csv(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly         sample
    #   from -- bigger means better shuffling but slower start up and     more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) *     batch_size
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    (label_batch, features_batch) = tf.train.shuffle_batch([label, features], batch_size=batch_size,                                                capacity=capacity,min_after_dequeue=min_after_dequeue)
    # We need to reshape to column vector
    label_batch = tf.reshape(label_batch, [batch_size, 1])

    return label_batch, features_batch

# Returns label and dense feature vector
def read_dense_tfrecord_criteo(filename_queue, num_features):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    example = tf.parse_single_example(serialized_example,
        features={
            'label': tf.FixedLenFeature([1], dtype=tf.int64),
            'index' : tf.VarLenFeature(dtype=tf.int64),
            'value' : tf.VarLenFeature(dtype=tf.float32),
        })    
    
    label = tf.to_float(example['label']) # Example label
    index = example['index'] # sp indeces
    features = example['value'] # Feature values

    # since we parsed a VarLenFeatures, they are returned as spTensors.
    # To run operations on then, we first convert them to dense Tensors as below.
    dense_features = tf.sparse_to_dense(tf.sp_tensor_to_dense(index), [num_features,],
                                       tf.sp_tensor_to_dense(features))

    return label, dense_features

# Returns label and dense feature vector
def read_sp_tfrecord_criteo(filename_queue, num_features):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    example = tf.parse_single_example(serialized_example,
        features={
            'label': tf.FixedLenFeature([1], dtype=tf.int64),
            'index' : tf.VarLenFeature(dtype=tf.int64),
            'value' : tf.VarLenFeature(dtype=tf.float32),
        })    
    
    label = tf.to_float(example['label']) # Example label
    sp_index = example['index'] # sp indeces
    sp_features = example['value'] # Feature values

    return label, sp_index, sp_features

# Read in a batch of sp criteo examples and return their dense versions
def input_dense_tfrecord_criteo_pipeline(filenames, batch_size, num_criteo_features, num_epochs=None):
    
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    label, dense_features = read_dense_tfrecord_criteo(filename_queue,num_criteo_features)
    
    # min_after_dequeue defines how big a buffer we will randomly         sample
    #   from -- bigger means better shuffling but slower start up and     more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) *     batch_size
    min_after_dequeue = 5
    capacity = min_after_dequeue + 3 * batch_size
    (label_batch,dense_features_batch) = tf.train.shuffle_batch([label, dense_features], 
        batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    
    # We need to reshape to column vector
    label_batch = tf.reshape(label_batch, [batch_size, 1])

    return label_batch, dense_features_batch

# Read in a batch of sp criteo examples
def input_sp_tfrecord_criteo_pipeline(filenames, batch_size, num_criteo_features, num_epochs=None):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    (label, sp_index, sp_features) = read_sp_tfrecord_criteo(filename_queue,num_criteo_features)

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    (label_batch,sp_index_batch,sp_features_batch) = tf.train.shuffle_batch([label, sp_index, sp_features], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    
    # We need to reshape to column vector
    label_batch = tf.reshape(label_batch, [batch_size, 1])

    return label_batch, sp_index_batch, sp_features_batch

    