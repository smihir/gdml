import tensorflow as tf
import numpy as np
import gc
import os
import sys

sys.path.append("../") # Append for io_utils
from io_utils import *

# Get prediction tensor to evaluate model accuracy
# Input: 
#       * w - Model weights (tf variable).
#       * t_index_batch - Tensor that gets the indeces of the model that will be needed for a given set of examples.
#       * t_example_batch - Tensor that reads in a batch of examples.
#       * batch_size - Size of batch.
def getValidationTensor(w, t_example_batch, t_index_batch, batch_size ):
    
    t_w_active = tf.gather(w, indices=t_index_batch.values,name="w_active")
    t_w_active = tf.sparse_to_dense(t_index_batch.indices, t_index_batch.shape, tf.squeeze(t_w_active),  default_value=0, validate_indices=True, name="w_active")
    t_f_active = tf.sparse_to_dense(t_index_batch.indices, t_index_batch.shape, t_example_batch.values,  default_value=0, validate_indices=True, name="f_active")
    t_predictions = tf.diag_part(tf.matmul(t_w_active,t_f_active,transpose_b=True))
    t_predictions = tf.reshape(t_predictions,shape=[batch_size,1])

    return t_predictions


def getGradientTensor(w, t_label_batch, t_index_batch,t_features_batch, learning_rate, batch_size, num_features ):

    t_w_active = tf.gather(w, indices=t_index_batch.values,name="w_active")
    # Shape for reshape  NEEDS to be an int32 for some reason (known TF issue)
    # t_shape = tf.cast(t_index_batch.shape, tf.int32)    
    
    # Make a small dense matrix for each sparse w and active features. 
    t_w_local = tf.sparse_to_dense(t_index_batch.indices, t_index_batch.shape, tf.squeeze(t_w_active),  default_value=0, validate_indices=True, name="w_local")
    t_f_active = tf.sparse_to_dense(t_index_batch.indices, t_index_batch.shape, t_features_batch.values,  default_value=0, validate_indices=True, name="f_active")
    # Returns the diagonal elements of the matrix multiplication which is what we are interested in.
    t_dot = tf.diag_part(tf.matmul(t_w_local,t_f_active,transpose_b=True))
    t_dot = tf.reshape(t_dot,shape=[batch_size,1])
    t_sigmoid = tf.sigmoid(t_label_batch * t_dot)

    t_loss = -1 * tf.log(t_sigmoid)
    t_avg_loss = tf.reduce_mean(t_loss)
    t_grad = (t_label_batch * (t_sigmoid - 1)) # Vector of size batch_size x 1
    # The following multiplication broadcast each row of t_grad to each row of the sparse features.
    # For instance the first row of t_grad will multiply every element of the first example in the features batch
    t_grad_local = t_grad * t_features_batch
    t_grad_local = tf.scalar_mul(learning_rate, t_grad_local) 
    
    # Prepare sparse gradients to send back for batch
    # We have to reshape to a 1D vector.
    # # Place the real index values as the indeces
    t_grad_local = tf.SparseTensor(indices=tf.expand_dims(t_index_batch.values, 1), values=t_grad_local.values, shape=[num_features])/batch_size

    # Combine all gradients that have the same indices
    # Get the unique indices, and their positions in the array
    (t_unique_indices, t_idx) = tf.unique(tf.squeeze(t_grad_local.indices))
    t_unique_indices_shape = tf.shape(t_unique_indices)
    # Use the positions of the unique values as the segment ids to get the unique values
    t_values = tf.unsorted_segment_sum(tf.squeeze(t_grad_local.values), t_idx, t_unique_indices_shape[0])

    t_grad_local = tf.SparseTensor(indices=tf.expand_dims(t_unique_indices, 1), values=t_values, shape=[num_features])
    t_grad_local_sq = tf.square(t_grad_local)
    return t_grad_local


def getAccuracy(num_correct_predictions, num_examples_test):
    accuracy = float(num_correct_predictions) / float(num_examples_test)
    error = (float(num_examples_test) - float(num_correct_predictions)) / float(num_examples_test)
    print "Accuracy: {0}, Error: {1}".format(accuracy, error)
    return (accuracy,error)

def getPrecisionAndRecall(labels, predictions):

    predictions_sign = np.sign(predictions)
    tp = np.logical_and(labels == 1, predictions_sign == 1)
    tn = np.logical_and(labels == -1, predictions_sign == -1)
    fp = np.logical_and(labels == -1, predictions_sign == 1)
    fn = np.logical_and(labels == 1, predictions_sign == -1)
    num_tp = np.sum(tp)
    num_tn = np.sum(tn)
    num_fp = np.sum(fp)
    num_fn = np.sum(fn)

    return (num_tp,num_tn,num_fp,num_fn)

# Get tensor that calculates the norm of a sparse tensor
def getNormTensor(t_grad):

    t_grad_sq = tf.square(t_grad)
    t_grad_sq_sum = tf.sparse_reduce_sum(t_grad_sq)
    t_norm = tf.sqrt(t_grad_sq_sum)

    return t_norm
