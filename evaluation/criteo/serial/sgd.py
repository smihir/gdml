"""
Group 28 
Single machine SGD for Binary logistic regression.
    - Mini-batch vanilla stochastic gradient descent for training
    -
"""

import tensorflow as tf
import numpy as np
import gc
import os
import sys
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score

sys.path.append("../") # Append for io_utils
from io_utils import *
from utils import *


# GRADIENT DESCENT PARAMETERS
learning_rate = 0.1 # eta parameter
training_epochs = 1 # Number of passes through the whole dataset
batch_size = 200 # Number of examples read at once and calc gradient at once
batch_size_test = 1000 # Number of examples read at once
batch_size_validation = 10000 # Number of examples read at once


# DATA_DIR="/home/ubuntu/workspace/criteo_data/"
DATA_DIR = "/home/ubuntu/nfs/criteo_data/criteo-tfr/"

# Files to read
filenames = [DATA_DIR + "tfrecords00"]
filenames_test = [DATA_DIR + "tfrecords22-tiny"] # subset of validation dataset
filenames_validation = [DATA_DIR + "tfrecords22"]

# filenames = ["../data/tfrecords_train"]
# filenames_test = ["../data/tfrecords22-tiny"] # subset of validation dataset
# filenames_validation = ["../data/tfrecords22"]
num_features = 33762578 # Total number of features after one hot encoding

# I/O PARAMETERES
num_examples = 2000000
num_examples_test = 10000
num_examples_validation = 2000000
num_batches = num_examples / batch_size # Number of training batches
# num_batches = 1 # Number of training batches
num_batches_test = num_examples_test / batch_size_test # Number of test batches
num_batches_validation = num_examples_validation / batch_size_validation # Number of validation bataches

num_steps_per_test = 10000

description = str(batch_size) + "_" + str(learning_rate)
out_filename = "output/Serial_Results_" + description

# # Start graph
g = tf.Graph()

with g.as_default():
    tf.set_random_seed(1024)

    # Measurement metrics
    gradient_norm = []
    test_accuracy = []
    test_error = []
    validation_accuracy = 0
    validation_error = 0  
    validation_precision = 0  
    validation_recall = 0  


    # SETUP VARIABLES
    w = tf.Variable(tf.random_normal([num_features, 1], stddev=0.35), name="model")
    accuracy = tf.Variable(tf.zeros([1]),name="accuracy")
    num_correct_predictions = 0.

    # SETUP INPUT TENSORS
    # Get Input batch tensors operators for training and testing data
    (t_label_batch, t_sp_index_batch, t_sp_features_batch) = input_sp_tfrecord_criteo_pipeline(filenames, batch_size, num_features, num_epochs=None) 
    (t_label_batch_test, t_sp_index_batch_test, t_sp_features_batch_test) = input_sp_tfrecord_criteo_pipeline(filenames_test, batch_size_test, num_features, num_epochs=None) 
    (t_label_batch_validation, t_sp_index_batch_validation, t_sp_features_batch_validation) = input_sp_tfrecord_criteo_pipeline(filenames_validation, batch_size_validation, num_features, num_epochs=None) 

    # SETUP TRAINING TENSORS
    t_sp_grad_local = getGradientTensor(w, t_label_batch, t_sp_index_batch, t_sp_features_batch, learning_rate, batch_size, num_features)
    t_norm = getNormTensor(t_sp_grad_local)

    # SETUP UPDATE TENSOR
    t_w_update = tf.scatter_sub(w, tf.squeeze(t_sp_grad_local.indices), tf.expand_dims(t_sp_grad_local.values, 1))

    # SETUP TESTING TENSORS: Every 10000 steps and with small dataset --> Takes ~10 seconds
    (t_predictions_test,t_predictions_test_p) = getValidationTensor(w, t_sp_features_batch_test, t_sp_index_batch_test, batch_size_test)
    # SETUP Validation TENSORS: Every 100000 and with full validation dataset --> Takes ~5 mins
    (t_predictions_validation,t_predictions_validation_p) = getValidationTensor(w, t_sp_features_batch_validation, t_sp_index_batch_validation, batch_size_validation)


    t_saver = tf.train.Saver({"model": w})
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    coord = tf.train.Coordinator()
    # this is new command and is used to initialize the queue based readers.
    # Effectively, it spins up separate threads to read from the files
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)


    for i in range(0,num_batches):
        (norm, w_update) = sess.run([t_norm,t_w_update])
        num_examples_processed = i*batch_size
        # if((num_examples_processed) % 1000 == 0):
        print "Batch {}, Examples Processed: {}, grad norm: {}".format(i,num_examples_processed, norm)
        gradient_norm.append(norm)

        # Run Testing
        if(((num_examples_processed) % num_steps_per_test) == 0):
            print "Testing..."
            num_tp_all = 0 
            num_tn_all = 0
            num_fp_all = 0
            num_fn_all = 0
            for j in range(0,num_batches_test):
                print "Test Batch {0}, Test Ex Processed {1}".format(j,j*batch_size_test)
                (labels, predictions) = sess.run([t_label_batch_test, t_predictions_test])
                (num_tp,num_tn,num_fp,num_fn) = getPrecisionAndRecall(labels, predictions)
                num_tp_all += num_tp
                num_tn_all += num_tn
                num_fp_all += num_fp
                num_fn_all += num_fn

            accuracy = float(float(num_tp_all + num_tn_all) / float(num_examples_test))
            error = float(float(num_fp_all + num_fn_all) / float(num_examples_test))
            test_accuracy.append(accuracy)
            test_error.append(error)
            print "Batch {}, Accuracy: {}, Error: {}".format(i,accuracy,error)
            print "tp: {}, tn: {}, fp: {}, fn: {}".format(num_tp_all,num_tn_all,num_fp_all,num_fn_all)

    # Run Validation
    print "Validating..."
    # Save current model
    t_saver.save(sess, "output/Serial_Model_"+description+".ckpt")

    num_tp_all = 0 
    num_tn_all = 0
    num_fp_all = 0
    num_fn_all = 0
    labels_all = np.zeros((num_examples_validation,1))
    predictions_all = np.zeros((num_examples_validation,1))
    for j in range(0,num_batches_validation):
        print "Validation Batch {0}, Val Ex Processed {1}".format(j,j*batch_size_validation)
        (labels, predictions,predictions_p) = sess.run([t_label_batch_validation, t_predictions_validation,t_predictions_validation_p])
        print predictions_p
        start_index = j * batch_size_validation
        end_index = ((j+1) * batch_size_validation)
        print start_index
        print end_index
        labels_all[start_index:end_index] = labels
        predictions_all[start_index:end_index] = predictions_p
        (num_tp,num_tn,num_fp,num_fn) = getPrecisionAndRecall(labels, predictions)
        num_tp_all += num_tp
        num_tn_all += num_tn
        num_fp_all += num_fp
        num_fn_all += num_fn    

    e_precision, e_recall, _ = precision_recall_curve( labels_all, predictions_all, pos_label=1 )
    e_fpr, e_tpr, _ = roc_curve( labels_all, predictions_all, pos_label=1 )
    
    accuracy = float(float(num_tp_all + num_tn_all) / float(num_examples_validation))
    error = float(float(num_fp_all + num_fn_all) / float(num_examples_validation))
    precision = float(float(num_tp_all) / float(num_tp_all + num_fp_all))
    recall = float(float(num_tp_all) / float(num_tp_all + num_fn_all))
    validation_accuracy = accuracy
    validation_error = error
    validation_precision = precision
    validation_recall = recall

    # Save validation arrays
    print "Model Accuracy: {}, Error: {}, Precision: {}, Recall: {}".format(accuracy,error,precision,recall)
    print "tp: {}, tn: {}, fp: {}, fn: {}".format(num_tp_all,num_tn_all,num_fp_all,num_fn_all)
    np.savez(out_filename, 
        gradient_norm=gradient_norm,
        test_error=test_error,
        test_accuracy=test_accuracy,
        validation_error=validation_error,
        validation_accuracy=validation_accuracy,
        validation_precision=validation_precision,
        validation_recall=validation_recall,
        e_precision=e_precision,
        e_recall=e_recall,
        e_fpr=e_fpr,
        e_tpr=e_tpr
        )


    coord.request_stop()
    coord.join(threads)
    sess.close()




