
import tensorflow as tf
import numpy as np
import os
import sys

from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score

sys.path.append("../")
from io_utils import *
from utils import *

num_workers = 5

# GRADIENT DESCENT PARAMETERS
learning_rate = 0.1 # eta parameter
training_epochs = 1 # Number of passes through the whole dataset
batch_size = 200 # Number of examples read at once and calc gradient at once
batch_size_test = 1000 # Number of examples read at once
batch_size_validation = 10000 # Number of examples read at once


DATA_DIR="/home/ubuntu/workspace/criteo_data/"
# Files to read
filenames = [DATA_DIR + "tfrecords00"]
filenames_test = [DATA_DIR + "tfrecords22-tiny"]
filenames_validation = [DATA_DIR + "tfrecords22"]

num_features = 33762578 # Total number of features after one hot encoding

# I/O PARAMETERES
num_examples = 500000
num_examples_test = 10000
num_examples_validation = 2000000
num_batches = num_examples / batch_size # Number of training batches
num_batches_test = num_examples_test / batch_size_test # Number of test batches
num_batches_validation = num_examples_validation / batch_size_validation # Number of validation bataches

num_steps_per_test = 25000

description = str(batch_size) + "_" + str(learning_rate)
out_filename = "output/SyncSGD_Results_" + description

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(1024)

    tf.logging.set_verbosity(tf.logging.INFO)
    # creating a model variable on task 0. This is a process running on node vm-48-1
    with tf.device("/job:worker/task:0"):
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
        curr_norm = tf.Variable(0., name="curr_norm")
        num_correct_predictions = 0.        
        assign_op = None
        # Create Validation Files Tensors - Only done in master
        (t_label_batch_test, t_sp_index_batch_test, t_sp_features_batch_test) = input_sp_tfrecord_criteo_pipeline(filenames_test, batch_size_test, num_features, num_epochs=None) 
        (t_label_batch_validation, t_sp_index_batch_validation, t_sp_features_batch_validation) = input_sp_tfrecord_criteo_pipeline(filenames_validation, batch_size_validation, num_features, num_epochs=None) 



    # creating 5 reader operators to be placed on different operators
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    gradients = []
    norms = []
    for i in range(1, num_workers):
        with tf.device("/job:worker/task:%d" % i):
            fnames = []
            for findex in range(i * num_workers, (i * num_workers) + num_workers):
                if findex > 21:
                    break
                fnames.append(DATA_DIR + "tfrecords%02d" % findex)

            print('node[{}]: process files: {}'.format(i, fnames))

            (t_label_batch, t_sp_index_batch, t_sp_features_batch) = input_sp_tfrecord_criteo_pipeline(fnames, batch_size, num_features, num_epochs=None)

            # # SETUP TRAINING TENSORS
            t_sp_grad_local = getGradientTensor(w, t_label_batch, t_sp_index_batch, t_sp_features_batch, learning_rate, batch_size, num_features)
            t_sp_grad_local = t_sp_grad_local / float(num_workers)
            t_norm_local = getNormTensor(t_sp_grad_local)
    
            gradients.append(t_sp_grad_local)
            norms.append(t_norm_local)

    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        tf.logging.info('Number of gradients received %d' % len(gradients))
        tf.logging.info('Number of norms received %d' % len(norms))
        g1 = tf.sparse_add(gradients[0], gradients[1])
        g2 = tf.sparse_add(gradients[2], gradients[3])
        # g3 = tf.sparse_add(gradients[4], g2)
        g = tf.sparse_add(g1, g2)
        t_w_update = tf.scatter_sub(w, tf.squeeze(g.indices), tf.expand_dims(g.values, 1))
        t_norm_update = tf.assign(curr_norm, tf.add_n(norms)/float(num_workers))
        

    # validation operator
    with tf.device("/job:worker/task:0"):
        # SETUP TESTING TENSORS: Every 10000 steps and with small dataset --> Takes ~10 seconds
        (t_predictions_test,t_predictions_test_p) = getValidationTensor(w, t_sp_features_batch_test, t_sp_index_batch_test, batch_size_test)
        # SETUP Validation TENSORS: Every 100000 and with full validation dataset --> Takes ~5 mins
        (t_predictions_validation,t_predictions_validation_p) = getValidationTensor(w, t_sp_features_batch_validation, t_sp_index_batch_validation, batch_size_validation)
        # Setup a tensor to save the model periodically
        t_saver = tf.train.Saver({"model": w})


    # START SESSION
    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://node0:2222", config=config) as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(0, num_batches):
            (norm_update, w_update) = sess.run([t_norm_update,t_w_update])
            num_examples_processed = (i*batch_size)
            batch_log = "Batch {0}, Examples Processed Per Worker: {1} ".format(i,num_examples_processed)
            print batch_log
            gradient_norm.append(norm_update)

            # Run Testing
            if(((num_examples_processed) % num_steps_per_test) == 0):
                print "Testing..."
                num_tp_all = 0 
                num_tn_all = 0
                num_fp_all = 0
                num_fn_all = 0
                for j in range(0,num_batches_test):
                    test_log = "Test Batch {0}, Test Ex Processed {1}".format(j,j*batch_size_test)
                    print test_log
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
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
