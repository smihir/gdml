import tensorflow as tf
import numpy as np
import os
import sys


sys.path.append("../")
from io_utils import *


def getAccuracy(num_correct_predictions, num_examples_criteo_test):
    accuracy = float(num_correct_predictions) / float(num_examples_criteo_test)
    error = (float(num_examples_criteo_test) - float(num_correct_predictions)) / float(num_examples_criteo_test)
    print "Accuracy: {0}, Error: {1}".format(accuracy, error)
    return (accuracy,error)

num_workers = 5

# GRADIENT DESCENT PARAMETERS
learning_rate = 0.01 # eta parameter
training_epochs = 1 # Number of passes through the whole dataset
batch_size = 1 # Number of examples read at once
batch_size_test = 1000 # Number of examples read at once
batch_size_validation = 10000 # Number of examples read at once

DATA_DIR="/home/ubuntu/workspace/criteo_data/"
# Files to read
filenames_criteo = [DATA_DIR + "tfrecords00"]
filenames_criteo_test = [DATA_DIR + "tfrecords22-tiny"]
filenames_criteo_validation = [DATA_DIR + "tfrecords22"]
num_features = 33762578 # Total number of features after one hot encoding

# I/O PARAMETERES
num_examples_criteo = 2000000
num_examples_criteo_test = 10000
num_examples_criteo_validation = 20000
num_batches = num_examples_criteo / batch_size # Number of training batches
num_batches_test = num_examples_criteo_test / batch_size_test # Number of test batches
num_batches_validation = num_examples_criteo_validation / batch_size_validation # Number of validation bataches

num_steps_per_test = 20
num_steps_per_validation = 40

g = tf.Graph()

with g.as_default():

    tf.logging.set_verbosity(tf.logging.INFO)
    # creating a model variable on task 0. This is a process running on node vm-48-1
    with tf.device("/job:worker/task:0"):
        # Measurement metrics
        loss_all = []
        gradient_norm_all = []
        test_accuracy_all = []
        test_error_all = []
        validation_accuracy_all = []
        validation_error_all = []    
        avg_time_per_batch = 0

        w = tf.Variable(tf.random_normal([num_features, 1], stddev=0.35), name="model")
        loss = tf.Variable(tf.zeros([1]), name="loss")
        assign_op = None
        # Create Validation Files Tensors - Only done in master
        (t_label_batch_test, t_sp_index_batch_test, t_sp_features_batch_test) = input_sp_tfrecord_criteo_pipeline(filenames_criteo, batch_size_test, num_features, num_epochs=None) 
        (t_label_batch_validation, t_sp_index_batch_validation, t_sp_features_batch_validation) = input_sp_tfrecord_criteo_pipeline(filenames_criteo, batch_size_validation, num_features, num_epochs=None) 



    # creating 5 reader operators to be placed on different operators
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    gradients = []
    avg_losses = []
    for i in range(0, num_workers):
        with tf.device("/job:worker/task:%d" % i):
            fnames = []
            for findex in range(i * num_workers, (i * num_workers) + num_workers):
                if findex > 21:
                    break
                fnames.append(DATA_DIR + "tfrecords%02d" % findex)

            print('node[{}]: process files: {}'.format(i, fnames))

            (t_label_batch, t_sp_index_batch, t_sp_features_batch) = input_sp_tfrecord_criteo_pipeline(fnames, batch_size, num_features, num_epochs=None)

            # # SETUP TRAINING TENSORS
            t_w_active = tf.gather(w, indices=t_sp_index_batch.values,name="w_active_"+str(i))
            # Shape for reshape  NEEDS to be an int32 for some reason (known TF issue)
            # t_shape = tf.cast(t_sp_index_batch.shape, tf.int32)    
    
            # Make a small dense matrix for each sparse w and active features. The dot product for each example will b e the diagonal elements of this small matrix
            t_w_local = tf.sparse_to_dense(t_sp_index_batch.indices, t_sp_index_batch.shape, tf.squeeze(t_w_active),  default_value=0, validate_indices=True, name="w_local_"+str(i))
            t_f_active = tf.sparse_to_dense(t_sp_index_batch.indices, t_sp_index_batch.shape, t_sp_features_batch.values,  default_value=0, validate_indices=True, name="f_active_"+str(i))
            # Compute gradient incrementally
            # Returns the diagonal elements of the matrix multiplication which is what we are interested in.
            t_dot = tf.diag_part(tf.matmul(t_w_local,t_f_active,transpose_b=True))
            t_dot = tf.reshape(t_dot,shape=[batch_size,1])
            t_sigmoid = tf.sigmoid(t_label_batch * t_dot)

            t_loss = -1 * tf.log(t_sigmoid)
            t_avg_loss = tf.reduce_mean(t_loss)
            avg_losses.append(t_avg_loss)
            t_grad = (t_label_batch * (t_sigmoid - 1)) # Vector of size batch_size x 1
            # The following multiplication broadcast each row of t_grad to each row of the sparse features.
            # For instance the first row of t_grad will multiply every element of the first example in the features batch
            t_sp_grad_local = t_grad * t_sp_features_batch
            t_sp_grad_local = tf.scalar_mul(learning_rate, t_sp_grad_local) 
    
            # Prepare sparse gradients to send back for batch
            # We have to reshape to a 1D vector.
            # # Place the real index values as the indeces
            t_sp_grad_local = tf.SparseTensor(indices=tf.expand_dims(t_sp_index_batch.values, 1), values=t_sp_grad_local.values, shape=[num_features])
            gradients.append(t_sp_grad_local)

    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        tf.logging.info('Number of gradients received %d' % len(gradients))
        for g in gradients:
            t_w_update = tf.scatter_sub(w, tf.squeeze(g.indices), tf.expand_dims(g.values, 1))


    # validation operator
    with tf.device("/job:worker/task:0"):
        # SETUP TESTING TENSORS: Every 10000 steps and with small dataset --> Takes ~10 seconds
        t_w_active_test = tf.gather(w, indices=t_sp_index_batch_test.values,name="w_active_test")
        t_w_active_test = tf.sparse_to_dense(t_sp_index_batch_test.indices, t_sp_index_batch_test.shape, tf.squeeze(t_w_active_test),  default_value=0, validate_indices=True, name="w_active_test")
        t_f_active_test = tf.sparse_to_dense(t_sp_index_batch_test.indices, t_sp_index_batch_test.shape, t_sp_features_batch_test.values,  default_value=0, validate_indices=True, name="f_active_test")
        t_predictions_test = tf.diag_part(tf.matmul(t_w_active_test,t_f_active_test,transpose_b=True))
        t_predictions_test = tf.reshape(t_predictions_test,shape=[batch_size_test,1])

        # SETUP Validation TENSORS: Every 100000 and with full validation dataset --> Takes ~10 mins
        t_w_active_validation = tf.gather(w, indices=t_sp_index_batch_validation.values,name="w_active_validation")
        t_w_active_validation = tf.sparse_to_dense(t_sp_index_batch_validation.indices, t_sp_index_batch_validation.shape, tf.squeeze(t_w_active_validation),  default_value=0, validate_indices=True, name="w_active_validation")
        t_f_active_validation = tf.sparse_to_dense(t_sp_index_batch_validation.indices, t_sp_index_batch_validation.shape, t_sp_features_batch_validation.values,  default_value=0, validate_indices=True, name="f_active_validation")
        t_predictions_validation = tf.diag_part(tf.matmul(t_w_active_validation,t_f_active_validation,transpose_b=True))
        t_predictions_validation = tf.reshape(t_predictions_validation,shape=[batch_size_validation,1])

        # Setup a tensor to save the model periodically
        t_saver = tf.train.Saver({"model": w})


    # START SESSION
    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://node0:2222", config=config) as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(0, num_batches):
            (w_update) = sess.run([t_w_update])
            num_examples_processed = (i*batch_size) * num_workers
            # if((num_examples_processed+1) % 100 == 0):
            print "Batch {0}, Examples Processed: {1} ".format(i,num_examples_processed)

            # Run Testing
            if(((num_examples_processed) % num_steps_per_test) == 0):
                print "Testing..."
                num_correct_predictions = 0
                for j in range(0,num_batches_test):
                    print "Test Batch {0}, Test Ex Processed {1}".format(j,j*batch_size_test)
                    (labels, predictions) = sess.run([t_label_batch_test, t_predictions_test])
                    correct_predictions_batch = (labels == np.sign(predictions))
                    num_correct_predictions += np.sum(correct_predictions_batch)
                (accuracy,error) = getAccuracy(num_correct_predictions, num_examples_criteo_test) 
                test_accuracy_all.append(accuracy)
                test_error_all.append(error)
                print "Batch {0}, Test Accuracy: {1}, Test Error: {2}".format(i,accuracy,error)


            # Run Validation
            if(((num_examples_processed) % num_steps_per_validation) == 0):
                print "Validating..."
                # Save test arrays
                np.save("output/SyncSGD_Test_Accuracy.npy",test_accuracy_all)
                np.save("output/SyncSGD_Test_Error.npy",test_error_all)
                # Save current model
                # t_saver.save(sess, "output/model_"+str(num_examples_processed)+".ckpt")
                np.save("output/model_"+str(num_examples_processed)+".npy",w_update)

                num_correct_predictions = 0
                for j in range(0,num_batches_validation):
                    print "Validation Batch {0}, Val Ex Processed {1}".format(j,j*batch_size_validation)
                    (labels, predictions) = sess.run([t_label_batch_validation, t_predictions_validation])
                    correct_predictions_batch = (labels == np.sign(predictions))
                    num_correct_predictions += np.sum(correct_predictions_batch)
                
                (accuracy,error) = getAccuracy(num_correct_predictions, num_examples_criteo_validation) 
                validation_accuracy_all.append(accuracy)
                validation_error_all.append(error)
                # Save validation arrays
                np.save("output/SyncSGD_Validation_Accuracy.npy",validation_accuracy_all)
                np.save("output/SyncSGD_Validation_Error.npy",validation_error_all)
                print "Batch {0}, Validation Accuracy: {1}, Validation Error: {2}".format(i,accuracy,error)                    

        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
