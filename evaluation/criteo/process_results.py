import numpy as np

# learning_rates = [0.01, 0.05, 0.1, 0.5]
learning_rates = [0.1]
# batch_size = [50, 100, 200, 400]
batch_size = [200]

description = str(batch_size[0]) + "_" + str(learning_rates[0])
filename = "syncsgd/output/SyncSGD_Results_" + description + ".npz"

with np.load(filename) as results:
	gradient_norm = results['gradient_norm']
	test_error = results['test_error']
	test_accuracy = results['test_accuracy']
	validation_error = results['validation_error'].item(0)
	validation_accuracy = results['validation_accuracy'].item(0)
	validation_precision = results['validation_precision'].item(0)
	validation_recall = results['validation_recall'].item(0)
	e_precision = results['e_precision']
	e_recall = results['e_recall']

	print "Norm Convergence: {}".format(gradient_norm)
	print "Test Error: {}".format(test_error)
	print "Test accuracy: {}".format(test_accuracy)
	print "Validation error: {}".format(validation_error)
	print "Validation accuracy: {}".format(validation_accuracy)
	print "Validation precision: {}".format(validation_precision)
	print "Validation recall: {}".format(validation_recall)
	print "e precision: {}".format(e_precision.shape)
	print "e recall: {}".format(e_recall.shape)
