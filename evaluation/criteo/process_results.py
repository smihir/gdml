import numpy as np

# learning_rates = [0.01, 0.05, 0.1, 0.5]
learning_rates = [0.1]
# batch_size = [50, 100, 200, 400]
batch_size = [200]

description = str(batch_size[0]) + "_" + str(learning_rates[0])
filename = "serial/output/Serial_Results_" + description + ".npz"

with np.load(filename) as results:
	gradient_norm = results['gradient_norm']
	test_error = results['test_error']
	test_accuracy = results['test_accuracy']
	test_precision = results['test_precision']
	test_recall = results['test_recall']
	validation_error = results['validation_error'].item(0)
	validation_accuracy = results['validation_accuracy'].item(0)
	validation_precision = results['validation_precision'].item(0)
	validation_recall = results['validation_recall'].item(0)

	print "Norm Convergence: {}".format(gradient_norm)
	print "Test Error: {}".format(test_error)
	print "Test accuracy: {}".format(test_accuracy)
	print "Test precision: {}".format(test_precision)
	print "Test recall: {}".format(test_recall)
	print "Validation error: {}".format(validation_error)
	print "Validation accuracy: {}".format(validation_accuracy)
	print "Validation precision: {}".format(validation_precision)
	print "Validation recall: {}".format(validation_recall)
