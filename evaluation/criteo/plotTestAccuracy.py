import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

learning_rates = [0.005, 0.01, 0.05, 0.1, 0.5]
learning_rate_syncsgd = 0.1
# learning_rates = [0.05]
# batch_size = [50, 100, 200, 400]
batch_size = 200

num_steps_per_test = 10000
num_steps_per_test_syncsgd = 25000
num_examples = 2000000
num_examples_syncsgd = 500000
num_examples_gdsyncsgd = 40000
num_batches = num_examples / batch_size 
num_batches_syncsgd = num_examples_syncsgd / batch_size 
num_batches_gdsyncsgd = num_examples_gdsyncsgd / batch_size 

test_batches = np.arange(0, num_examples, num_steps_per_test)
test_batches_syncsgd = np.arange(0, num_examples_syncsgd, num_steps_per_test_syncsgd)
# test_batches_syncsgd = np.arange(0, num_examples_syncsgd, num_steps_per_test_syncsgd)


# for i in range(0,5):
# 	lr = learning_rates[i]
# 	description = str(batch_size) + "_" + str(lr)
# 	filename = "serial/output/Serial_Results_" + description + ".npz"
# 	with np.load(filename) as results:

# 		test_accuracy = results['test_accuracy']
# 		validation_accuracy = results['validation_accuracy'].item(0)

# 		plt.plot(test_batches,test_accuracy,label="serial lr = "+str(lr),linewidth=2.0)



lr = learning_rate_syncsgd
description = str(batch_size) + "_" + str(lr)
filename = "syncsgd/output/SyncSGD_Results_" + description + ".npz"
with np.load(filename) as results:
	test_accuracy = results['test_accuracy']
	validation_accuracy = results['validation_accuracy'].item(0)
	plt.plot(test_batches_syncsgd,test_accuracy,label="dist sync lr = "+str(lr),linewidth=2.0)

lr = learning_rate_syncsgd
description = str(batch_size) + "_" + str(lr)
filename = "gdsyncsgd_naive/output/GDSyncSGDNaive_Results_" + description + ".npz"
with np.load(filename) as results:
	test_accuracy = results['test_accuracy']
	validation_accuracy = results['validation_accuracy'].item(0)
	plt.plot(40000,validation_accuracy,'*',label="gddist sync lr = "+str(lr),linewidth=3.0)




plt.title("Accuracy of serial mini-batch sgd", fontsize=14,fontweight='bold')
plt.xlabel('Num Examples Processed', fontsize=14,fontweight='bold')
plt.ylabel('Accuracy', fontsize=14,fontweight='bold')
plt.legend(loc='lower right',fontsize=14)
plt.show()


