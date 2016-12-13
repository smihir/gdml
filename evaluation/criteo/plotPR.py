import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

learning_rates = [0.005, 0.01, 0.05, 0.1, 0.5]
# learning_rates = [0.05]
# batch_size = [50, 100, 200, 400]
batch_size = 200
num_steps_per_test = 10000
num_examples = 2000000
num_batches = num_examples / batch_size 
batches = np.arange(0,num_examples,batch_size)
batches_sub = np.arange(0,num_batches,100)

print(batches[0:num_batches:100].shape)

for i in range(0,5):
	lr = learning_rates[i]
	description = str(batch_size) + "_" + str(lr)
	filename = "serial/output/Serial_Results_" + description + ".npz"
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
		e_fpr = results['e_fpr']
		e_tpr = results['e_tpr']

		plt.plot(e_precision,e_recall,label="learning rate = "+str(lr),linewidth=2.0)


plt.title("Precision-Recall of serial mini-batch sgd", fontsize=14,fontweight='bold')
plt.xlabel('Precision', fontsize=14,fontweight='bold')
plt.ylabel('Recall', fontsize=14,fontweight='bold')
plt.legend(loc='lower right',fontsize=14)
plt.show()


