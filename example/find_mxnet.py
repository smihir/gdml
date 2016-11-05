try:
	print "importing installed mxnet"
	import mxnet as mx
except ImportError:
	print "importing mxnet in parent dir"
	import os, sys
	curr_path = os.path.abspath(os.path.dirname(__file__))
	sys.path.append(os.path.join(curr_path, "../mxnet/python"))
	sys.path.append(os.path.join(curr_path, "../mxnet"))
	import mxnet as mx
