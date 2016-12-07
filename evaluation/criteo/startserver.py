"""
A simple script to start tensorflow servers with different roles.
"""
import tensorflow as tf

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
FLAGS = tf.app.flags.FLAGS

#
tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec = tf.train.ClusterSpec({
    "worker" : [
        "node0:2222",
        "node1:2222",
        "node2:2222",
        "node3:2222",
        "node4:2222"
    ]
})

server = tf.train.Server(clusterSpec, job_name="worker", task_index=FLAGS.task_index)
server.join()
