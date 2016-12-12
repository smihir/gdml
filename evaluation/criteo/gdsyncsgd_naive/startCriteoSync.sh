#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
scp node0:/home/ubuntu/workspace/gdml/evaluation/criteo/gdsyncsgd_naive/* node10:/home/ubuntu/workspace/gdml/evaluation/criteo/gdsyncsgd_naive/
scp node0:/home/ubuntu/workspace/gdml/evaluation/criteo/*.* node10:/home/ubuntu/workspace/gdml/evaluation/criteo/

source ../gd_tfdefs.sh

# startserver.py has the specifications for the cluster.
start_cluster ../gd_startserver.py

# echo "Executing the distributed tensorflow job from criteoSync.py"
# # testdistributed.py is a client that can run jobs on the cluster.
# # please read testdistributed.py to understand the steps defining a Graph and
# # launch a session to run the Graph
time python gdsyncsgd.py

# # defined in tfdefs.sh to terminate the cluster
terminate_cluster
