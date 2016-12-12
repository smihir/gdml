#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source ../tfdefs.sh

export GDML_PATH="/home/ubuntu/workspace/gdml/"
export GDML_CRITEO_PATH="${GDML_PATH}evaluation/criteo/"

pdsh -R ssh -w node[0-4] "mkdir -p ${GDML_CRITEO_PATH}asyncsgd/output/"

echo "Copying the script to all the remote hosts."
pdcp -R ssh -w node[0-4] asyncsgd.py ${GDML_CRITEO_PATH}asyncsgd/
pdcp -R ssh -w node[0-4] ../*.* ${GDML_CRITEO_PATH}


# startserver.py has the specifications for the cluster.
start_cluster ../startserver.py

echo "Executing the distributed tensorflow job from asynchronoussgd.py"


nohup python asyncsgd.py --task_index=0 > output/asynclog-0.out 2>&1&
sleep 2 # wait for variable to be initialized
nohup python asyncsgd.py --task_index=1 > output/asynclog-1.out 2>&1&
nohup python asyncsgd.py --task_index=2 > output/asynclog-2.out 2>&1&
nohup python asyncsgd.py --task_index=3 > output/asynclog-3.out 2>&1&
nohup python asyncsgd.py --task_index=4 > output/asynclog-4.out 2>&1&


# nohup python asynchronoussgd.py --task_index=0 &
# sleep 2 # wait for variable to be initialized
# nohup python asynchronoussgd.py --task_index=1 &
# nohup python asynchronoussgd.py --task_index=2 &
# nohup python asynchronoussgd.py --task_index=3 &
# nohup python asynchronoussgd.py --task_index=4 &

terminate_cluster