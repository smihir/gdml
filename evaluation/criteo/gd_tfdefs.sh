#!/bin/bash
export TF_RUN_DIR="/home/ubuntu/run/"
export TF_BINARY_URL="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc2-cp27-none-linux_x86_64.whl"

function terminate_cluster() {
    echo "Terminating the servers"
    CMD="ps aux | grep -v 'grep' | grep -v 'bash' | grep -v 'ssh' | grep 'python startserver' | awk -F' ' '{print \$2}' | xargs kill -9"
    for i in `seq 0 4`; do
        ssh ubuntu@node$i "$CMD"
    done
}

function install_tensorflow() {
    pdsh -R ssh -w node[5-14] "sudo apt-get update"
    pdsh -R ssh -w node[5-14] "sudo apt-get install --assume-yes python-pip python-dev"
    pdsh -R ssh -w node[5-14] "sudo apt-get install python-numpy python-scipy python-sympy python-nose python-sklearn"
    pdsh -R ssh -w node[5-14] "sudo pip install --upgrade $TF_BINARY_URL"
}

function start_cluster() {
    if [ -z $1 ]; then
        echo "Usage: start_cluster <python script>"
        echo "Here, <python script> contains the cluster spec that assigns an ID to all server."
    else
        echo "Create $TF_RUN_DIR on remote hosts if they do not exist."
        pdsh -R ssh -w node[0-4] "mkdir -p $TF_RUN_DIR"
        echo "Copying the script to all the remote hosts."
        pdcp -R ssh -w node[0-4] $1 $TF_RUN_DIR

        echo "Starting tensorflow servers on all hosts based on the spec in $1"
        echo "The server output is logged to serverlog-i.out, where i = 1, ..., 5 are the VM numbers."
        nohup ssh ubuntu@node0 "cd /home/ubuntu/run ; python startserver.py --task_index=0" > output/serverlog-0.out 2>&1&
        nohup ssh ubuntu@node1 "cd /home/ubuntu/run ; python startserver.py --task_index=1" > output/serverlog-1.out 2>&1&
        nohup ssh ubuntu@node2 "cd /home/ubuntu/run ; python startserver.py --task_index=2" > output/serverlog-2.out 2>&1&
        nohup ssh ubuntu@node3 "cd /home/ubuntu/run ; python startserver.py --task_index=3" > output/serverlog-3.out 2>&1&
        nohup ssh ubuntu@node4 "cd /home/ubuntu/run ; python startserver.py --task_index=4" > output/serverlog-4.out 2>&1&
    fi
}
