#!/bin/bash

SS_LOG_DIR="/home/ubuntu/ss"
function stop_sslogs() {
    echo "Terminating the servers"
    #CMD="ps aux | grep -v 'grep' | grep -v 'bash' | grep -v 'ssh' | grep 'collect.sh' | awk -F' ' '{print \$2}' | xargs kill -9"
    ps aux | grep -v 'grep' | grep -v 'bash' | grep 'collect.sh' | awk -F' ' '{print $2}' | xargs kill -9

    mkdir -p sslogs
    rm -rf sslogs/*
    for i in `seq 1 5`; do
        #ssh ubuntu@node$i "$CMD"
	scp -r ubuntu@vm\-28\-$i:$SS_LOG_DIR/sslogs/* sslogs/
    done
}

function start_sslogs() {
	echo "Create $SS_LOG_DIR on remote hosts if they do not exist."
	pdsh -R ssh -w vm\-28\-[1-5] "mkdir -p $SS_LOG_DIR"
	echo "Copying the script to all the remote hosts."
	pdcp -R ssh -w vm\-28\-[1-5] collect.sh $SS_LOG_DIR

	echo "Starting stat collection on all hosts based on the spec in $1"
	nohup ssh ubuntu@vm\-28\-1 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@vm\-28\-2 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@vm\-28\-3 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@vm\-28\-4 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@vm\-28\-5 "cd $SS_LOG_DIR ; ./collect.sh"&
}
