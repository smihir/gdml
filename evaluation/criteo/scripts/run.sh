#!/bin/bash

SS_LOG_DIR="/home/ubuntu/ss"
function stop_sslogs() {
    echo "Terminating the servers"
    ps aux | grep -v 'grep' | grep -v 'bash' | grep 'collect.sh' | awk -F' ' '{print $2}' | xargs kill -9

    mkdir -p sslogs
    rm -rf sslogs/*
    for i in `seq 0 10`; do
	    scp -r ubuntu@node$i:$SS_LOG_DIR/sslogs/* sslogs/
    done
}

function start_sslogs() {
	echo "Create $SS_LOG_DIR on remote hosts if they do not exist."
	pdsh -R ssh -w node[0-10] "mkdir -p $SS_LOG_DIR"
	echo "Copying the script to all the remote hosts."
	pdcp -R ssh -w node[0-10] collect.sh $SS_LOG_DIR

	echo "Starting stat collection on all hosts based on the spec in $1"
	nohup ssh ubuntu@node0 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node1 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node2 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node3 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node4 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node5 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node6 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node7 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node8 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node9 "cd $SS_LOG_DIR ; ./collect.sh"&
	nohup ssh ubuntu@node10 "cd $SS_LOG_DIR ; ./collect.sh"&

}
