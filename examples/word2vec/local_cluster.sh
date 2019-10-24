#!/bin/bash
echo "WARNING: This script only for run PaddlePaddle Fluid on one node"

if [ ! -d "./model" ]; then
  mkdir ./model
  echo "Create model folder for store infer model"
fi

if [ ! -d "./result" ]; then
  mkdir ./result
  echo "Create result folder for store training result"
fi

if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi

# environment variables for fleet distribute training
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=2
export PADDLE_PORT=36011,36012
export PADDLE_PSERVERS=127.0.0.1
export POD_IP=127.0.0.1
export CPU_NUM=2
export OUTPUT_PATH="output"
export SYS_JOB_ID="local_cluster"
export FLAGS_communicator_send_queue_size=1
export FLAGS_communicator_thread_pool_size=5
export FLAGS_communicator_max_merge_var_num=20
export FLAGS_communicator_fake_rpc=0

export PADDLE_PSERVER_PORTS=36011,36012
export PADDLE_PSERVER_PORT_ARRAY=(36011 36012)

export PADDLE_PSERVER_NUMS=2
export PADDLE_TRAINERS=2

training_method=$1
role=$2

if [[ ${training_method} = "local" ]]
then
    echo "Run Local Training"
    python -u model.py --training_method=${training_method} &> ./log/local_training.log &
fi

if [[ ${role} = "ps" && ${training_method}!="local" ]]
then
    export TRAINING_ROLE=PSERVER
    export GLOG_v=0
    export GLOG_logtostderr=1
    ps -ef|grep python|awk '{print $2}'|xargs kill -9

    for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
    do
        cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
        echo "PADDLE WILL START PSERVER "$cur_port
	      PADDLE_TRAINER_ID=$i
	      python -u model.py --training_method=${training_method} &> ./log/pserver.$i.log &
    done
fi

if [[ ${role} = "tr" && ${training_method}!="local" ]]
then
    export TRAINING_ROLE=TRAINER
    export GLOG_v=0
    export GLOG_logtostderr=1

    for((i=0;i<$PADDLE_TRAINERS;i++))
    do
        echo "PADDLE WILL START Trainer "$i
        PADDLE_TRAINER_ID=$i
	      python -u model.py --training_method=${training_method}  &> ./log/trainer.$i.log &
    done
fi
