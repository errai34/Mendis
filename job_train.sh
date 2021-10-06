#!/bin/bash

#PBS -P y89
#PBS -q gpuvolta
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l place=scatter
#PBS -l mem=380GB
#PBS -l jobfs=200GB
#PBS -l walltime=48:00:00
#PBS -l wd

NODES=5
GPUS=4

module load use.own
module load cuda/10.1
module load intel-mkl/2019.3.199
module load python3/3.7.4
module load sklearn/0.24.1
module load pytorch/1.5.1

if [[ -z $MASTER_ADDR ]]; then
	export MASTER_ADDR=$(hostname -i)
	export MASTER_PORT=8888

	for i in $(seq 1 $(expr $NODES - 1)); do

		echo "Spwaning child $i"
		qsub -v MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,NIDX=$i job_train.sh
	done

	echo "This is master $MASTER_ADDR on port $MASTER_PORT of a total of $NODES nodes, each with $GPUS GPUS"
	python3 train.py -c config.json -n $NODES -g $GPUS > joint_train.log.0
else
	echo "This is child $NIDX of master $MASTER_ADDR on port $MASTER_PORT, from a total of $NODES nodes, each with $GPUS GPUS"
	python3 train.py -c config.json -n $NODES -g $GPUS -r $NIDX > joint_train.log.$NIDX
fi
