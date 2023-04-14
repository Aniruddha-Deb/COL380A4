#!/bin/sh
### Set the job name (for your reference)
#PBS -N col380_a1_20cs10869
#PBS -o job.out
#PBS -e job.err

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

make apr1
make apr1_test
time ./approach_1 test/input2 apr1_output/output2
