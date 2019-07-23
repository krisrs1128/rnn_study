#!/bin/bash
#################
#set a job name
#SBATCH --job-name=rnn_study
#################
#a file for job output, you can check job progress
#SBATCH --account=rpp-bengioy
#SBATCH --output=runn_study.out
#SBATCH --qos=high
#################
# a file for errors from the job
#SBATCH --error=run_study.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=00:30:00
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sankaran.kris@gmail.com

module load singularity
singularity shell --nv --bind /scratch/sankarak/data:/data,/home/sankarak/rnn_study:/home /scratch/sankarak/images/text.img /home/code/cluster/main.sh