#!/bin/bash
##NMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --account=122822881065
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train     #Set the job name to "JobExample4"
#SBATCH --time=40:00:00                #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --mem=16000M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=FPN.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu




module load Anaconda/3-5.0.0.1
source activate ML
python train.py
echo job done!