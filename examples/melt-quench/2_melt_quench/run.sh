#!/bin/bash
#SBATCH -N 4
##SBATCH -n 1
#SBATCH --time=5:00:00
#SBATCH --constraint=centos7
#SBATCH --mem=186G
#SBATCH --ntasks-per-node=40
#SBATCH -p sched_mit_ase
#SBATCH -o job-%x.out 
#SBATCH -e job-%x.err
#SBATCH -J melt

module purge
module load openmpi/gcc/64/1.8.1
module load gcc/8.3.0

#export PATH=$PATH:/home/rohskopf/modecode/src
export PATH=$PATH:/home/rohskopf/lammps/src
#source /home/rohskopf/venv3.6.10/bin/activate

mpirun -np $SLURM_NTASKS lmp_mpi < in.run

