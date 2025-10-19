#!/bin/bash

sbatch cifar_exp1_lr.slurm
sbatch cifar_exp2_lr.slurm
sbatch cifar_exp3_lt.slurm
sbatch cifar_exp4.slurm
sbatch minst_exp1_lr.slurm
sbatch minst_exp2_lr.slurm
sbatch minst_exp3_lt.slurm
sbatch minst_exp4.slurm