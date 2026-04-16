#!/usr/bin/env bash
#SBATCH --job-name=navigation_sft_phase1
#SBATCH --partition=preempt
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=peilab
#SBATCH --output=slurm-%j.out

srun \
  --container-image=nvidia+pytorch+25.01-py3.sqsh \
  --container-mounts=/project/peilab/atst:/project \
  --container-workdir=/project/VAGEN \
  bash -lc "NPROC_PER_NODE=2 LAUNCHER=torchrun DRY_RUN=false bash examples/train/navigation/train_sft_phase1.sh"