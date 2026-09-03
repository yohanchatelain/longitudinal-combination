#!/usr/bin/env bash
#SBATCH --job-name=rfff_pilot
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=registration_free_fixed_features/slurm/logs/pilot_%j.out
#SBATCH --error=registration_free_fixed_features/slurm/logs/pilot_%j.err

set -euo pipefail

project_root=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
python=${project_root}/registration_free_fixed_features/.venv/bin/python
pilot_dir=${project_root}/registration_free_fixed_features/pilot/ppmi

cd "${project_root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${python}" -m registration_free_fixed_features.cli extract \
    --config registration_free_fixed_features/config.pilot_gpu.yaml \
    --manifest "${pilot_dir}/development_manifest.csv" \
    --output "${pilot_dir}/visit_features.npz"
