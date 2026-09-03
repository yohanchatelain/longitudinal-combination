#!/usr/bin/env bash
#SBATCH --job-name=rfff_gpu_smoke
# This cluster assigns GPUs by selecting the GPU partition; it has no GRES/TRES.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=registration_free_fixed_features/slurm/logs/gpu_smoke_%j.out
#SBATCH --error=registration_free_fixed_features/slurm/logs/gpu_smoke_%j.err

set -euo pipefail

project_root=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
python=${project_root}/registration_free_fixed_features/.venv/bin/python
output=${project_root}/registration_free_fixed_features/slurm/logs/gpu_smoke_${SLURM_JOB_ID}.json

if [[ ! -x ${python} ]]; then
    echo "Missing virtual-environment interpreter: ${python}" >&2
    exit 2
fi

cd "${project_root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${python}" -m registration_free_fixed_features.cli gpu-smoke \
    --config registration_free_fixed_features/config.example.yaml \
    --size 48 \
    --compare-cpu \
    --output "${output}"
