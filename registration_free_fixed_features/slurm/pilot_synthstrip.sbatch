#!/usr/bin/env bash
#SBATCH --job-name=rfff_masks
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=registration_free_fixed_features/slurm/logs/masks_%j.out
#SBATCH --error=registration_free_fixed_features/slurm/logs/masks_%j.err

set -euo pipefail

project_root=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
python=${project_root}/registration_free_fixed_features/.venv/bin/python
pilot_dir=${project_root}/registration_free_fixed_features/pilot/ppmi
container=${project_root}/registration_free_fixed_features/containers/synthstrip_1.8.sif
container_sha256=fea95be33f3e2d4102d513349b2d95266c8528015e6b4d708d2abbcf91e1e462

cd "${project_root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

"${python}" -m registration_free_fixed_features.synthstrip_masks \
    --source-manifest "${pilot_dir}/development_manifest.csv" \
    --output-dir "${pilot_dir}" \
    --container "${container}" \
    --container-sha256 "${container_sha256}" \
    --config registration_free_fixed_features/config.pilot_gpu.yaml \
    --cpu \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
