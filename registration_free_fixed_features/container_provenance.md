# Frozen external container

The local pilot uses `containers/synthstrip_1.8.sif`, built by Apptainer 1.4.2
from:

```text
docker://freesurfer/synthstrip@sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06
```

The local SIF is intentionally ignored because it is 382,615,552 bytes. Its
SHA-256 is locked in `slurm/pilot_synthstrip.sbatch`:

```text
fea95be33f3e2d4102d513349b2d95266c8528015e6b4d708d2abbcf91e1e462
```

Both the immutable upstream manifest digest and the exact executable SIF hash
are retained in generated provenance.
