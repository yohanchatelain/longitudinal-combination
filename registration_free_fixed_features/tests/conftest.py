from __future__ import annotations

from pathlib import Path
import sys


# Pytest may choose registration_free_fixed_features/ as rootdir when only this
# suite is selected. Add the repository parent so the package is importable in
# both focused and repository-wide test runs.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
