import sys
from pathlib import Path
import warnings

# Ensure src/ is on the path so tests can import the package without installation
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Suppress environment-specific noise in CI/sandboxes
warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: Unexpected error.*",
    category=UserWarning,
    module=r"torch\.cuda.*",
)
