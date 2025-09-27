import sys
from pathlib import Path

from nodetool.config.logging_config import configure_logging

configure_logging("DEBUG")

# Ensure local src is on path so tests import local package
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Also add adjacent nodetool-core src if present (monorepo/dev setup)
CORE_SRC_PATH = Path(__file__).resolve().parents[2] / "nodetool-core" / "src"
if CORE_SRC_PATH.exists() and str(CORE_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_SRC_PATH))
