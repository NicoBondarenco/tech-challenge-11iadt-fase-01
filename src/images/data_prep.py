
from __future__ import annotations

try:
    from .dataset import main
except ImportError:
    from dataset import main


if __name__ == "__main__":
    main()
