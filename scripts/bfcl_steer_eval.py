#!/usr/bin/env python3
"""A1 entry: BFCL held-out alpha sweep for tool-propensity F1."""

from bfcl_propensity import main

if __name__ == "__main__":
    import sys

    if "--phase" not in sys.argv:
        sys.argv.extend(["--phase", "eval"])
    main()
