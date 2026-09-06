#!/usr/bin/env python3
"""A0 entry: fit v_tool on BFCL select (Qwen3-1.7B, no LoRA)."""

from bfcl_propensity import main

if __name__ == "__main__":
    import sys

    if "--phase" not in sys.argv:
        sys.argv.extend(["--phase", "fit"])
    main()
