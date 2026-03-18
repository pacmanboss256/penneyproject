from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CythonInplaceBuildHook(BuildHookInterface):
    """
    Hatchling build hook that compiles the Cython extensions before building.
    This is necessary to follow the constraint that we can't have any .py files in the top-level directory other than main.py
    (we'd need a setup.py file and a MANIFEST.in file otherwise)
    """

    def initialize(self, version: str, build_data: dict) -> None:
        project_root = Path(self.root)
        src_dir = project_root / "src"

        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(src_dir),
            check=True,
        )

